from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Additional imports for prediction
import config
import os
import cv2
import einops
import numpy as np
import torch
import random
from PIL import Image
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3


def predict(control_image, prompt, 
           negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
           added_prompt="best quality, extremely detailed",
           num_samples=1,
           image_resolution=512,
           ddim_steps=20,
           guess_mode=False,
           strength=1.0,
           scale=9.0,
           seed=-1,
           eta=0.0):
    """
    Generate images using ControlNet model.
    
    Args:
        control_image: Input control image (numpy array or PIL Image)
        prompt: Text prompt for generation
        negative_prompt: Negative prompt to avoid unwanted features
        added_prompt: Additional positive prompt
        num_samples: Number of images to generate
        image_resolution: Output image resolution
        ddim_steps: Number of sampling steps
        guess_mode: Whether to use guess mode
        strength: Control strength
        scale: Guidance scale
        seed: Random seed (-1 for random)
        eta: DDIM eta parameter
    
    Returns:
        List of generated images as numpy arrays
    """
    with torch.no_grad():
        # Preprocess control image
        if isinstance(control_image, Image.Image):
            control_image = np.array(control_image)
        
        img = resize_image(HWC3(control_image), image_resolution)
        H, W, C = img.shape
        
        # Convert control image to tensor
        control = torch.from_numpy(img.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        # Set random seed
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        
        # Memory optimization
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        # Prepare conditioning
        full_prompt = prompt + ', ' + added_prompt if added_prompt else prompt
        cond = {
            "c_concat": [control], 
            "c_crossattn": [model.get_learned_conditioning([full_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control], 
            "c_crossattn": [model.get_learned_conditioning([negative_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)
        
        # Memory optimization
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)
        
        # Set control scales
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        
        # Sample
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples,
            shape, cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )
        
        # Memory optimization
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)
        
        # Decode latents to images
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        results = [x_samples[i] for i in range(num_samples)]
        return results


def predict_batch(control_images, prompts, **kwargs):
    """
    Generate images for multiple control images and prompts.
    
    Args:
        control_images: List of control images
        prompts: List of prompts (should match length of control_images)
        **kwargs: Additional arguments for predict function
    
    Returns:
        List of lists containing generated images for each input
    """
    if len(control_images) != len(prompts):
        raise ValueError("Number of control images must match number of prompts")
    
    results = []
    for control_img, prompt in zip(control_images, prompts):
        result = predict(control_img, prompt, **kwargs)
        results.append(result)
    
    return results


# Example usage function
def example_usage():
    """
    Example of how to use the prediction function
    """
    # Load a test image
    test_img_path = '/home/ofirgila/PycharmProjects/ControlNet/training/fill50k/source/0.png'
    test_prompt = "blue and yellow"
    if os.path.exists(test_img_path):
        control_img = cv2.imread(test_img_path)
        control_img = cv2.cvtColor(control_img, cv2.COLOR_BGR2RGB)
        
        # Generate image
        results = predict(
            control_image=control_img,
            prompt=test_prompt,
            num_samples=2,
            ddim_steps=20,
            scale=9.0
        )
        
        # Save results
        for i, result in enumerate(results):
            cv2.imwrite(f'generated_image_{i}.png', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        print(f"Generated {len(results)} images")
        return results
    else:
        print(f"Test image not found at {test_img_path}")
        return None


if __name__ == "__main__":
    # Configs
    resume_path = './lightning_logs/version_5562633/checkpoints/epoch=3-step=49999.ckpt'

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v21.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    print("Model successfully loaded!~")

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    example_usage()
