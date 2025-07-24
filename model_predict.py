from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

# Additional imports for prediction
import config
import os
import pathlib
import cv2
import einops
import numpy as np
import torch
import random
from PIL import Image
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3

# Root path for the project
root_path = str(pathlib.Path(__file__).parent)


def load_model(resume_path):
    """
    Load the ControlNet model from a checkpoint.
    Args:
        resume_path: Path to the model checkpoint.
    Returns:
        model: Loaded ControlNet model.
        ddim_sampler: DDIM sampler for image generation.
    """
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model_path = os.path.join(root_path, 'models/cldm_v21.yaml')
    model = create_model(model_path).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    print("Model successfully loaded!~")

    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler


def predict(model, ddim_sampler,    # model and sampler
            control_image, prompt,  # control image and prompt
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


def single_predict(model, ddim_sampler, 
                   test_img_path, test_prompt, test_output_path,
                   hyper_parameters):
    """
    Single image prediction function.
    """
    # Load a test image
    if os.path.exists(test_img_path):
        control_img = cv2.imread(test_img_path)
        control_img = cv2.cvtColor(control_img, cv2.COLOR_BGR2RGB)
        
        # Generate image
        results = predict(
            model=model,
            ddim_sampler=ddim_sampler,
            control_image=control_img,
            prompt=test_prompt,
            **hyper_parameters
        )
        
        # Save results
        for i, result in enumerate(results):
            input_stem = pathlib.Path(test_img_path).stem
            input_suffix = pathlib.Path(test_img_path).suffix
            output_filepath = os.path.join(test_output_path, f'{input_stem}_{i}{input_suffix}')
            cv2.imwrite(output_filepath, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        print(f"Generated {len(results)} images")
        return results
    
    # Fallback if test image is not found
    else:
        print(f"Test image not found at {test_img_path}")
        return None


def test_predict():
    ###########
    # Configs #
    ###########
    resume_path = './lightning_logs/version_5562633/checkpoints/epoch=22-step=274999.ckpt'
    test_img_path = '/home/ofirgila/PycharmProjects/ControlNet/my_images/test_circle.png'
    test_prompt = "red and green"
    test_output_path = './output_images/'

    hyper_parameters = dict(
        negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        added_prompt="best quality, extremely detailed",
        num_samples=1,
        image_resolution=512,
        ddim_steps=20,
        guess_mode=False,
        strength=1.0,
        scale=9.0,
        seed=-1,
        eta=0.0
    )

    ########
    # Flow #
    ########
    os.makedirs(test_output_path, exist_ok=True)
    model, ddim_sampler = load_model(resume_path)
    results = single_predict(
        model=model,
        ddim_sampler=ddim_sampler,
        test_img_path=test_img_path,
        test_prompt=test_prompt,
        test_output_path=test_output_path,
        hyper_parameters=hyper_parameters
    )
    if results is not None:
        print("Prediction completed successfully.")
    else:
        print("Prediction failed.")


if __name__ == "__main__":
    test_predict()
