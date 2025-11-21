from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
# from cldm.logger import ImageLogger
from tutorial_dataset_custom import MyDataset, DynamicMyDataset
from cldm.logger_custom import ImageLogger
from cldm.model import create_model, load_state_dict

# Additional imports for prediction
from PIL import Image
import os
import pathlib
import numpy as np
import torch

# Root path for the project
root_path = str(pathlib.Path(__file__).parent)


def load_model(resume_path):
    """
    Load the ControlNet model from a checkpoint.
    Args:
        resume_path: Path to the model checkpoint.
    Returns:
        model: Loaded ControlNet model.
    """
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model_path = os.path.join(root_path, 'models/cldm_v21.yaml')
    model = create_model(model_path).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    print("Model successfully loaded!~")

    model = model.cuda()
    is_train = model.training
    if is_train:
        model.eval()
    return model


def predict(model, control_image_path, prompt):
    """
    Generate images using ControlNet model.
    
    Args:
        control_image_path: Path to the input control image
        prompt: Text prompt for generation
    Returns:
        List of generated images as numpy arrays
    """
    # Preprocess control image
    dataset = DynamicMyDataset([control_image_path], [prompt])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    with torch.no_grad():
        images = model.log_images(batch)

    clamp = True
    for k in images:
        N = images[k].shape[0]
        images[k] = images[k][:N]
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().cpu()
            if clamp:
                images[k] = torch.clamp(images[k], -1., 1.)

    # root = os.path.join(save_dir, "image_log", self.init_timestamp, split)
    rescale = True
    results = []
    for k in images:
        if "samples" in k:
            samples = images[k]
            if rescale:
                samples = (samples + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            samples = samples.transpose(1, 2).transpose(2, 3)  # b,c,h,w -> b,h,w,c
            samples = samples.numpy()
            samples = (samples * 255).astype(np.uint8)
            if "samples" in k:
                results.extend([samples[i] for i in range(samples.shape[0])])

    return results


def single_predict(model, test_img_path, test_prompt, test_output_path):
    """
    Single image prediction function.
    """
    # Load a test image
    if os.path.exists(test_img_path):
        # Generate image
        results = predict(
            model=model,
            control_image_path=test_img_path,
            prompt=test_prompt,
        )
        
        # Save results
        for i, result in enumerate(results):
            input_stem = pathlib.Path(test_img_path).stem
            input_suffix = pathlib.Path(test_img_path).suffix
            output_filepath = os.path.join(test_output_path, f'{input_stem}_{i}{input_suffix}')
            Image.fromarray(result).save(output_filepath)
        
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
    test_img_path = './my_inputs/test_circle.png'
    test_prompt = "red and green"
    test_output_path = './my_outputs/'

    ########
    # Flow #
    ########
    os.makedirs(test_output_path, exist_ok=True)
    model = load_model(resume_path)
    results = single_predict(
        model=model,
        test_img_path=test_img_path,
        test_prompt=test_prompt,
        test_output_path=test_output_path
    )
    if results is not None:
        print("Prediction completed successfully.")
    else:
        print("Prediction failed.")


def test_predict_online():
    resume_path = './lightning_logs/version_5562633/checkpoints/epoch=22-step=274999.ckpt'
    test_output_path = './my_outputs/'

    ########
    # Flow #
    ########
    os.makedirs(test_output_path, exist_ok=True)
    model = load_model(resume_path)

    online_flag = True
    while online_flag:
        test_img_path = input("Enter the path to the test image:\n")
        if test_img_path.lower().strip() == 'exit':
            print("Exiting the prediction loop.")
            break

        test_prompt = input("Enter the prompt for image generation:\n")
        if test_prompt.lower().strip() == 'exit':
            print("Exiting the prediction loop.")
            break
        
        results = single_predict(
            model=model,
            test_img_path=test_img_path,
            test_prompt=test_prompt,
            test_output_path=test_output_path
        )
        if results is not None:
            print("Prediction completed successfully.")
        else:
            print("Prediction failed.")


if __name__ == "__main__":
    test_predict()
    # test_predict_online()
    # TODO: Predict folder
