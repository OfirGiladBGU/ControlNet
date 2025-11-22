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
from tqdm import tqdm

# Root path for the project
root_path = str(pathlib.Path(__file__).parent)


########
# Core #
########
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


def predict(model, image_path_list, prompt_list, 
            output_folder_path=None, return_all_results=False, 
            hyper_params=None):
    """
    Generate images using ControlNet model.
    
    Args:
        image_path_list: Path to the input control image
        prompt_list: Text prompt for generation
        output_folder_path: Folder to save generated images
        return_all_results: Whether to return all generated images
        hyper_params: Dictionary of hyper parameters
    Returns:
        List of generated images as numpy arrays
    """
    # Parse hyper parameters
    if hyper_params is None:
        hyper_params = {}
    batch_size = hyper_params.get('batch_size', 1)
    sample = hyper_params.get('sample', False)
    ddim_steps = hyper_params.get('ddim_steps', 50)
    unconditional_guidance_scale = hyper_params.get('unconditional_guidance_scale', 9.0)

    # Preprocess control image
    dataset = DynamicMyDataset(image_path_list, prompt_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Loop params
    clamp = True
    rescale = True

    all_results = []
    image_idx = -1
    for idx, batch in enumerate(dataloader): 
        print(f"Processing batch {idx+1}/{len(dataloader)}...")

        results = []
        with torch.no_grad():
            # images = model.log_images(batch, N=batch_size)
            images = model.custom_log_images(
                batch, 
                N=batch_size, 
                sample=sample, 
                ddim_steps=ddim_steps, 
                unconditional_guidance_scale=unconditional_guidance_scale
            )

        # clamp = True
        for k in images:
            N = images[k].shape[0]
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)

        # rescale = True
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
        
        # Aggregate results
        if return_all_results:
            all_results.extend(results)
        # Export results
        else:
            if output_folder_path is None:
                raise ValueError("output_folder_path must be specified when return_all_results is False.")
            
            for result in tqdm(results):
                image_idx += 1
                input_name = pathlib.Path(image_path_list[image_idx]).name
                output_filepath = os.path.join(output_folder_path, str(input_name))
                Image.fromarray(result).save(output_filepath)

    if return_all_results:
        return all_results
    else:
        results_count = image_idx + 1
        return results_count


#####################
# Single Prediction #
#####################
def single_predict(model, image_path, prompt, output_folder_path, 
                   hyper_params=None):
    """
    Single image prediction function.
    """
    # Load a test image
    if os.path.exists(image_path):
        # Generate image
        results = predict(
            model=model,
            image_path_list=[image_path],
            prompt_list=[prompt],
            return_all_results=True,
            hyper_params=hyper_params
        )
        
        # Save results (Single image case)
        result = results[0]
        input_name = pathlib.Path(image_path).name
        output_filepath = os.path.join(output_folder_path, input_name)
        Image.fromarray(result).save(output_filepath)
        print(f"Generated image at {output_filepath}")
        return result
    
    # Fallback if test image is not found
    else:
        print(f"Test image not found at {image_path}")
        return None


def test_predict():
    ###########
    # Configs #
    ###########
    resume_path = './lightning_logs/version_5562633/checkpoints/epoch=22-step=274999.ckpt'
    test_image_path = './my_inputs/test_circle.png'
    test_prompt = "red and green"
    test_output_folder_path = './my_outputs/'

    # Hyper parameters
    hyper_params = dict(
        batch_size=1,
    )

    ########
    # Flow #
    ########
    os.makedirs(test_output_folder_path, exist_ok=True)
    model = load_model(resume_path)
    result = single_predict(
        model=model,
        image_path=test_image_path,
        prompt=test_prompt,
        output_folder_path=test_output_folder_path,
        hyper_params=hyper_params
    )
    if result is not None:
        print("Prediction completed successfully.")
    else:
        print("Prediction failed.")


def test_predict_online():
    resume_path = './lightning_logs/version_5562633/checkpoints/epoch=22-step=274999.ckpt'
    test_output_folder_path = './my_outputs/'

    # Hyper parameters
    hyper_params = dict(
        batch_size=1,
    )

    ########
    # Flow #
    ########
    os.makedirs(test_output_folder_path, exist_ok=True)
    model = load_model(resume_path)

    online_flag = True
    while online_flag:
        test_image_path = input("Enter the path to the test image:\n")
        if test_image_path.lower().strip() == 'exit':
            print("Exiting the prediction loop.")
            break

        test_prompt = input("Enter the prompt for image generation:\n")
        if test_prompt.lower().strip() == 'exit':
            print("Exiting the prediction loop.")
            break
        
        result = single_predict(
            model=model,
            image_path=test_image_path,
            prompt=test_prompt,
            output_folder_path=test_output_folder_path,
            hyper_params=hyper_params
        )
        if result is not None:
            print("Prediction completed successfully.")
        else:
            print("Prediction failed.")


##########################
# Multi-image Prediction #
##########################
def folder_predict(model, input_folder_path, prompt, output_folder_path, 
                   hyper_params=None):
    """
    Multi-image prediction function.
    """
    if os.path.exists(input_folder_path):
        image_path_list = sorted(pathlib.Path(input_folder_path).glob('*.*'))
        prompt_list = [prompt] * len(image_path_list)
        
        # Generate image
        results_count = predict(
            model=model,
            image_path_list=image_path_list,
            prompt_list=prompt_list,
            output_folder_path=output_folder_path,
            return_all_results=False,
            hyper_params=hyper_params
        )
        
        return results_count
    
    # Fallback if test image is not found
    else:
        print(f"Input folder does not exist at {input_folder_path}")
        return None


def test_predict_folder():
    ###########
    # Configs #
    ###########
    resume_path = './lightning_logs/version_8280725/checkpoints/epoch=1-step=19999.ckpt'
    test_input_folder = './training/data_grads_v3/source'
    test_prompt = "Stippling"
    test_output_path = './training/data_grads_v3/output'

    # Hyper parameters
    hyper_params = dict(
        # batch_size=16,  # Max for RTX 6000
        batch_size=4,
        sample=False,
        ddim_steps=10,
        unconditional_guidance_scale=9.0,
    )

    ########
    # Flow #
    ########
    os.makedirs(test_output_path, exist_ok=True)
    model = load_model(resume_path)

    results_count = folder_predict(
        model=model,
        input_folder_path=test_input_folder,
        prompt=test_prompt,
        output_folder_path=test_output_path,
        hyper_params=hyper_params,
    )
    if results_count is not None:
        print("Folder prediction completed successfully.")
    else:
        print("Folder prediction failed.")


if __name__ == "__main__":
    # test_predict()
    # test_predict_online()
    test_predict_folder()
