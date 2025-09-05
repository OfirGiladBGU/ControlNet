from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset_custom import MyDataset
from cldm.logger_custom import ImageLogger
from cldm.model import create_model, load_state_dict

import wandb
import os
import datetime


# Configs
resume_path = './models/control_sd21_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

data_root = "../Stable_Diffusion/data"
dot_env_path = "./env"
max_epochs = 1  # Set the number of epochs for training

# Initialize WandB
def init_wandb(dot_env_path):
    if os.path.exists(dot_env_path):
        with open(dot_env_path, 'r') as f:
            for line in f:
                if line.startswith("WANDB_API_KEY"):
                    os.environ["WANDB_API_KEY"] = line.replace('WANDB_API_KEY=','').strip(" \"")
        API_KEY = os.environ.get("WANDB_API_KEY")
        wandb.login(key=API_KEY)

    wandb_project = "ControlNet"
    init_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    wandb_name = f"cn-sd21>{init_timestamp}"
    wandb.init(
        project=wandb_project,
        name=wandb_name,
        tags=["sd21", f"{max_epochs} epochs", f"{batch_size} batch size"]
    )
init_wandb(dot_env_path)


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset(data_root=data_root)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(dataset=dataset, batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_epochs=max_epochs)


# Train!
trainer.fit(model, dataloader)


###############################
# Kill the job after training #
###############################
import subprocess

def cancel_slurm_job(job_id):
    try:
        # Construct the command
        command = ["scancel", str(job_id)]
        
        # Execute the command
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Print the result
        print("Job canceled successfully:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Failed to cancel job:", e.stderr)

# Call the function with your job ID
# job_id = 5860013
# cancel_slurm_job(job_id)
