import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

import wandb
import sys
import pathlib
ROOT_DIR = pathlib.Path(str(__file__)).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
# from tutorial_dataset import MyDataset
from tutorial_dataset_custom import MyDataset
from torch.utils.data import DataLoader


class ImageLogger(Callback):
    def __init__(self, dataset=None, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, test_data_root=None):
        super().__init__()
        self.dataset = dataset
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

        import datetime
        self.init_timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        # Init wandb test dataset and dataloader
        if test_data_root is None:
            raise ValueError("test_data_root must be provided for ImageLogger.")
        self.test_data_root = test_data_root
        self.test_dataset = MyDataset(data_root=self.test_data_root)
        self.test_dataloader = DataLoader(self.test_dataset, num_workers=0, batch_size=len(self.test_dataset.data), shuffle=False)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        # MODIFY: Save images to a specific directory structure
        root = os.path.join(save_dir, "image_log", self.init_timestamp, split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_wandb(self, pl_module, global_step, current_epoch, batch_idx, split="train"):
        batch = next(iter(self.test_dataloader))

        images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N]
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)

        single_mode = False
        merged_grids = []
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=1)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)

            if single_mode:
                remote_filename = "custom_{}_gs-{:06}_e-{:06}_b-{:06}".format(k, global_step, current_epoch, batch_idx)
                wandb.log(
                    data={f"[{remote_filename}]": wandb.Image(grid)}
                )
            else:
                merged_grids.append(grid)

        # After the loop, merge and upload
        if single_mode is False:
            merged = np.concatenate(merged_grids, axis=0)  # vertical stack
            remote_filename = "custom_gs-{:06}_e-{:06}_b-{:06}".format(global_step, current_epoch, batch_idx)
            wandb.log(
                data={f"[{remote_filename}]": wandb.Image(merged)}
            )

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            # NOTE: Log to wandb
            self.log_wandb(pl_module, pl_module.global_step, pl_module.current_epoch, 
                           batch_idx, split=split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")
            wandb.log({'train_loss': outputs['loss'].item()}, step=trainer.global_step)
