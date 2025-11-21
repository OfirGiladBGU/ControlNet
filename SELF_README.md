# Self Notes

## Custom Scripts

For custom training, the following are used:
- [logger_custom.py](cldm/logger_custom.py) - For exporting results both locally and to WANDB
- [tutorial_train_sd21_custom.py](tutorial_train_sd21_custom.py) - For training using the "logger_custom"
- [train_in_background.py](train_in_background.py) - Running the training in the background, without the need to hold interactive session.


For custom prediction, the following are used:
- [model_predict.py](model_predict.py) - With interactive mode to select image to run


IMPORTANT:
- [logger_custom.py](cldm/logger_custom.py) needs a path `x` for a test dataset similar to `fill50k` example. \
  Example for setup before training:
    - `training/fill50k`
    - `training/fill50k_test`

## Notices:

- Training local test images are exported to the folder: [image_log/train](image_log/<date>/train).
- Trained model weights are exported to the folder: [lightning_logs/version_XXXXXXX](lightning_logs)

## Notes about ControlNet 

- Training step location: [ddpm.py](ldm/models/diffusion/ddpm.py) -> 
  ```
  def training_step(self, batch, batch_idx)
  ```
- Highest level model warpper: [cldm.py](cldm/cldm.py)  -> 
  ```
  class ControlLDM(LatentDiffusion)
  ```
- The MyDataset dict keys: [tutorial_dataset.py](tutorial_dataset.py) -> 
  ```
  return dict(jpg=target, txt=prompt, hint=source)
  ``` 
  come from: [cldm_v21.yaml](models/cldm_v21.yaml) / [cldm_v15.yaml](models/cldm_v15.yaml) -> 
  ```
  first_stage_key: "jpg"
  cond_stage_key: "txt"
  control_key: "hint"
  ```
