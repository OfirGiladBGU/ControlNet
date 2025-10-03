# Self Notes

For custom training, the following are used:
- [logger_custom.py](cldm/logger_custom.py) - For exporting results both locally and to WANDB
- [tutorial_train_sd21_custom.py](tutorial_train_sd21_custom.py) - For training using the "logger_custom"
- [train_in_background.py](train_in_background.py) - Running the training in the background, without the need to hold interactive session.


For custom prediction, the following are used:
- [model_predict.py](model_predict.py) - With interactive mode to select image to run


IMPORTANT:
- [logger_custom.py](cldm/logger_custom.py) assumes that a '<dataset>_test' folder exists, for `fill50k` example, the following folders are required:
    - `training/fill50k`
    - `training/fill50k_test`

## Notices:

- Training local test images are exported to the folder: [image_log/train](image_log/<date>/train).
- Trained model weights are exported to the folder: [lightning_logs/version_XXXXXXX](lightning_logs)