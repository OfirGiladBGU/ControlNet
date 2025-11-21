import json
import cv2
import numpy as np
import os

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data_root='./training/fill50k/'):
        self.data = []

        # Save the data root and ensure it does not end with a slash
        self.data_root = data_root
        if self.data_root.endswith('/'):
            self.data_root = self.data_root[:-1]
        
        with open(os.path.join(self.data_root, 'prompt.json'), 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join(self.data_root, source_filename))
        target = cv2.imread(os.path.join(self.data_root, target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


class DynamicMyDataset(Dataset):
    def __init__(self, image_list, prompt_list):
        self.data = []
        
        for source_path, prompt in zip(image_list, prompt_list):
            item = {
                'source': source_path,
                'target': source_path,  # Dummy target; replace as needed
                'prompt': prompt
            }
            self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source']
        target_filepath = item['target']
        prompt = item['prompt']

        source = cv2.imread(source_filepath)
        target = cv2.imread(target_filepath)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
