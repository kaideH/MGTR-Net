import os, accimage
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision
torchvision.set_image_backend('accimage')
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, label_dir, transforms, sample_rate, get_label, sub_set=None):
        super().__init__()
        self.transforms = transforms

        self.samples = []
        for video_name in tqdm(os.listdir(frame_dir)):
            if sub_set and video_name not in sub_set:
                continue
            if video_name.startswith("."): # ignore .ipynb_checkpoints
                continue
            
            # get label
            label_path = os.path.join(label_dir, f"{video_name}-phase.txt")
            label_dict = get_label(label_path)

            # get image
            video_path = os.path.join(frame_dir, video_name)
            for frame_name in os.listdir(video_path):
                if frame_name.startswith("."): # ignore .ipynb_checkpoints
                    continue
                frame_idx = int(frame_name.split(".")[0])
                if int(frame_idx) % sample_rate != 0:
                    continue
                img_path = os.path.join(video_path, frame_name)
                label = label_dict[frame_idx]
                self.samples.append([img_path, label, video_name, frame_idx])
        return

    def __getitem__(self, index):
        img_path, label, video_name, frame_idx = self.samples[index]
        img = accimage.Image(img_path)
        img = self.transforms(img)
        return img, label, video_name, frame_idx

    def __len__(self):
        return len(self.samples)



