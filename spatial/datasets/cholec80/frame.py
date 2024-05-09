import os, accimage
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision
torchvision.set_image_backend('accimage')
from torchvision import transforms


phase_dict_key = {
    'Preparation': 0, 
    'CalotTriangleDissection': 1, 
    'ClippingCutting': 2, 
    'GallbladderDissection': 3, 
    'GallbladderPackaging': 4, 
    'CleaningCoagulation': 5, 
    'GallbladderRetraction': 6
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, label_dir, transforms, sample_rate):
        super().__init__()
        self.transforms = transforms

        self.samples = []
        for video_name in tqdm(os.listdir(frame_dir)):
            if video_name.startswith("."): # ignore .ipynb_checkpoints
                continue
            
            # get label
            label_path = os.path.join(label_dir, f"{video_name}-phase.txt")
            label_dict = {}
            f = open(label_path)
            lines = f.readlines()[1:] # first line is "Frame\tPhase"
            f.close()
            for line in lines:
                frame_idx, label = line.strip().split("\t")
                label_dict[int(frame_idx)] = phase_dict_key[label]

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


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    frame_dir = "/root/dataspace/cholec80/frames"
    label_dir = "/root/dataspace/cholec80/phase_annotations"

    data_transforms = transforms.Compose([
        transforms.CenterCrop((306, 544)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])

    dataset = Dataset(frame_dir, label_dir, data_transforms, 25)
    data_loader = DataLoader(dataset, batch_size=64, num_workers=6, shuffle=False, pin_memory=True)
    print(data_loader)
    print("total iters", len(data_loader))
    
    c = 0
    for inputs, labels, video_name, frame_idx in tqdm(data_loader):
        print("inputs", inputs.cpu().numpy().shape)
        print("labels", labels.cpu().numpy().shape)
        print(labels.cpu().numpy())
        print("video_name", video_name)
        print("frame_idx", frame_idx)

        c += 1
        if c == 10:
            break


