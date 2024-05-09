import os, time, pickle, argparse, copy, random, numbers, cv2, accimage
import numpy as np
from tqdm import tqdm

import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature_dir):
        super().__init__()

        self.samples = []
        for f_name in tqdm(os.listdir(feature_dir)):
            if not f_name.endswith("-labels.npy"):
                continue

            video_name = f_name.split("-")[0]

            # get label
            label_path = os.path.join(feature_dir, f"{video_name}-labels.npy")
            labels = np.load(label_path)

            # get image
            video_path = os.path.join(feature_dir, f"{video_name}-features.npy")
            features = np.load(video_path)

            self.samples.append([features, labels, video_name])
        return


    def __getitem__(self, index):
        return self.samples[index]


    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":

    from torch.utils.data import DataLoader
    
    dataset = Dataset("/root/workspace/phaseRecognition/MGTR-Net/features/cholec80/train")
    print(len(dataset))
    loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=True, pin_memory=True)
    print(len(loader))

    for epoch in range(3):
        print("epoch", epoch)

        for data in tqdm(loader):
            features, labels, video_name = data
            # print(features.shape[1], labels.shape[1], video_name)
            print(features.shape, labels.shape, video_name)
            # print(features)
