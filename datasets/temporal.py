import os, time, pickle, argparse, copy, random, numbers, cv2, accimage
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn


# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, feature_dir):
#         super().__init__()

#         self.samples = []
#         for f_name in tqdm(os.listdir(feature_dir)):
#             if not f_name.endswith("-labels.npy"):
#                 continue

#             video_name = f_name.split("-")[0]

#             # get label
#             label_path = os.path.join(feature_dir, f"{video_name}-labels.npy")
#             labels = np.load(label_path)

#             # get image
#             video_path = os.path.join(feature_dir, f"{video_name}-features.npy")
#             features = np.load(video_path)

#             self.samples.append([features, labels, video_name])
#         return


#     def __getitem__(self, index):
#         return self.samples[index]


#     def __len__(self):
#         return len(self.samples)





class Dataset(torch.utils.data.Dataset):
    def __init__(self, feature_base, training=True):
        super().__init__()

        self.feature_base = feature_base

        self.channel_mask = False
        self.channel_mask_rate = 0.5
        self.channel_mask_layer = nn.Dropout2d(p=self.channel_mask_rate)

        self.time_mask = False
        self.num_time_masks = 3
        self.time_mask_rate = 0.5
        self.max_time_mask_len = 0.1
        self.replace_with_zero = True
        if training:
            self.channel_mask = True
            self.time_mask = True

        self.training = training

        self.samples = []
        for f_name in tqdm(os.listdir(self.feature_base)):
            if not f_name.endswith("-labels.npy"):
                continue

            video_name = f_name.split("-")[0]
            # get label
            label_path = os.path.join(self.feature_base, f"{video_name}-labels.npy")
            labels = np.load(label_path)

            # get image
            video_path = os.path.join(self.feature_base, f"{video_name}.npy")
            features = np.load(video_path)

            self.samples.append([features, labels, video_name])
        return


    def __getitem__(self, index):
        features, labels, video_name = self.samples[index]  # features = [T, D]
        features = torch.from_numpy(features).float()
        T, D = features.shape

        if self.channel_mask: # 数据增强, 把所有时间上某一个channel的值全部置0
            features = features.unsqueeze(0).permute(0, 2, 1) # [T, D] -> [B, T, D] -> [B, D, T]
            features = self.channel_mask_layer(features)
            features = features.squeeze(0).permute(1, 0) # [B, D, T] -> [T, D]

        if self.time_mask: # 数据增强, 随机把某个连续时间片段上的内容全部置0
            for i in range(self.num_time_masks):
                if random.random() > self.time_mask_rate:
                    continue
                
                mask_len = random.randrange(1, int(T * self.max_time_mask_len))
                mask_start = random.randrange(0, T - mask_len)

                if self.replace_with_zero: 
                    features[mask_start:mask_start+mask_len] = 0
                else: 
                    features[mask_start:mask_start+mask_len] = features.mean()
            
        return [features, labels, video_name]


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
