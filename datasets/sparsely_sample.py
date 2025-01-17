import os, accimage, random, numbers
from tqdm import tqdm
from itertools import groupby

import torch
from torch.utils.data import DataLoader
import torchvision
torchvision.set_image_backend('accimage')
from torchvision import transforms
from torchvision.transforms import Lambda


class Dataset(torch.utils.data.Dataset):
    def __init__(self, transforms, image_base, label_base, sample_rate, sub_set, N, L, get_label, shuffle=False):
        super().__init__()

        self.image_base = image_base
        self.label_base = label_base
        self.sample_rate = sample_rate
        self.transforms = transforms

        self.N = N
        self.L = L
        self.shuffle = shuffle

        self.get_label = get_label
        self.sub_set = sub_set

        self.samples = {}
        for video_name in tqdm(os.listdir(self.image_base)):
            if video_name not in self.sub_set:
                continue
            
            # get label
            label_path = os.path.join(self.label_base, f"{video_name}-phase.txt")
            label_dict = self.get_label(label_path)

            # get all frames
            total_frames = []
            total_labels = []
            total_indexs = []
            video_path = os.path.join(self.image_base, video_name)
            for frame_idx in range(len(os.listdir(video_path))):
                if frame_idx % self.sample_rate != 0:
                    continue

                frame_name = f"{frame_idx}.jpg"
                img_path = os.path.join(video_path, frame_name)
                label = label_dict[frame_idx]

                total_frames.append(img_path)
                total_labels.append(label)
                total_indexs.append(frame_idx)

            self.samples[video_name] = {
                "total_frames": total_frames,
                "total_labels": total_labels,
                "total_indexs": total_indexs
            }
        self.reset_buffer()
        return 


    def reset_buffer(self):
        self.clip_buffer = []
        for video_name in self.sub_set:
            total_frames = self.samples[video_name]["total_frames"]
            total_labels = self.samples[video_name]["total_labels"]
            total_indexs = self.samples[video_name]["total_indexs"]

            label_groups = []
            group_indexes = []
            for idx, label in enumerate(total_labels):
                if not group_indexes or label != total_labels[group_indexes[-1]]:
                    if group_indexes:
                        label_groups.append([total_labels[group_indexes[0]], group_indexes])
                    group_indexes = []
                group_indexes.append(idx)
            if group_indexes: 
                label_groups.append([total_labels[group_indexes[0]], group_indexes])

            for label, indices in label_groups:
                phase_frames = [total_frames[i] for i in indices]
                phase_labels = [total_labels[i] for i in indices]
                phase_indexs = [total_indexs[i] for i in indices]

                clip_list = []
                start_idx = random.randrange(0, self.L)
                for i in range(start_idx, len(phase_frames)-self.L+1, self.L):
                    imgs = phase_frames[i:i+self.L]
                    labels = phase_labels[i:i+self.L]
                    frame_idx = phase_indexs[i:i+self.L]
                    clip_list.append([imgs, labels, frame_idx])

                if self.shuffle and random.random() <= 0.8:
                    random.shuffle(clip_list)
                
                while len(clip_list) != 0:
                    N_inputs, N_labels, N_frame_ids = [], [], []
                    for i in range(self.N):
                        if len(clip_list) == 0:
                            break
                        imgs, labels, frame_idx = clip_list.pop(0)
                        N_inputs.append(imgs)
                        N_labels.append(labels) # [T]
                        N_frame_ids.append(frame_idx) # [T]

                    self.clip_buffer.append([N_inputs, N_labels, video_name, N_frame_ids])
        return


    def __getitem__(self, index):
        N_images, N_labels, video_name, N_frame_ids = self.clip_buffer[index]

        N_inputs = []
        for clip_imgs in N_images:
            clip_imgs = [accimage.Image(path) for path in clip_imgs]
            clip_imgs = self.transforms(clip_imgs) 
            N_inputs.append(clip_imgs)
        N_inputs = torch.stack(N_inputs, dim=0)

        return N_inputs, torch.tensor(N_labels), [video_name] * len(N_inputs), torch.tensor(N_frame_ids)


    def __len__(self):
        return len(self.clip_buffer)


def collate_batch(batch_list):
    video_inputs, video_labels, video_names, video_frame_ids = zip(*batch_list)
    # video_inputs = [N, T, C, H, W]
    video_inputs = torch.cat(video_inputs, dim=0)
    video_labels = torch.cat(video_labels, dim=0)
    batch_video_names = []
    for video_name in video_names:
        batch_video_names.extend(video_name)
    video_frame_ids = torch.cat(video_frame_ids, dim=0)
    return video_inputs, video_labels, batch_video_names, video_frame_ids


class AccimageListRandomResizeCrop(object):

    def __init__(self, size=224, scale=(0.8, 0.95), ratio=(1.5, 1.8)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, imgs): # imgs, list of pil image, w < h
        scale = random.uniform(*self.scale)
        ratio = random.uniform(*self.ratio)

        h, w = imgs[0].size
        tw = int(w * scale) # w is shorter
        th = int(tw * ratio)

        h1 = random.randint(0, h - th)
        w1 = random.randint(0, w - tw)
        crop_imgs = [ img.crop(box=(h1, w1, h1 + th, w1 + tw)) for img in imgs ]
        new_imgs = [img.resize(self.size) for img in crop_imgs]
        return new_imgs


# acclelerate data loading
class Prefetcher(): 
    def __init__(self, loader):
        self.data_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        self.inputs = None
        self.labels = None
        self.video_name = None
        self.frame_idx = None

        self.preload()

    def preload(self):
        try:
            self.inputs, self.labels, self.video_name, self.frame_idx = next(self.loader)
        except StopIteration:
            self.inputs = None
            self.labels = None
            self.video_name = None
            self.frame_idx = None
            self.data_loader.dataset.reset_buffer()
            self.loader = iter(self.data_loader)
            return

        with torch.cuda.stream(self.stream):
            self.inputs = self.inputs.cuda(non_blocking=True).float()
            self.labels = self.labels.cuda(non_blocking=True).long()

    def get_next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.inputs
        labels = self.labels
        video_name = self.video_name
        frame_idx = self.frame_idx

        if inputs is not None:
            inputs.record_stream(torch.cuda.current_stream())
        if labels is not None:
            labels.record_stream(torch.cuda.current_stream())
            
        self.preload()
        return inputs, labels, video_name, frame_idx

    def __len__(self):
        return len(self.loader)


def build_loader(args, sub_set, get_label, training=True):
    if training:
        data_transforms = transforms.Compose([
            AccimageListRandomResizeCrop(size=224, scale=(0.8, 0.95), ratio=(1.5, 1.7777777777777777)),
            Lambda(lambda imgs: torch.stack([transforms.ToTensor()(img) for img in imgs])),
            Lambda(lambda imgs: torch.stack([transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(img) for img in imgs]))
        ])
        shuffle = True
    else:
        data_transforms = transforms.Compose([
            Lambda(lambda imgs: [transforms.CenterCrop((306, 544))(img) for img in imgs]),
            Lambda(lambda imgs: [transforms.Resize((224, 224))(img) for img in imgs]),
            Lambda(lambda imgs: [transforms.ToTensor()(img) for img in imgs]),
            Lambda(lambda imgs: torch.stack([transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(img) for img in imgs]))
        ])
        shuffle = False

    print("Data subset:", sub_set)
    dataset = Dataset(data_transforms, args.image_base, args.label_base, args.sample_rate, sub_set, args.N, args.L, get_label, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=3, shuffle=shuffle, pin_memory=True, collate_fn=collate_batch)
    prefetcher = Prefetcher(loader)
    return prefetcher




