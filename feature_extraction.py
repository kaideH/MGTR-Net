import os, datetime, shutil, wandb, math, argparse
import numpy as np
from tqdm import tqdm

import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader


def main(args):
    # student encoder
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model = nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])

    # dataset
    from datasets.frame import Dataset
    if args.dataset == "cholec80":
        from data_splits.cholec80 import DATA_SPLIT, get_label
    else:
        raise Exception(f"unknown dataset {args.datasets}")
    data_transforms = transforms.Compose([
        transforms.CenterCrop((306, 544)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])
    dataset = Dataset(args.frame_dir, args.label_dir, data_transforms, 25, get_label, DATA_SPLIT[f"all"])
    data_loader = DataLoader(dataset, batch_size=1024, num_workers=6, shuffle=False, pin_memory=True)

    # visual feature save path
    train_feature_dir = os.path.join(args.feature_base, "train")
    val_feature_dir = os.path.join(args.feature_base, "val")
    test_feature_dir = os.path.join(args.feature_base, "test")
    if not os.path.exists(train_feature_dir):
        os.makedirs(train_feature_dir)
        os.makedirs(val_feature_dir)
        os.makedirs(test_feature_dir)

    # extract visual features by student encoder
    video_bank = {}
    with torch.no_grad():
        model.eval()
        for data in tqdm(data_loader):
            inputs, labels, video_names, frame_idxs = data
            features = model(inputs) # features = [batch_size, feature_size]
            for idx, feature in enumerate(features):
                video_name = video_names[idx]
                frame_idx = frame_idxs[idx]
                label = labels[idx]
                if video_bank.get(video_name) is None:
                    video_bank[video_name] = []
                video_bank[video_name].append([frame_idx.item(), feature.cpu().numpy(), label.item()])
    
    # save visual features
    for video_name in video_bank.keys():
        video_features = video_bank[video_name]
        sorted_features = sorted(video_features)
        features = np.array([p[1] for p in sorted_features])
        labels = np.array([p[2] for p in sorted_features])

        if video_name in DATA_SPLIT["train"]:
            feature_save_dir = train_feature_dir
        elif video_name in DATA_SPLIT["val"]:
            feature_save_dir = val_feature_dir
        elif video_name in DATA_SPLIT["test"]:
            feature_save_dir = test_feature_dir
        else:
            print(f"unknown video name {video_name}")

        np.save(os.path.join(feature_save_dir, f"{video_name}.npy"), features)
        np.save(os.path.join(feature_save_dir, f"{video_name}-labels.npy"), labels)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment configs', add_help=False)
    parser.add_argument('--gpus', default="0", type=str, help='GPU id to use.')
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--frame-dir', type=str)
    parser.add_argument('--label-dir', type=str)
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--feature-base', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"

    main(args)









