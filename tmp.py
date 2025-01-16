import os, datetime, shutil, wandb, argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from torchvision import transforms

from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")


def feature_extraction(args, model, data_loader, data_sub_set):
    # extract features
    video_bank = {}
    with torch.no_grad():
        model.eval()
        for inputs, labels, video_names, frame_idxs in tqdm(data_loader):
            B, C, H, W = inputs.shape

            if args.half:
                with autocast():
                    features = model(inputs).view(B, -1) # outputs = [B, D]
            else:
                features = model(inputs).view(B, -1) # outputs = [B, D]
            
            for idx, feature in enumerate(features):
                video_name = video_names[idx]
                frame_idx = frame_idxs[idx]
                label = labels[idx]
                if video_bank.get(video_name) is None:
                    video_bank[video_name] = []
                video_bank[video_name].append([frame_idx.item(), feature.cpu().numpy(), label.item()])
    
    # save features
    print(data_sub_set)
    for video_name in video_bank.keys():
        video_features = video_bank[video_name]
        video_features = sorted(video_features, key=lambda x: x[0])
        features = np.array([p[1] for p in video_features])
        labels  = np.array([p[2] for p in video_features])
        print(features.shape, labels.shape)

        if video_name in data_sub_set[f"train"]:
            video_save_base = args.train_save_base
        elif video_name in data_sub_set[f"val"]:
            video_save_base = args.val_save_base
        elif video_name in data_sub_set["test"]:
            video_save_base = args.test_save_base
        else:
            print(f"unkown video name: {video_name}")
        np.save(os.path.join(video_save_base, f"{video_name}.npy"), features)
        np.save(os.path.join(video_save_base, f"{video_name}-labels.npy"), labels)
    
    return


def ensemble(temp_dir, save_dir):
    feature_bank = {}
    for cross in os.listdir(temp_dir):
        cross_dir = os.path.join(temp_dir, cross)
        for f_name in os.listdir(cross_dir):
            if not f_name.endswith("labels.npy"):
                continue
            
            video_name = f_name.split("-")[0]
            if feature_bank.get(video_name) is None:
                f_path = os.path.join(cross_dir, f_name)
                label = np.load(f_path)        
                feature_bank[video_name] = {"feature": [], "label": label}

            feature_path = os.path.join(cross_dir, f"{video_name}.npy")
            feature = np.load(feature_path)
            feature_bank[video_name]["feature"].append(feature)
    
    for video_name in feature_bank.keys():
        features = np.array(feature_bank[video_name]["feature"])
        feature_mean = np.mean(features, axis=0)
        np.save(os.path.join(save_dir, f"{video_name}.npy"), feature_mean)
        labels = feature_bank[video_name]["label"]
        np.save(os.path.join(save_dir, f"{video_name}-labels.npy"), labels)
        print(feature_mean.shape, labels.shape)
    return


def main(args):
    # 根据不同数据集引用相关代码
    from spatial.datasets.cholec80.frame import Dataset
    import data_split
    data_transforms = transforms.Compose([
        transforms.CenterCrop((306, 544)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])
    sub_set = data_split.data_sub_set_cholec80

    save_base = "features"
    args.train_save_base = os.path.join(save_base, "train")
    args.val_save_base = os.path.join(save_base, "val")
    os.makedirs(args.train_save_base)
    os.makedirs(args.val_save_base)

    if args.ensemble:
        # ensemble 特征提取分两部分, 第一部分提取他对应的val特征, 另一部分提取test特征, test特征需要ensemble
        for cross in range(1, 6):
            print("cross", cross)
            # 用来暂存test特征, 全部提取完成之后做ensemble
            args.test_save_base = os.path.join(save_base, "test_temp", f"cross_{cross}")
            os.makedirs(args.test_save_base)

            from spatial.feature_encoder import Network
            model = Network(args.num_classes)
            ckpt_path = os.path.join("exps", args.dataset, "SoC", f"cross_{cross}", "ckpts", f"{args.feature_type}.pth")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'])
            model = model.backbone.cuda()

            print(sub_set[f"cross_{cross}"])
            dataset = Dataset("/root/dataspace/cholec80/frames", "/root/dataspace/cholec80/phase_annotations", data_transforms, 25, sub_set[f"cross_{cross}"])
            data_loader = DataLoader(dataset, batch_size=256, num_workers=6, shuffle=False, pin_memory=True)
            feature_extraction(args, model, data_loader, sub_set)

            print(sub_set[f"test"])
            dataset = Dataset("/root/dataspace/cholec80/frames", "/root/dataspace/cholec80/phase_annotations", data_transforms, 25, sub_set[f"test"])
            data_loader = DataLoader(dataset, batch_size=256, num_workers=6, shuffle=False, pin_memory=True)
            feature_extraction(args, model, data_loader, sub_set)

        print("start ensemble features . . .")
        temp_dir = os.path.join(save_base, "test_temp")
        save_dir = os.path.join(save_base, "test")
        os.makedirs(save_dir)
        ensemble(temp_dir, save_dir)
        shutil.rmtree(temp_dir)

    else:
        args.test_save_base = os.path.join(save_base, "test")
        os.makedirs(args.test_save_base)

        data_loader = build_loader(args, "all", training=False)
        print(f"Feature Extraction Iters: {len(data_loader)}")

        # 加载模型  
        # from spatial.models.res50 import Network
        # # from spatial.models.res50 import Network_LSTM
        # model = Network(args.num_classes)
        # checkpoint_path = os.path.join("exps", args.dataset, args.exp_name, args.exp_type, "ckpts")
        # ckpt_path = os.path.join(checkpoint_path, f"{args.feature_type}.pth") 
        # checkpoint = torch.load(ckpt_path, map_location="cpu")
        # model.load_state_dict(checkpoint['state_dict'])
        # model = model.backbone.cuda()
        ## student model
        import torchvision.models
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2) # backbone
        model.fc = nn.Identity()
        # model.fc = nn.Linear(2048, args.num_classes)
        model = nn.DataParallel(model).cuda()
        checkpoint_path = os.path.join("exps", args.dataset, args.exp_name, args.exp_type, "ckpts")
        ckpt_path = os.path.join(checkpoint_path, f"{args.feature_type}.pth") 
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])

        print(model)

        feature_extraction(args, model, data_loader, data_sub_set)
    print("Feature Extraction Done !!!")
    return


"""
python tmp.py --dataset=cholec80 --gpus="0" --sample-rate=25 --lr=1e-3 --num-classes=7 --feature-type=best_val_acc --half --ensemble

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment configs', add_help=False)
    parser.add_argument('--dataset', default="cholec80", type=str)

    parser.add_argument('--gpus', default="0", type=str, help='GPU id to use.')
    parser.add_argument('--cross', default=1, type=int)

    parser.add_argument('--batch-size', default=200, type=int)
    parser.add_argument('--sample-rate', default=25, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--num-classes', default=7, type=int)

    parser.add_argument('--feature-type', default="best_train_loss", type=str) # for esd 5-fold cross validation

    parser.add_argument('--half', action="store_true") # 是否使用半精度提取特征
    parser.add_argument('--ensemble', action="store_true") # 是否使用半精度提取特征

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"
    os.environ["WANDB_MODE"] = "offline"

    main(args)









