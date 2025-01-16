import os, datetime, shutil, wandb, math, argparse
import numpy as np
from tqdm import tqdm

import torch, torchvision
import torch.nn as nn
from torchvision import transforms

from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")


DATA_SPLIT_DICT = {
    "cholec80": {
        "train": [
            'video01', 'video02', 'video03', 'video04', 'video05', 'video06', 'video07', 'video08', 'video09', 'video10',
            'video11', 'video12', 'video13', 'video14', 'video15', 'video16', 'video17', 'video18', 'video19', 'video20',
            'video21', 'video22', 'video23', 'video24', 'video25', 'video26', 'video27', 'video28', 'video29', 'video30',
            'video31', 'video32',
        ],
        "val": [
            'video33', 'video34', 'video35', 'video36', 'video37', 'video38', 'video39', 'video40',
        ],
        "test": [
            'video41', 'video42', 'video43', 'video44', 'video45', 'video46', 'video47', 'video48', 'video49', 'video50', 
            'video51', 'video52', 'video53', 'video54', 'video55', 'video56', 'video57', 'video58', 'video59', 'video60', 
            'video61', 'video62', 'video63', 'video64', 'video65', 'video66', 'video67', 'video68', 'video69', 'video70', 
            'video71', 'video72', 'video73', 'video74', 'video75', 'video76', 'video77', 'video78', 'video79', 'video80',
        ],
    },
    "cataracts101": {
        "train": [
            'case_278', 'case_749', 'case_827', 'case_846', 'case_911', 'case_785', 'case_925', 'case_750', 'case_808', 'case_734', 
            'case_853', 'case_882', 'case_891', 'case_809', 'case_738', 'case_895', 'case_871', 'case_270', 'case_926', 'case_810', 
            'case_863', 'case_829', 'case_896', 'case_887', 'case_770', 'case_768', 'case_784', 'case_771', 'case_288', 'case_908', 
            'case_804', 'case_900', 'case_849', 'case_799', 'case_909', 'case_835', 'case_279', 'case_880', 'case_828', 'case_899', 
            'case_807', 'case_760', 'case_883', 'case_294', 'case_778', 'case_830', 'case_806', 'case_781', 'case_889', 'case_821', 
            'case_797', 'case_847', 'case_901', 'case_931', 'case_890'
        ],
        "val": [
            'case_921', 'case_350', 'case_898', 'case_902', 'case_834', 'case_840', 'case_801', 'case_884', 'case_845', 'case_295', 
            'case_892', 'case_928', 'case_796', 'case_850', 'case_280', 'case_825', 'case_354', 'case_906'
        ],
        "test": [
            'case_857', 'case_861', 'case_907', 'case_269', 'case_886', 'case_868', 'case_856', 'case_933', 'case_855', 'case_764', 
            'case_922', 'case_356', 'case_929', 'case_292', 'case_802', 'case_786', 'case_867', 'case_866', 'case_788', 'case_932', 
            'case_739', 'case_745', 'case_751', 'case_841', 'case_817', 'case_865', 'case_934', 'case_271'
        ],
    },
}


def main(args):
    # student encoder
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    model = nn.DataParallel(model).cuda()
    # ckpt_path = f"ckpts/{args.dataset}/student_encoder.pth" 
    ckpt_path = f"exps/{args.dataset}/CM/ckpts/best_train_loss.pth" 
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])

    # dataset
    import data_split
    from spatial.datasets.cholec80.frame import Dataset
    data_transforms = transforms.Compose([
        transforms.CenterCrop((306, 544)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
    ])
    args.frame_dir = "/root/dataspace/cholec80/frames"
    args.label_dir = "/root/dataspace/cholec80/phase_annotations"
    sub_set = data_split.data_sub_set_cholec80
    dataset = Dataset(args.frame_dir, args.label_dir, data_transforms, 25, sub_set[f"all"])
    data_loader = DataLoader(dataset, batch_size=1024, num_workers=6, shuffle=False, pin_memory=True)

    # visual feature save path
    train_feature_dir = f"features/{args.dataset}/train"
    val_feature_dir = f"features/{args.dataset}/val"
    test_feature_dir = f"features/{args.dataset}/test"
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

        data_split = DATA_SPLIT_DICT[args.dataset]
        if video_name in data_split["train"]:
            feature_save_dir = train_feature_dir
        elif video_name in data_split["val"]:
            feature_save_dir = val_feature_dir
        elif video_name in data_split["test"]:
            feature_save_dir = test_feature_dir
        else:
            print(f"unknown video name {video_name}")

        np.save(os.path.join(feature_save_dir, f"{video_name}.npy"), features)
        np.save(os.path.join(feature_save_dir, f"{video_name}-labels.npy"), labels)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment configs', add_help=False)
    parser.add_argument('--gpus', default="2 ", type=str, help='GPU id to use.')
    parser.add_argument('--frame-dir', default="/root/dataspace/cholec80/frames", type=str, help='Path to the video frames.')
    parser.add_argument('--label-dir', default="/root/dataspace/cholec80/phase_annotations", type=str, help='Path to labels.')
    parser.add_argument('--dataset', default="cholec80", type=str, help='Dataset used.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"

    main(args)









