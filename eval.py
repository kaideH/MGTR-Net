import os, time, pickle, argparse, copy, random, importlib, pprint, builtins, datetime, shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from temporal.datasets import Dataset 
from torch.utils.data import DataLoader

from temporal.model import Network
from evaluations.metrics import Metrics

import warnings
warnings.filterwarnings("ignore")


phase_dict_key = {
        0: 'Preparation', 
        1: 'CalotTriangleDissection', 
        2: 'ClippingCutting', 
        3: 'GallbladderDissection', 
        4: 'GallbladderPackaging', 
        5: 'CleaningCoagulation', 
        6: 'GallbladderRetraction', 
    }


def main(args):

    # read data
    dataset = Dataset(args.feature_dir)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

    # load temporal encoder
    model = Network(in_feature=2048, num_layers=3, num_classes=args.num_classes) 
    model = model.cuda()
    ckpt_path = f"ckpts/{args.dataset}/temporal_encoder.pth" 
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])

    # save prediction results
    anno_dir = f"evaluations/matlab-eval/{args.dataset}/labels"
    pred_save_dir = f"evaluations/matlab-eval/{args.dataset}/predicts"
    if os.path.exists(pred_save_dir):
        shutil.rmtree(pred_save_dir)
    os.makedirs(pred_save_dir)

    # generate predictions
    metrics = Metrics(args.num_classes) # calculate evaluation metrics
    with torch.no_grad():
        model.eval()
        for features, labels, video_name in tqdm(data_loader):
            inputs = features.cuda()
            labels = labels.view(-1).numpy()
            outputs = model(inputs)

            _, predict = torch.max(outputs[0], 1)
            predict = predict.detach().cpu().numpy()
            metrics.add_video_sample(video_name[0], labels, predict)

            # extend prediction with 25 pfs
            total_predicts = []
            for p in predict:
                total_predicts.extend([p]*25)

            anno_file = open(os.path.join(anno_dir, f"{video_name[0]}-phase.txt"))
            annos = anno_file.readlines()[1:]
            anno_file.close()
            if len(annos) < len(total_predicts):
                total_predicts = total_predicts[:len(annos)]
            elif len(annos) > len(total_predicts):
                print(video_name[0], len(annos), len(total_predicts))
                # append_label = total_predicts[-1]
                # total_predicts.extend([append_label]*(len(annos)-len(total_predicts)))

            # write final predictions into txt files
            f = open(os.path.join(pred_save_dir, f"{video_name[0]}-pred.txt"), "w")
            if args.dataset == "cholec80":
                f.write("frame\tphase\n")
            for idx, pred in enumerate(total_predicts):
                curr_pred = int(pred)
                if args.dataset == "cholec80":
                    curr_pred = phase_dict_key[curr_pred]
                f.write(f"{idx}\t{curr_pred}\n")
            f.close()

    # evaluation metrics
    print(metrics.metrics_dict())
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment configs', add_help=False)
    parser.add_argument('--gpus', default="0", type=str, help='GPU id to use.')
    parser.add_argument('--num-classes', default=7, type=int, help='Number of surgical phases.')
    parser.add_argument('--feature-dir', default="features/cholec80/test", type=str, help='Path to labels.')
    parser.add_argument('--dataset', default="cholec80", type=str, help='Dataset used.')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"
    os.environ["WANDB_MODE"] = "offline"

    main(args)









