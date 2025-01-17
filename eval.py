import os, time, pickle, argparse, copy, random, importlib, pprint, builtins, datetime, shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.temporal import Dataset 
from models.temporal_encoder import Network
from evaluations.metrics import Metrics

import warnings
warnings.filterwarnings("ignore")


def main(args):

    # read data
    dataset = Dataset(args.feature_path, training=False)
    data_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

    # load temporal encoder
    model = Network(in_feature=args.input_size, num_layers=3, num_classes=args.num_classes) 
    model = model.cuda()
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # generate predictions
    metrics = Metrics(args.num_classes) # calculate evaluation metrics
    with torch.no_grad():
        for features, labels, video_name in tqdm(data_loader):
            inputs = features.cuda()
            labels = labels.view(-1).numpy()
            outputs = model(inputs).view(-1, args.num_classes)

            _, predict = torch.max(outputs, 1)
            predict = predict.detach().cpu().numpy()
            metrics.add_video_sample(video_name[0], labels, predict)

    print("Final Results:")
    print(f"acc: {metrics.acc()[0]:.2f}({metrics.acc()[1]:.2f})")

    meanPrec, stdPrec, meanprecphase, stdprecphase = metrics.precision()
    prec_each_phase = " ".join([f"{i}: {meanprecphase[i]:.2f}({stdprecphase[i]:.2f})" for i in range(len(meanprecphase))])
    print(f"precision: {meanPrec:.2f}({stdPrec:.2f}), [{prec_each_phase}]")

    meanRec, stdRec, meanrecphase, stdrecphase = metrics.recall()
    rec_each_phase = " ".join([f"{i}: {meanrecphase[i]:.2f}({stdrecphase[i]:.2f})" for i in range(len(meanrecphase))])
    print(f"recall: {meanRec:.2f}({stdRec:.2f}), [{rec_each_phase}]")

    meanJacc, stdJacc, meanjaccphase, stdjaccphase = metrics.jaccard()
    jacc_each_phase = " ".join([f"{i}: {meanjaccphase[i]:.2f}({stdjaccphase[i]:.2f})" for i in range(len(meanjaccphase))])
    print(f"jaccard: {meanJacc:.2f}({stdJacc:.2f}), [{jacc_each_phase}]")

    meanF1, stdF1, meanf1phase, stdf1phase = metrics.f1()
    F1_each_phase = " ".join([f"{i}: {meanf1phase[i]:.2f}({stdf1phase[i]:.2f})" for i in range(len(meanf1phase))])
    print(f"F1: {meanF1:.2f}({stdF1:.2f}), [{F1_each_phase}]")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment configs', add_help=False)
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--num-classes', default=7, type=int)
    parser.add_argument('--input-size', default=2048, type=int)

    parser.add_argument('--feature-path', type=str)
    parser.add_argument('--ckpt-path', type=str)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"
    os.environ["WANDB_MODE"] = "offline"

    main(args)









