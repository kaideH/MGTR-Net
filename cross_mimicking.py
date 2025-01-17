import os, datetime, shutil, pprint, math, traceback, argparse, sys, builtins
import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models

from torch.cuda.amp import GradScaler, autocast

from utils import graph_construction

import warnings
warnings.filterwarnings("ignore")


def kd_l2_loss(outputs, labels):
    diff = (outputs - labels) ** 2
    l2 = torch.sum(diff, dim=1)
    loss = torch.mean(l2) * 0.5
    return loss


def train_epoch(data_loader, teachers, student, criterion, optimizer, args, scaler, data_sub_set):
    student.train()
    for teacher_key in teachers.keys():
        teacher = teachers[teacher_key]
        teacher.eval()

    total_loss = 0.0
    total_soft_cls_loss, total_feature_loss = 0, 0
    
    bar = tqdm(total=len(data_loader))
    time_start = datetime.datetime.now()
    inputs, labels, video_idxs, frame_idxs = data_loader.get_next()
    while inputs is not None:

        feature_bins = {}
        for i in range(1, args.K+1):
            feature_bins[f"cross_{i}"] = []
        
        for idx, video_idx in enumerate(video_idxs):
            for key in feature_bins.keys():
                if video_idx in data_sub_set[key]:
                    feature_bins[key].append(idx)

        T_outputs, T_s_features = [], []
        S_outputs, S_s_features = [], []
        for teacher_key in feature_bins.keys():
            teacher = teachers[teacher_key]
            teacher.eval()

            curr_inputs = inputs[feature_bins[teacher_key]] # inputs = [B, T, C, H, W]
            if len(curr_inputs) == 0:
                continue
            N, T, C, H, W = curr_inputs.shape
            
            curr_video_idxs = [video_idxs[i] for i in feature_bins[teacher_key]]
            curr_frame_idxs = frame_idxs[feature_bins[teacher_key]] # frame_idxs: [B, T]
            curr_frame_idxs = curr_frame_idxs / args.sample_rate
            edge_index = graph_construction(curr_video_idxs, curr_frame_idxs)
            
            curr_inputs = curr_inputs.view(N * T, C, H, W)
            frame_encodings = curr_frame_idxs.view(-1).long()

            with autocast():
                # teacher 
                with torch.no_grad():
                    teacher_outputs, teacher_s_features, _ = teacher(curr_inputs, edge_index, frame_encodings)
                T_outputs.append(teacher_outputs) # [N, C]
                T_s_features.append(teacher_s_features) # [N, D]

                # student
                student_s_features = student(curr_inputs)
                student_outputs, _ = teacher.gtm_forward(student_s_features, edge_index, frame_encodings)
                S_outputs.append(student_outputs)
                S_s_features.append(student_s_features)

        
        # step 3: 综合计算损失函数
        optimizer.zero_grad()
        loss = 0
        # soft cls loss
        T_outputs = torch.cat(T_outputs, dim=0).detach() # [B, C]
        S_outputs = torch.cat(S_outputs, dim=0)
        soft_cls_loss = criterion(F.log_softmax(S_outputs / args.T, dim=1), F.softmax(T_outputs / args.T, dim=1)) * args.T * args.T * 100

        # spatial feature loss
        T_s_features = torch.cat(T_s_features, dim=0).detach()
        S_s_features = torch.cat(S_s_features, dim=0)
        feature_loss = kd_l2_loss(S_s_features, T_s_features)

        loss = args.soft_cls_lambda * soft_cls_loss + args.feature_lambda * feature_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_soft_cls_loss += soft_cls_loss.item()
        total_feature_loss += feature_loss.item()

        inputs, labels, video_idxs, frame_idxs = data_loader.get_next()
        bar.set_postfix_str(f"Train Loss={loss.item():.4f}, Soft cls Loss={soft_cls_loss.item():.4f}, Spatial Feature Loss={feature_loss.item():.4f}")
        bar.update(1)

    time_spend = (datetime.datetime.now() - time_start).seconds
    average_loss = total_loss / len(data_loader)
    average_soft_cls_loss = total_soft_cls_loss / len(data_loader)
    total_feature_loss = total_feature_loss / len(data_loader)
    loss_items = [average_loss, average_soft_cls_loss, total_feature_loss]
    return loss_items, time_spend


def main(args):
    ## preparation
    args.base_path = os.path.join("exps", args.dataset, "CM") 
    assert not os.path.exists(args.base_path), f"Experiment folder {args.base_path} already exists!" 
    os.makedirs(args.base_path)

    args.checkpoint_path = os.path.join(args.base_path, "ckpts")
    os.makedirs(args.checkpoint_path)

    log_path = os.path.join(args.base_path, 'log.log')
    logger.remove()
    logger.add(sys.stdout)
    logger.add(log_path, encoding='utf-8')
    def print_func(*args):
        for arg in args:
            logger.info(arg)
    builtins.print = print_func
    print(args)


    ## data 
    from datasets.sparsely_sample import build_loader
    if args.dataset == "cholec80":
        from data_splits.cholec80 import get_label, DATA_SPLIT
    else:
        raise Exception(f"unknown dataset {args.datasets}")
    train_loader = build_loader(args, DATA_SPLIT[f"train_all"], get_label, training=True)
    print("Iters train: {}".format(len(train_loader))) 

    scaler = GradScaler() 

    ## model
    # teacher encoders
    from models.feature_encoder import Network
    teachers = {}
    for i in range(1, args.K+1):
        model = Network(num_classes=args.num_classes)
        ckpt_path = os.path.join("exps", args.dataset, "SoC", f"cross_{i}/ckpts/best_val_acc.pth") 
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        for param in model.parameters():
            param.requires_grad=False
        teachers[f"cross_{i}"] = model.cuda()
    
    # student encoder
    student = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2) # backbone
    student.fc = nn.Identity()
    student = nn.DataParallel(student).cuda()


    ## loss and optimizer
    criterion = nn.KLDivLoss()
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-7)
    

    # training procedure
    best_kd_loss, bset_epoch_loss = 1000, -1
    for epoch in range(1, args.epochs+1):
        # learning rate
        lr = optimizer.state_dict().get("param_groups")[-1].get("lr")
        print(f"Epoch {epoch}, LR: {lr}")


        ## train
        loss_items, time_spend = train_epoch(train_loader, teachers, student, criterion, optimizer, args, scaler, DATA_SPLIT)
        average_loss, average_soft_cls_loss, average_feature_loss = loss_items
        print(f"Epoch {epoch}, Train time: {time_spend}, Train loss: {average_loss:.6f}, Soft cls Loss={average_soft_cls_loss:.6f}, Spatial Feature Loss={average_feature_loss:.6f}")
        scheduler.step()

        # save model
        checkpoint_dict = {
            'epoch': epoch, 
            'state_dict': student.state_dict(), 
            'optimizer': optimizer.state_dict()
        }
        pthFilePath = os.path.join(args.checkpoint_path, "last.pth")
        torch.save(checkpoint_dict, pthFilePath)

        # update training msg
        if average_loss < best_kd_loss:
            best_kd_loss = average_loss
            bset_epoch_loss = epoch
            shutil.copyfile(pthFilePath, os.path.join(args.checkpoint_path, "best_train_loss.pth"))
            print("Train Loss decreased to {}".format(average_loss))
        else:
            print("Train Loss not decrease, best Loss is {} in epoch {}".format(best_kd_loss, bset_epoch_loss))
        

    print("All Done !!!")
    print('Bset Train Loss: {} in epoch: {}, .'.format(best_kd_loss, bset_epoch_loss))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment configs', add_help=False)
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--gpus', default="0", type=str, help='GPU id to use.')

    parser.add_argument('--image-base', type=str)
    parser.add_argument('--label-base', type=str)

    parser.add_argument('--dataset', default="cholec80", type=str)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--batch-size', default=6, type=int)
    parser.add_argument('--N', default=10, type=int)
    parser.add_argument('--L', default=10, type=int)
    parser.add_argument('--sample-rate', default=25, type=int)

    parser.add_argument('--K', default=5, type=int)

    parser.add_argument('--T', default=5, type=float)
    parser.add_argument('--soft-cls-lambda', default=100, type=float)
    parser.add_argument('--feature-lambda', default=1, type=float)

    parser.add_argument('--num-classes', default=7, type=int)
    parser.add_argument('--cross', default=1, type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"
    main(args)









