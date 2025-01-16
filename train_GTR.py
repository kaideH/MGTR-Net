import os, datetime, shutil, json, argparse, sys, builtins
import numpy as np
from loguru import logger
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.cuda.amp import GradScaler, autocast

from utils import graph_construction
# from grasp_eval import eval_grasp_mAP

import warnings
warnings.filterwarnings("ignore")


def train_epoch(data_loader, model, optimizer, criterion, mse, args, scaler):
    model.train()
    total_loss = 0.
    corrects, total = 0., 0.

    time_start = datetime.datetime.now()
    bar = tqdm(total=len(data_loader))

    pred_dicts = {}
    inputs, labels, video_names, frame_idxs = data_loader.get_next()
    while inputs is not None:
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)
        labels = labels.view(-1) # [B, T] -> [B * T]

        # graph construction
        frame_idxs = (frame_idxs / args.sample_rate).long()
        edge_index = graph_construction(video_names, frame_idxs)
        frame_encodings = frame_idxs.view(-1)

        optimizer.zero_grad()
        with autocast():
            outputs, _, _ = model(inputs, edge_index, frame_encodings) # outputs = [B*T, C]
            cls_loss = criterion(outputs, labels)
            output_clip = outputs.view(B, T, -1)
            smooth_loss = 0.15 * torch.mean( torch.clamp( mse( F.log_softmax(output_clip[:, 1:, :], dim=-1), F.log_softmax(output_clip.detach()[:, :-1, :], dim=-1) ), min=0, max=16 ) )
            loss = cls_loss + smooth_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        _, predicts = torch.max(outputs, dim=1)
        corrects += torch.sum(predicts==labels).item()
        total += labels.shape[0]
        
        inputs, labels, video_names, frame_idxs = data_loader.get_next()
        bar.update(1)

    time_spend = (datetime.datetime.now() - time_start).seconds
    avg_loss = total_loss / len(data_loader)
    acc = corrects / total * 100

    return acc, avg_loss, time_spend


@torch.no_grad()
def test_epoch(data_loader, model, criterion, mse):
    model.eval()
    total_loss = 0
    corrects, total = 0, 0

    pred_dicts = {}
    time_start = datetime.datetime.now()
    bar = tqdm(total=len(data_loader))

    inputs, labels, video_names, frame_idxs = data_loader.get_next()
    while inputs is not None:
        B, T, C, H, W = inputs.shape
        inputs = inputs.view(B * T, C, H, W)
        labels = labels.view(-1) # [B, T] -> [B * T]

        # graph construction
        frame_idxs = (frame_idxs / args.sample_rate).long()
        edge_index = graph_construction(video_names, frame_idxs)
        frame_encodings = frame_idxs.view(-1)

        with autocast():
            outputs, _, _ = model(inputs, edge_index, frame_encodings) # outputs = [B*T, C]
            cls_loss = criterion(outputs, labels)
            output_clip = outputs.view(B, T, -1)
            smooth_loss = 0.15 * torch.mean( torch.clamp( mse( F.log_softmax(output_clip[:, 1:, :], dim=-1), F.log_softmax(output_clip.detach()[:, :-1, :], dim=-1) ), min=0, max=16 ) )
            loss = cls_loss + smooth_loss
        total_loss += loss.item()

        _, predicts = torch.max(outputs, dim=1)
        corrects += torch.sum(predicts==labels).item()
        total += labels.shape[0]

        ## GraSP mAP
        # new_video_names = []
        # for video_name in video_names:
        #     new_video_names.extend([video_name] * 10)
        # pred_scores = torch.softmax(outputs, 1).detach().cpu().numpy()
        # for i in range(len(pred_scores)):
        #     pred_dicts[f"{new_video_names[i]}/{frame_encodings[i]:05d}.jpg"] = { "phases_score_dist": pred_scores[i].tolist() }

        inputs, labels, video_names, frame_idxs = data_loader.get_next()
        bar.update(1)

    time_spend = (datetime.datetime.now() - time_start).seconds
    avg_loss = total_loss / len(data_loader)
    acc = corrects / total * 100

    # json.dump(pred_dicts, open("grasp_json/grasp_prediction.json", "w"))

    return acc, avg_loss, time_spend


def main(args):
    ## setup experiment
    args.base_path = os.path.join("exps", args.dataset, "SoC", f"cross_{args.cross}") 
    assert not os.path.exists(args.base_path), f"Experiment folder {args.base_path} already exists!" 
    os.makedirs(args.base_path)

    # checkpoint
    args.checkpoint_path = os.path.join(args.base_path, "ckpts")
    os.makedirs(args.checkpoint_path)

    # logger
    log_path = os.path.join(args.base_path, 'log.log')
    logger.remove()
    logger.add(sys.stdout)
    logger.add(log_path, encoding='utf-8')
    def print_func(*args):
        for arg in args:
            logger.info(arg)
    builtins.print = print_func
    print(args)


    # data 
    from spatial.datasets.sparsely_sample import build_loader
    import data_split
    if args.dataset == "cholec80":
        args.image_base = "/root/dataspace/cholec80/frames"
        args.label_base = "/root/dataspace/cholec80/phase_annotations"
        get_label = data_split.get_label_cholec80
        sub_set = data_split.data_sub_set_cholec80
    elif args.dataset == "cataracts101":
        from spatial.datasets.cataracts101.cataracts101_clip import build_loader
    elif args.dataset == "autolaparo":
        from spatial.datasets.autolaparo.clip import build_loader
    elif args.dataset == "GraSP":
        from spatial.datasets.grasp.clip import build_loader
    elif args.dataset == "MultiBypass140":
        from spatial.datasets.mbp140.clip import build_loader
    elif args.dataset == "Heidelberg":
        from spatial.datasets.heidelberg.clip import build_loader
    elif args.dataset == "cholecT50":
        from spatial.datasets.cholecT50.clip import build_loader
    elif args.dataset == "Hei-Chole":
        from spatial.datasets.heichole.clip import build_loader
    else:
        raise NotImplementedError(f"unkown dataset {args.dataset}")

    train_loader = build_loader(args, sub_set[f"cross_{args.cross}_train"], get_label, training=True)
    val_loader  =  build_loader(args, sub_set[f"cross_{args.cross}"], get_label, training=False)
    print(f"Iters train: {len(train_loader)}, val: {len(val_loader)}")
    

    # model
    from spatial.feature_encoder import Network
    model = Network(args.num_classes).cuda()
    param_group = [
        {'params': model.backbone.parameters(), 'lr': args.lr / 10},
        {'params': model.gnn.parameters()},
        {'params': model.head.parameters()},
    ]


    # loss and optimizer
    criterion = nn.CrossEntropyLoss() # cls loss
    mse = nn.MSELoss(reduction='none') # smooth loss
    optimizer = optim.AdamW(param_group, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()


    # training procedure
    best_test_acc, best_epoch_test_acc = 0, -1
    best_val_acc, best_epoch_val_acc = 0, -1
    for epoch in range(1, args.epochs+1):
        # learning rate
        lr = optimizer.state_dict().get("param_groups")[-1].get("lr")
        print(f"Epoch {epoch}, LR: {lr}")


        ## train
        train_acc, average_loss, time_spend = train_epoch(train_loader, model, optimizer, criterion, mse, args, scaler)
        print(f"Epoch {epoch}, Train time: {time_spend}, Train loss: {average_loss:.6f}, Train acc: {train_acc:.4f}")
        scheduler.step()


        # save model
        checkpoint_dict = {
            'epoch': epoch, 
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict()
        }
        pthFilePath = os.path.join(args.checkpoint_path, "last.pth")
        torch.save(checkpoint_dict, pthFilePath)


        ## val
        val_acc, average_loss, time_spend = test_epoch(val_loader, model, criterion, mse)
        print(f"Epoch {epoch}, Val time: {time_spend}, Val loss: {average_loss:.6f}, Val acc: {val_acc:.4f}")
        # test_mAP = eval_grasp_mAP(coco_ann_path=f"grasp_json/cross_{args.cross}.json", pred_path="grasp_json/grasp_prediction.json")

        # update test msg
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch_val_acc = epoch
            shutil.copyfile(pthFilePath, os.path.join(args.checkpoint_path, "best_val_acc.pth"))
            print("Val ACC increased to {}".format(best_val_acc))
        else:
            print("Val ACC not increase, best ACC is {} in epoch {}".format(best_val_acc, best_epoch_val_acc))

    print("All Done !!!")
    print('Bset val ACC: {} in epoch: {}, .'.format(best_val_acc, best_epoch_val_acc))
    return


"""
python train_GTR.py --dataset=cholec80 --gpus="2" --epochs=30 --batch-size=4 --sample-rate=25 --lr=1e-3 --num-classes=7 --N=10 --L=10 --cross=1
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment configs', add_help=False)
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--gpus', default="0", type=str, help='GPU id to use.')

    parser.add_argument('--dataset', default="cholec80", type=str)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--batch-size', default=6, type=int)
    parser.add_argument('--N', default=10, type=int)
    parser.add_argument('--L', default=10, type=int)
    parser.add_argument('--sample-rate', default=25, type=int)

    parser.add_argument('--num-classes', default=7, type=int)
    parser.add_argument('--cross', default=1, type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"

    main(args)









