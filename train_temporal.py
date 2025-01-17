import os, time, pickle, argparse, copy, random, importlib, pprint, builtins, datetime, shutil, json, sys
import numpy as np
from tqdm import tqdm
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from evaluations.metrics import Metrics
from datasets.temporal import Dataset 


def loss_function(outputs, labels, ce, mse, args):
    loss = 0
    outputs = outputs.view(-1, args.num_classes)
    loss += ce(outputs, labels)
    loss += args.alpha * torch.mean( torch.clamp( mse( F.log_softmax(outputs[1:, :], dim=1), F.log_softmax(outputs.detach()[:-1, :], dim=1) ), min=0, max=16 ) )
    return loss, outputs


def train_epoch(data_loader, model, optimizer, criterion, args):
    model.train()

    metrics = Metrics(args.num_classes)
    total_loss = 0.0
    ce, mse = criterion

    time_start = datetime.datetime.now()
    for features, labels, video_name in tqdm(data_loader):
        inputs = features.cuda().float()
        labels = labels.long().cuda().view(-1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss, outputs = loss_function(outputs, labels, ce, mse, args)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicts = torch.max(outputs, 1)
        predicts = predicts.detach().cpu().numpy()
        metrics.add_video_sample(video_name[0], labels.cpu().numpy(), predicts)
        
    time_spend = (datetime.datetime.now() - time_start).seconds
    average_loss = total_loss / len(data_loader)

    return average_loss, time_spend, metrics


@torch.no_grad()
def test_epoch(data_loader, model, criterion, args):
    model.eval()

    metrics = Metrics(args.num_classes)
    total_loss = 0.0
    ce, mse = criterion

    time_start = datetime.datetime.now()
    for features, labels, video_name in tqdm(data_loader):
        inputs = features.cuda().float()
        labels = labels.long().cuda().view(-1) 

        outputs = model(inputs)
        loss, outputs = loss_function(outputs, labels, ce, mse, args)

        total_loss += loss.item()

        _, predicts = torch.max(outputs, 1)
        predicts = predicts.detach().cpu().numpy()
        metrics.add_video_sample(video_name[0], labels.cpu().numpy(), predicts)

    time_spend = (datetime.datetime.now() - time_start).seconds
    average_loss = total_loss / len(data_loader)

    return average_loss, time_spend, metrics


def main(args):
    ## preparation
    args.base_path = os.path.join("exps", args.dataset, "temporal") 
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
    dataset = Dataset(os.path.join(args.feature_path, "train"), training=True)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True, pin_memory=True)

    dataset = Dataset(os.path.join(args.feature_path, "val"), training=False)
    val_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)

    dataset = Dataset(os.path.join(args.feature_path, "test"), training=False)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False, pin_memory=True)
    print(f"Iters train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")


    from models.temporal_encoder import Network
    model = Network(in_feature=args.input_size, num_layers=3, num_classes=args.num_classes) 
    model = model.cuda()
    print(f'Model Size: {sum(p.numel() for p in model.parameters())}')
    

    ## exp components
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    mse = nn.MSELoss(reduction='none')
    criterion = [ce, mse]
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-6)


    # train 
    best_val_acc, bset_val_epoch = 0, -1
    best_test_acc, bset_test_epoch = 0, -1
    for epoch in range(1, args.epochs+1):
        # learning rate
        lr = optimizer.state_dict()["param_groups"][-1]["lr"]
        print(f"Epoch {epoch}, LR: {lr}")

        ## train
        average_loss, time_spend, metric = train_epoch(train_loader, model, optimizer, criterion, args)
        print(f"Epoch {epoch}, Train time: {time_spend}, Train loss: {average_loss:.6f}")
        print(f"Train metrics: \n{pprint.pformat(metric.metrics_dict())}.")
        scheduler.step(average_loss)


        # save train checkpoint
        checkpoint_dict = {
            'epoch': epoch, 
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'criterion': [ce.state_dict(), mse.state_dict()]
        }
        pthFilePath = os.path.join(args.checkpoint_path, "last.pth")
        torch.save(checkpoint_dict, pthFilePath)


        ## val
        average_loss, time_spend, metric = test_epoch(val_loader, model, criterion, args)
        print(f"Epoch {epoch}, Val time: {time_spend}, Val loss: {average_loss:.6f}")
        print(f"Val metrics: \n{pprint.pformat(metric.metrics_dict())}.")

        if metric.acc()[0] > best_val_acc:
            best_val_acc = metric.acc()[0]
            bset_val_epoch = epoch
            shutil.copyfile(pthFilePath, os.path.join(args.checkpoint_path, f"best_val_acc.pth"))
            print("Val ACC increased to {}".format(best_val_acc))
        else:
            print("Val ACC not increase, best ACC is {} in epoch {}".format(best_val_acc, bset_val_epoch))


        ## test
        average_loss, time_spend, metric = test_epoch(test_loader, model, criterion, args)
        print(f"Epoch {epoch}, Test time: {time_spend}, Test loss: {average_loss:.6f}")
        print(f"Test metrics: \n{pprint.pformat(metric.metrics_dict())}.")

        if metric.acc()[0] > best_test_acc:
            best_test_acc = metric.acc()[0]
            bset_test_epoch = epoch
            print("Test ACC increased to {}".format(best_test_acc))
        else:
            print("Test ACC not increase, best ACC is {} in epoch {}".format(best_test_acc, bset_test_epoch))


    print("Model Train Success !")
    print(f'Bset val epoch: {bset_val_epoch}, ACC: {best_val_acc}.')
    print(f'Bset test epoch: {bset_test_epoch}, ACC: {best_test_acc}.')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('experiment configs', add_help=False)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--gpus', default="0", type=str)

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--feature-path', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--num-classes', default=7, type=int)
    parser.add_argument('--input-size', default=2048, type=int)

    parser.add_argument('--alpha', default=0.15, type=float)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"
    os.environ["WANDB_MODE"] = "offline"

    main(args)









