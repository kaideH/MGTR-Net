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

    # diff = torch.norm(outputs - labels, p=2, dim=1)
    # loss = torch.mean(diff)
    return loss


def get_distance_relations(features, edge_index):
    node1 = features[edge_index[0]] # [N, D]
    node2 = features[edge_index[1]]
    return torch.norm(node1 - node2, p=2, dim=1)


def get_angle_relations(features, edge_index, num_nodes):
    triplets = [[], [], []]
    for node_i in range(num_nodes):
        connected_nodes = []
        # node_i point to
        indices = (edge_index[0] == node_i).nonzero().view(-1).tolist()
        connected_nodes.extend(edge_index[1][indices].tolist()) 
        # point to node_i
        indices = (edge_index[1] == node_i).nonzero().view(-1).tolist()
        connected_nodes.extend(edge_index[0][indices])

        for i in range(len(connected_nodes)-1):
            for j in range(i+1, len(connected_nodes)):
                triplets[0].append(connected_nodes[i])
                triplets[1].append(node_i)
                triplets[2].append(connected_nodes[j])

    node1 = features[triplets[0]]
    node2 = features[triplets[1]]
    node3 = features[triplets[2]]
    v1 = node1 - node2 
    v1 = v1 / torch.norm(v1, p=2, dim=1).view(-1, 1)
    v2 = node3 - node2 
    v2 = v2 / torch.norm(v2, p=2, dim=1).view(-1, 1)
    return torch.sum(v1 * v2, dim=1)


def train_epoch(data_loader, teachers, student, criterion, mse, optimizer, args, scaler, data_sub_set):
    student.train()
    for teacher_key in teachers.keys():
        teacher = teachers[teacher_key]
        teacher.eval()

    total_loss = 0.0
    total_soft_cls_loss, total_s_feature_loss, total_t_feature_loss, total_distance_loss = 0, 0, 0, 0
    
    bar = tqdm(total=len(data_loader))
    time_start = datetime.datetime.now()
    inputs, labels, video_idxs, frame_idxs = data_loader.get_next()
    while inputs is not None:

        # step 1: 把输入数据分配给对应的teacher
        feature_bins = {}
        for i in range(1, args.K+1):
            feature_bins[f"cross_{i}"] = []
        
        for idx, video_idx in enumerate(video_idxs):
            for key in feature_bins.keys():
                if video_idx in data_sub_set[key]:
                    feature_bins[key].append(idx)

        # step 2: 跟别计算每个teacher和student的输出结果
        T_outputs, T_s_features, T_t_features = [], [], []
        S_outputs, S_s_features, S_t_features = [], [], []
        T_distance, S_distance = [], []
        for teacher_key in feature_bins.keys():
            teacher = teachers[teacher_key]
            teacher.eval()

            # get current teacher inputs
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

            try:
                with autocast():
                    # teacher 
                    with torch.no_grad():
                        teacher_outputs, teacher_s_features, teacher_t_features = teacher(curr_inputs, edge_index, frame_encodings)
                        teacher_distance = get_distance_relations(teacher_s_features, edge_index).detach()

                    # student
                    student_s_features = student(curr_inputs)
                    student_outputs, student_t_features = teacher.gtm_forward(student_s_features, edge_index, frame_encodings)
                    student_distance = get_distance_relations(student_s_features, edge_index)
                    
                    T_outputs.append(teacher_outputs) # [N, C]
                    T_s_features.append(teacher_s_features) # [N, D]
                    T_t_features.append(teacher_t_features) # [N, D]
                    T_distance.append(teacher_distance)

                    S_outputs.append(student_outputs)
                    S_s_features.append(student_s_features)
                    S_t_features.append(student_t_features)
                    S_distance.append(student_distance)
            except:
                print(traceback.format_exc())

        
        # step 3: 综合计算损失函数
        optimizer.zero_grad()
        loss = 0
        # soft cls loss
        T_outputs = torch.cat(T_outputs, dim=0).detach() # [B, C]
        S_outputs = torch.cat(S_outputs, dim=0)
        soft_cls_loss = criterion(F.log_softmax(S_outputs / args.T, dim=1), F.softmax(T_outputs / args.T, dim=1)) * args.T * args.T

        # spatial feature loss
        T_s_features = torch.cat(T_s_features, dim=0).detach()
        S_s_features = torch.cat(S_s_features, dim=0)
        s_feature_loss = kd_l2_loss(S_s_features, T_s_features)

        # temporal feature loss
        T_t_features = torch.cat(T_t_features, dim=0).detach()
        S_t_features = torch.cat(S_t_features, dim=0)
        t_feature_loss = kd_l2_loss(S_t_features, T_t_features)

        # distance relation loss
        if args.distance_lambda > 0:
            T_distance = torch.cat(T_distance, dim=0).detach()
            S_distance = torch.cat(S_distance, dim=0)
            distance_loss = mse(S_distance, T_distance)
        else:
            distance_loss = torch.tensor(0.0)


        loss = args.soft_cls_lambda * soft_cls_loss + args.s_feature_lambda * s_feature_loss + args.t_feature_lambda * t_feature_loss + args.distance_lambda * distance_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_soft_cls_loss += soft_cls_loss.item()
        total_s_feature_loss += s_feature_loss.item()
        total_t_feature_loss += t_feature_loss.item()
        total_distance_loss += distance_loss.item()

        inputs, labels, video_idxs, frame_idxs = data_loader.get_next()
        bar.set_postfix_str(f"Train Loss={loss.item():.4f}, Soft cls Loss={soft_cls_loss.item():.4f}, Spatial Feature Loss={s_feature_loss.item():.4f}")
        bar.update(1)

    time_spend = (datetime.datetime.now() - time_start).seconds
    average_loss = total_loss / len(data_loader)
    average_soft_cls_loss = total_soft_cls_loss / len(data_loader)
    average_s_feature_loss = total_s_feature_loss / len(data_loader)
    average_t_feature_loss = total_t_feature_loss / len(data_loader)
    average_distance_loss = total_distance_loss / len(data_loader)

    loss_items = [average_loss, average_soft_cls_loss, average_s_feature_loss, average_t_feature_loss, average_distance_loss]

    return loss_items, time_spend


def main(args):
    ## setup experiment
    args.base_path = os.path.join("exps", args.dataset, "CM") 
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


    ## data 
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
        from spatial.datasets.autolaparo.utils import data_sub_set
        from spatial.datasets.autolaparo.clip import build_loader
    elif args.dataset == "GraSP":
        from spatial.datasets.grasp.utils import data_sub_set
        from spatial.datasets.grasp.clip import build_loader
    elif args.dataset == "MultiBypass140":
        from spatial.datasets.mbp140.utils import data_sub_set
        from spatial.datasets.mbp140.clip import build_loader
    elif args.dataset == "Heidelberg":
        from spatial.datasets.heidelberg.utils import data_sub_set
        from spatial.datasets.heidelberg.clip import build_loader
    elif args.dataset == "cholecT50":
        from spatial.datasets.cholecT50.utils import data_sub_set
        from spatial.datasets.cholecT50.clip import build_loader
    elif args.dataset == "Hei-Chole":
        from spatial.datasets.heichole.utils import data_sub_set
        from spatial.datasets.heichole.clip import build_loader
    else:
        raise NotImplementedError(f"unkown dataset {args.dataset}")
    train_loader = build_loader(args, sub_set[f"train_all"], get_label, training=True)
    print("Iters train: {}".format(len(train_loader))) 

    scaler = GradScaler() 

    ## model
    # teacher encoders
    from spatial.feature_encoder import Network
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
    mse = nn.MSELoss()
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.02)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=1e-7)
    

    # training procedure
    best_kd_loss, bset_epoch_loss = 1000, -1
    for epoch in range(1, args.epochs+1):
        # learning rate
        lr = optimizer.state_dict().get("param_groups")[-1].get("lr")
        print(f"Epoch {epoch}, LR: {lr}")


        ## train
        loss_items, time_spend = train_epoch(train_loader, teachers, student, criterion, mse, optimizer, args, scaler, sub_set)
        average_loss, average_soft_cls_loss, average_s_feature_loss, average_t_feature_loss, average_distance_loss = loss_items
        print(f"Epoch {epoch}, Train time: {time_spend}, Train loss: {average_loss:.6f}, Soft cls Loss={average_soft_cls_loss:.6f}, Spatial Feature Loss={average_s_feature_loss:.6f}")

        scheduler.step() # train_loss val_loss

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


"""
python cross_mimicking.py --dataset=cholec80 --gpus="2" --epochs=100 --batch-size=4 --sample-rate=25 --lr=1e-3 --num-classes=7 --L=10 --K=5 \
--T=5 --soft-cls-lambda=100 --s-feature-lambda=1 --t-feature-lambda=0 --distance-lambda=0
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

    parser.add_argument('--K', default=5, type=int)

    parser.add_argument('--T', default=5, type=float)
    parser.add_argument('--soft-cls-lambda', default=100, type=float)
    parser.add_argument('--s-feature-lambda', default=1, type=float)
    parser.add_argument('--t-feature-lambda', default=0, type=float)
    parser.add_argument('--distance-lambda', default=0, type=float)

    parser.add_argument('--num-classes', default=7, type=int)
    parser.add_argument('--cross', default=1, type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpus}"
    main(args)









