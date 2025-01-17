import numpy as np
import torch


def graph_construction(video_names, frame_idxs): # frame_idxs = [B, T]
    B, T = frame_idxs.shape

    video_dict = {}
    for clip_idx in range(B):
        video_name = video_names[clip_idx]
        if video_dict.get(video_name) is None:
            video_dict[video_name] = []
        clip_frame_start = frame_idxs[clip_idx][0].item()
        clip_idx_start = clip_idx * T
        clip_idx_end = clip_idx_start + T - 1
        video_dict[video_name].append([clip_frame_start, clip_idx_start, clip_idx_end])

    edge_index = [[],[]]
    for video_name in video_dict.keys():
        clips = video_dict[video_name]
        clips.sort()

        prev_idxs = []
        for clip_frame_start, clip_idx_start, clip_idx_end in clips:
            for prev_idx in prev_idxs:
                edge_index[0].append(prev_idx)
                edge_index[1].append(clip_idx_start)
                
            for i in range(clip_idx_start, clip_idx_end+1):
                for j in range(i+1, clip_idx_end+1):
                    edge_index[0].append(i)
                    edge_index[1].append(j)
            prev_idxs.append(clip_idx_end)

    edge_index = torch.tensor(edge_index).cuda()
    
    return edge_index

