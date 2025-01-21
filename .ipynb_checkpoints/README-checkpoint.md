# MGTR-Net
Code for the paper "**Multi-Teacher Graph Temporal Regulation Network for Surgical Workflow Recognition**"

## Data
- We have evaluated our MGTR-Net on eight datasets: Cholec80, Cataract-101, AutoLaparo, MultiBypass140, HeiChole, Heidelberg, GraSP, and CholecT50. 
- To train and test this method, first download the original datasets and parse the surgical videos into discrete video frames. The dataset format should be structured as follows:
```
dataset_name/
├── frames
│   ├── video01
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ...
│   │   └── last_frame_index.jpg
│   ├── video02
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ...
│   │   └── last_frame_index.jpg
│   ├── ...
├── phase_annotations
│   ├── video01-phase.txt
│   ├── video02-phase.txt
│   ├── ...
```
- Taking the Cholec80 dataset as an example, we provide a simple Python script for extracting video frames from surgical videos.
```
python video_2_frames.py --image_save_dir="path/to/save/the/extracted/frames" --video_dir="path/of/the/original/videos"
```

## Train the MGTR-Net
- Simply run `sh train.sh`
- Trained models and logs will be saved in the `exps` folder

## Use our trained models

- **Feature Extraction**
1. [Download](https://drive.google.com/drive/folders/1Rg8B1soyGkr0-24zI0o6BowZIbNvBcTj?usp=share_link) the pre-trained checkpoint file of the feature encoder.
2. Run the following code to extract the visual features: 
```bash
python feature_extraction.py --dataset="your_dataset" --ckpt-path="path/to/the/feature/encoder.pth" --frame-dir="path/of/video/frames" --label-dir="path/of/labels" --feature-base="path/to/save/features"
```
or you can directly [download](https://drive.google.com/drive/folders/1TzmSUc2W_BBP5qB1NcP1Defc4C2XbfPl?usp=share_link) our pre-extracted features.

- **Temporal Reasoning**
1. [Download](https://drive.google.com/drive/folders/1Rg8B1soyGkr0-24zI0o6BowZIbNvBcTj?usp=share_link) the pre-trained checkpoint file of the temporal encoder.
2. Run the following code to generate surgical workflow predictions of each frame:
```
python eval.py --gpus="0" --num-classes=num_classes --input-size=2048 --feature-path="path/to/save/features" --ckpt-path="path/to/the/temporal/encoder.pth"
```