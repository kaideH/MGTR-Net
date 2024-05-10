# MGTR-Net

Code for the paper "**Multi-Teacher Graph Temporal Relational Network for Online Surgical Workflow Recognition**"


## Evaluate the MGTR-Net

- **stage 1: Feature Extraction**

1. [Download](https://drive.google.com/drive/folders/1Rg8B1soyGkr0-24zI0o6BowZIbNvBcTj?usp=share_link) the pre-trained checkpoint file of the feature encoder and put it to `MGTR-Net/ckpts/cholec80/student_encoder.pth`
2. Run the following code to extract the visual features: 

```bash
python feature_extraction.py
```

or you can [download](https://drive.google.com/drive/folders/1TzmSUc2W_BBP5qB1NcP1Defc4C2XbfPl?usp=share_link) our pre-extracted features and put them to `MGTR-Net/features/cholec80/`



- **stage 2: Temporal Reasoning**

1. [Download](https://drive.google.com/drive/folders/1Rg8B1soyGkr0-24zI0o6BowZIbNvBcTj?usp=share_link) the pre-trained checkpoint file of the temporal encoder and put it to `MGTR-Net/ckpts/cholec80/temporal_encoder.pth`
2. Run the following code to generate predictions of the surgical workflow:

```
python eval.py
```



- **Evaluate the MGTR-Net with a 10 second relaxed boundary**

To evaluate the with 10 second relaxed boundaries, move to `MGTR-Net/evaluations/matlab-eval/cholec80/`, then run the following code to calculate the evaluation metrics:

```
matlab Main.m
```



## Train the MGTR-Net

The training code will be released as soon as this manuscript is accepted to publish.

