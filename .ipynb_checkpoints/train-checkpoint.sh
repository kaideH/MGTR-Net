# Step 1: Train K teacher encoders with the "sequence of clips" strategy
python train_GTR.py --dataset=cholec80 --gpus="0" --epochs=30 --batch-size=4 --sample-rate=25 --lr=1e-3 --num-classes=7 --N=30 --L=10 --cross=1 --image-base="path/of/video/frames" --label-base="path/of/labels"
python train_GTR.py --dataset=cholec80 --gpus="0" --epochs=30 --batch-size=4 --sample-rate=25 --lr=1e-3 --num-classes=7 --N=30 --L=10 --cross=2 --image-base="path/of/video/frames" --label-base="path/of/labels"
python train_GTR.py --dataset=cholec80 --gpus="0" --epochs=30 --batch-size=4 --sample-rate=25 --lr=1e-3 --num-classes=7 --N=30 --L=10 --cross=3 --image-base="path/of/video/frames" --label-base="path/of/labels"
python train_GTR.py --dataset=cholec80 --gpus="0" --epochs=30 --batch-size=4 --sample-rate=25 --lr=1e-3 --num-classes=7 --N=30 --L=10 --cross=4 --image-base="path/of/video/frames" --label-base="path/of/labels"
python train_GTR.py --dataset=cholec80 --gpus="0" --epochs=30 --batch-size=4 --sample-rate=25 --lr=1e-3 --num-classes=7 --N=30 --L=10 --cross=5 --image-base="path/of/video/frames" --label-base="path/of/labels"

# Step 2: Cross mimicking, ensemble K teacher encoders into a student encoder with knowledge distillation
python cross_mimicking.py --dataset=cholec80 --image-base="path/of/video/frames" --label-base="path/of/labels" --gpus="0" --epochs=100 --batch-size=4 --sample-rate=25 --lr=1e-3 --num-classes=7 --N=10 --L=10 --K=5 --T=5 --soft-cls-lambda=1 --feature-lambda=1

# Step 3: Extract spatial features
python feature_extraction.py --dataset="cholec80" --ckpt-path="exps/cholec80/CM/ckpts/best_train_loss.pth" --frame-dir="path/of/video/frames" --label-dir="path/of/labels" --feature-base="exps/cholec80/CM/features"

# Step 4: Train temporal encoder
python train_temporal.py --dataset=cholec80 --gpus="0" --epochs=100 --num-classes=7 --input-size=2048 --feature-path="exps/cholec80/CM/features"

# Step 5: Evaluation
python eval.py --gpus="0" --num-classes=7 --input-size=2048 --feature-path="exps/cholec80/CM/features/test" --ckpt-path="exps/cholec80/temporal/ckpts/best_val_acc.pth"



