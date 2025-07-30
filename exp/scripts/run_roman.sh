#!/bin/sh

python -m exp.run \
    --add_hp=False \
    --add_lp=False \
    --d=2 \
    --dataset=roman-empire \
    --dropout=0 \
    --early_stopping=2000 \
    --epochs=2000 \
    --folds=3 \
    --hidden_channels=64 \
    --input_dropout=0.7 \
    --layers=5 \
    --lr=0.01 \
    --model=BundleSheaf \
    --second_linear=False \
    --sheaf_decay=0.0 \
    --weight_decay=0.0 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --stop_strategy=acc \
    --entity="${ENTITY}" 