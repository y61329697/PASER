# bash name: run_model.sh
# description: train and eval the PASER with different parameters

OUT_PATH="./output/checkpoint"
BATCH_SIZE=16
FINETUNE="--finetune"
KPH=1.0
GPU=2
SEED=42
N_EPOCH=100   # you can train more epochs to get better performance like 200 epochs
MODE='mean'   # pool mode: mean, att, att_fusion, t_att, multi_trans
FOLD=1
SCHD='fixed'
KSPK=0.0
FUSE='concat' # fuse mode: concat, cross, cross_simple, project, gate, bi_linear
WARM=0.2
LR=2e-5
OH=0.0
ABLATION=-1   # -1 represent no ablation, 0 represent full ablation, 1 represent ablation decoder, 2 represent ablation SE module
FIXCOS=1      # whether use two schedulers
STRATEGY=4   # train strategy: 1. Grad_Norm 2. DWA 3. uncertainty weight 4. dynamic task priority(DTP)
FREEZE_ALL=0  # freeze all parameters of pretrain model

python -u main.py --out_path $OUT_PATH --batch_size $BATCH_SIZE $FINETUNE --train_strategy $STRATEGY\
                --gpu $GPU --seed $SEED --n_epoch $N_EPOCH --mode $MODE --fixcos $FIXCOS --freeze_all $FREEZE_ALL\
                --weight_phoneme $KPH --fold_id $FOLD --schd_mode $SCHD  --ablation_level $ABLATION \
                --spk_weight $KSPK --fuse_mode $FUSE --warmup_ratio $WARM  --learning_rate $LR --other_head $OH\
                  # --use_sampler
