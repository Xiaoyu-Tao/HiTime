if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

export CUDA_VISIBLE_DEVICES=0

dataset=SRS1

patch_size=32
dw_ks='37,37,43,43,53,53'

python -u run.py \
    --is_training 1 \
    --root_path ../datas/datasets/$dataset/ \
    --model_id $dataset \
    --lora_rank 32 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --lora_target_modules 'q,v' \
    --test_batch_size 16 \
    --batch_size 16 \
    --des 'Exp' \
    --d_model 768 \
    --d_ff 1536 \
    --pretrained_data Small \
    --patch_size $patch_size \
    --dw_ks $dw_ks \
    --warmup_lr 0.00001 \
    --init_lr 0.0002 \
    --train_epochs 100 \
    --patience 10 >logs/grid_$dataset.log 