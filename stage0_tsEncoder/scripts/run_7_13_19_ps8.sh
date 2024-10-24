if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

export CUDA_VISIBLE_DEVICES=0

model_name=ConvTimeNet

declare -a datasets=(CT EP HB NATOPS SAD SRS2) 


patch_size=8
dw_ks='7,7,13,13,19,19'
for dataset in ${datasets[@]}; do
    python -u run.py \
    --is_training 1 \
    --root_path ../datas/datasets/stage_1/$dataset/ \
    --model_id $dataset \
    --model $model_name \
    --e_layers 3 \
    --batch_size 16 \
    --d_model 768 \
    --d_ff 1536 \
    --top_k 3 \
    --dw_ks $dw_ks \
    --patch_size $patch_size \
    --des 'Exp' \
    --itr 3 \
    --learning_rate 0.001 \
    --train_epochs 100 \
    --patience 100 >logs/$model_name'_'$dataset.log 
done
