if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# export CUDA_VISIBLE_DEVICES=4

model_name=ConvTimeNet

epochs=100
patience=10

declare -a datasets=(CT PD SAD) #  

patch_size=8
dw_ks='7,7,13,13,19,19'
for dataset in ${datasets[@]}; do
    python -u run.py \
    --is_training 1 \
    --root_path ../datas/datasets/$dataset/ \
    --model_id $dataset \
    --batch_size 16 \
    --des 'Exp' \
    --d_model 768 \
    --d_ff 1536 \
    --pretrained_data Large \
    --patch_size $patch_size \
    --dw_ks $dw_ks \
    --learning_rate 0.001 \
    --train_epochs $epochs \
    --patience $patience >logs/Large_$dataset.log 
done

declare -a datasets=(FD)

patch_size=8
dw_ks='19,19,29,29,37,37'
for dataset in ${datasets[@]}; do
    python -u run.py \
    --is_training 1 \
    --root_path ../datas/datasets/$dataset/ \
    --model_id $dataset \
    --batch_size 16 \
    --des 'Exp' \
    --pretrained_data Large \
    --d_model 768 \
    --d_ff 1536 \
    --patch_size $patch_size \
    --dw_ks $dw_ks \
    --learning_rate 0.001 \
    --train_epochs $epochs \
    --patience $patience >logs/Large_$dataset.log 
done
