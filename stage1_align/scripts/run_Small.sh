if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# export CUDA_VISIBLE_DEVICES=4

model_name=ConvTimeNet

epochs=100
patience=10

declare -a datasets=(EP HB NATOPS SRS2)

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
    --pretrained_data Small \
    --patch_size $patch_size \
    --dw_ks $dw_ks \
    --learning_rate 0.001 \
    --train_epochs $epochs \
    --patience $patience >logs/Small_$dataset.log 
done

declare -a datasets=(PEMS_SF)

patch_size=16
dw_ks='19,19,29,29,37,37'
for dataset in ${datasets[@]}; do
    python -u run.py \
    --is_training 1 \
    --root_path ../datas/datasets/$dataset/ \
    --model_id $dataset \
    --batch_size 16 \
    --des 'Exp' \
    --d_model 768 \
    --d_ff 1536 \
    --pretrained_data Small \
    --patch_size $patch_size \
    --dw_ks $dw_ks \
    --learning_rate 0.001 \
    --train_epochs $epochs \
    --patience $patience >logs/Small_$dataset.log 
done

declare -a datasets=(SRS1) 

patch_size=32
dw_ks='37,37,43,43,53,53'
for dataset in ${datasets[@]}; do
    python -u run.py \
    --is_training 1 \
    --root_path ../datas/datasets/$dataset/ \
    --model_id $dataset \
    --batch_size 16 \
    --des 'Exp' \
    --d_model 768 \
    --d_ff 1536 \
    --pretrained_data Small \
    --patch_size $patch_size \
    --dw_ks $dw_ks \
    --learning_rate 0.001 \
    --train_epochs $epochs \
    --patience $patience >logs/Small_$dataset.log 
done

