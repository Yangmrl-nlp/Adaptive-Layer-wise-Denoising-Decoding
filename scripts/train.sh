export CUDA_VISIBLE_DEVICES=3

cd /mnt/data1/yangmrl/ALW_debug
model='llava1.5_7b'


declare -a lr_list=(
    1e-05
    # 2e-05
    # 5e-05
    # 0.0001
)

for lr in "${lr_list[@]}"; do
    # 'textvqa'
    python -u ./train.py \
        --classifier 'roberta-large' \
        --dataset 'pope' \
        --llm ${model} \
        --decode-method 'alw' \
        --epoch 30 \
        --batch-size 16 \
        --lr ${lr} \
        --save-every 100 \
        --print-every 100 \
        --pope random
done