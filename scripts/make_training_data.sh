export CUDA_VISIBLE_DEVICES=0

cd /mnt/data2/yangmrl/project/ALD^2
model='llavanext_8b'


declare -a lr_list=(
    1e-05
    # 2e-05
    # 5e-05
    # 0.0001
)

for lr in "${lr_list[@]}"; do
    # 'textvqa'
    python -u ./train.py \
        --classifier 'roberta-base' \
        --dataset 'pope' \
        --llm ${model} \
        --decode-method 'alw' \
        --epoch 30 \
        --batch-size 32 \
        --lr ${lr} \
        --save-every 100 \
        --print-every 100 \
        --pope adversarial
done
