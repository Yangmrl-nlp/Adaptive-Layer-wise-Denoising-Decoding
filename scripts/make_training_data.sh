export CUDA_VISIBLE_DEVICES=1

cd /mnt/data1/yangmrl/ALW_debug
model='llava1.5_7b'


declare -a dataset_list=(
    #'textvqa'
    'pope'
)

for dataset in "${dataset_list[@]}"; do
    # 'textvqa'
    python -u ./make_training_data.py \
        --dataset 'pope' \
        --llm ${model} \
        --pope 'random' \
        --Prune 'True' 

done