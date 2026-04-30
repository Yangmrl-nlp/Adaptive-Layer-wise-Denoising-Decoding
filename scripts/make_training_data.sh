export CUDA_VISIBLE_DEVICES=1

cd /mnt/data2/yangmrl/project/ALD^2
model='llavanext_8b'


declare -a dataset_list=(
    'random'
    'popular'
    'adversarial'
)

for dataset in "${dataset_list[@]}"; do
    python -u /path/to/make_training_data.py \
        --dataset 'pope' \
        --llm ${model} \
        --pope ${dataset} \
        --Prune 'True' 

done
