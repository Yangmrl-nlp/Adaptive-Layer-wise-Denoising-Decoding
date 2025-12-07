export CUDA_VISIBLE_DEVICES=2
cd /mnt/data2/yangmrl/project/ALD^2

# dola
# python /mnt/data2/yangmrl/project/ALD^2/infer.py \
#     --llm 'llava1.5_7b' \
#     --dataset 'pope' \
#     --decode-method 'dola' \
#     --pope 'random' \
#     --dola 'static' \
#     --max-new-tokens 4

# vanilla
# python /mnt/data2/yangmrl/project/ALD^2/infer.py \
#     --llm 'instructblip_vicuna_7b' \
#     --dataset 'chair' \
#     --decode-method 'vanilla' \
#     --classifier '' \
#     --tuned-list '' \
#     --tuned-path '' \
#     --pope '' \
#     --max-new-tokens 20

#alw
# pope
python /mnt/data2/yangmrl/project/ALD^2/infer.py \
    --llm 'instructblip_vicuna_7b' \
    --dataset 'chair' \
    --decode-method 'alw' \
    --classifier 'roberta-base' \
    --tuned-list '' \
    --tuned-path '/mnt/data2/yangmrl/project/ALD^2/ckpts/llava1.5_7b/pope/random/lr-epoch-bs-1e-05-30-32/100.pth' \
    --pope '' \
    --max-new-tokens 20 \
    --Prune 'True' \


# python /mnt/data2/yangmrl/project/ALD^2/infer.py \
#     --llm 'llavanext_8b' \
#     --dataset 'pope' \
#     --decode-method 'alw' \
#     --classifier 'roberta-base' \
#     --tuned-list '/mnt/data2/yangmrl/project/ALD^2/ckpts/llavanext_8b/roberta-base/pope/random/lr-epoch-bs-1e-05-30-32' \
#     --tuned-path '' \
#     --pope 'popular' \
#     --max-new-tokens 2 \
#     --Prune 'True' \

# python /mnt/data2/yangmrl/project/ALD^2/infer.py \
#     --llm 'llavanext_8b' \
#     --dataset 'pope' \
#     --decode-method 'alw' \
#     --classifier 'roberta-base' \
#     --tuned-list '/mnt/data2/yangmrl/project/ALD^2/ckpts/llavanext_8b/roberta-base/pope/random/lr-epoch-bs-1e-05-30-32' \
#     --tuned-path '' \
#     --pope 'adversarial' \
#     --max-new-tokens 2 \
#     --Prune 'True' \

#MME
# python /mnt/data2/yangmrl/project/ALD^2/infer.py \
#     --llm 'llavanext_8b' \
#     --dataset 'mme' \
#     --decode-method 'alw' \
#     --classifier 'roberta-base' \
#     --tuned-list '' \
#     --tuned-path '/mnt/data2/yangmrl/project/ALD^2/ckpts/llava1.5_7b/pope/popular/lr-epoch-bs-1e-05-30-32/100.pth' \
#     --Prune 'True' 

#CHAIR
# python /mnt/data2/yangmrl/ALD^2/infer.py \
#     --llm 'instruct_vicuna_7b' \
#     --dataset 'chair' \
#     --decode-method 'alw' \
#     --classifier 'roberta-base' \
#     --tuned-list '/mnt/data2/yangmrl/ALD^2/ckpts/instructblip_vicuna_7b/pope/popular/lr-epoch-bs-1e-05-30-32' \
#     --tuned-path '' \
#     --pope '' \
#     --dola ''      \
#     --Prune 'True' 


#Ablation
# portion_list=(0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)


# for portion in "${portion_list[@]}"; do
#     python /mnt/data2/yangmrl/project/ALD^2/infer.py \
#         --llm 'instructblip_vicuna_7b' \
#         --dataset 'pope' \
#         --decode-method 'alw' \
#         --classifier 'roberta-base' \
#         --tuned-list '' \
#         --tuned-path '/mnt/data2/yangmrl/project/ALD^2/ckpts/instructblip_vicuna_7b/pope/random/lr-epoch-bs-1e-05-30-32/100.pth' \
#         --pope 'adversarial' \
#         --dola ''      \
#         --Prune 'True' \
#         --Prune-portion ${portion} \
#         --max-new-tokens 512
# done



#source /mnt/data2/yangmrl/anaconda3/bin/activate Yangmrl
