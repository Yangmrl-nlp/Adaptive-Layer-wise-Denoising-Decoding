export CUDA_VISIBLE_DEVICES=3
cd /mnt/data1/yangmrl/ALW_debug


# dola
python /mnt/data1/yangmrl/ALW_debug/infer.py \
    --llm 'llavanext_8b' \
    --dataset 'mme' \
    --decode-method 'dola' \
    --pope '' \
    --dola 'static'

# vanilla
# python /mnt/data1/yangmrl/ALW_debug/infer.py \
#     --llm 'llavanext_8b' \
#     --dataset 'chair' \
#     --decode-method 'vanilla' \
#     --classifier '' \
#     --tuned-list '' \
#     --tuned-path '' \
#     --pope '' \
#     --dola ''

#alw
# pope
# python /mnt/data1/yangmrl/ALW_debug/infer.py \
#     --llm 'llavanext_8b' \
#     --dataset 'pope' \
#     --decode-method 'alw' \
#     --classifier 'roberta-base' \
#     --tuned-list '/mnt/data1/yangmrl/ALW_debug/ckpts/instructblip_vicuna_7b/pope/random/lr-epoch-bs-1e-05-30-32' \
#     --tuned-path '' \
#     --pope 'random' \
#     --dola ''      \
#     --Prune 'True'


#MME
# python /mnt/data1/yangmrl/ALW_debug/infer.py \
#     --llm 'llava1.5_7b' \
#     --dataset 'mme' \
#     --decode-method 'alw' \
#     --classifier 'roberta-base' \
#     --tuned-list '/mnt/data1/yangmrl/ALW_debug/ckpts/llava1.5_7b/pope/random/lr-epoch-bs-1e-05-30-32' \
#     --tuned-path '' \
#     --pope '' \
#     --dola ''      \
#     --Prune 'True' 

#CHAIR
# python /mnt/data1/yangmrl/ALW_debug/infer.py \
#     --llm 'instruct_vicuna_7b' \
#     --dataset 'chair' \
#     --decode-method 'alw' \
#     --classifier 'roberta-base' \
#     --tuned-list '/mnt/data1/yangmrl/ALW_debug/ckpts/instructblip_vicuna_7b/pope/popular/lr-epoch-bs-1e-05-30-32' \
#     --tuned-path '' \
#     --pope '' \
#     --dola ''      \
#     --Prune 'True' 

