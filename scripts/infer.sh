export CUDA_VISIBLE_DEVICES=2
cd /mnt/data2/yangmrl/project/ALD^2

# dola
python /mnt/data2/yangmrl/project/ALD^2/infer.py \
    --llm 'llava1.5_7b' \
    --dataset 'pope' \
    --decode-method 'dola' \
    --pope 'random' \
    --dola 'static' \
    --max-new-tokens 4

# vanilla
python /mnt/data2/yangmrl/project/ALD^2/infer.py \
    --llm 'instructblip_vicuna_7b' \
    --dataset 'chair' \
    --decode-method 'vanilla' \
    --classifier '' \
    --tuned-list '' \
    --tuned-path '' \
    --pope '' \
    --max-new-tokens 20
    
# pope
python /mnt/data2/yangmrl/project/ALD^2/infer.py \
    --llm 'instructblip_vicuna_7b' \
    --dataset 'chair' \
    --decode-method 'alw' \
    --classifier 'roberta-base' \
    --tuned-list '' \
    --tuned-path '/path/to/bset_predictor.pth' \
    --pope '' \
    --max-new-tokens 20 \
    --Prune 'True' \

#MME
python /mnt/data2/yangmrl/project/ALD^2/infer.py \
    --llm 'llavanext_8b' \
    --dataset 'mme' \
    --decode-method 'alw' \
    --classifier 'roberta-base' \
    --tuned-list '' \
    --tuned-path '/path/to/bset_predictor.pth' \
    --Prune 'True' 

#CHAIR
python /mnt/data2/yangmrl/ALD^2/infer.py \
    --llm 'instruct_vicuna_7b' \
    --dataset 'chair' \
    --decode-method 'alw' \
    --classifier 'roberta-base' \
    --tuned-list '/path/to/predictor' \
    --tuned-path '' \
    --pope '' \
    --dola ''      \
    --Prune 'True' 


#source /mnt/data2/yangmrl/anaconda3/bin/activate Yangmrl
