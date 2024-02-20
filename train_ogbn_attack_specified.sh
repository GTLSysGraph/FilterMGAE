TRAIN_MODE=$1

CUDA_VISIBLE_DEVICES=0 python train.py --dataset "Attack-ogbn-arxiv"  --attack  "DICE-0.5"      --mode $1 --model_name FilterMGAE >"./logs/FilterMGAE/FilterMGAE_DICE_ogbn-arxiv_0.5_$1_a800.file" &
CUDA_VISIBLE_DEVICES=0 python train.py --dataset "Attack-ogbn-arxiv"  --attack  "random-0.5"    --mode $1 --model_name FilterMGAE >"./logs/FilterMGAE/FilterMGAE_random_ogbn-arxiv_0.5_$1_a800.file" &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset "Attack-ogbn-arxiv"  --attack  "PRBCD-0.5"     --mode $1 --model_name FilterMGAE >"./logs/FilterMGAE/FilterMGAE_PRBCD_ogbn-arxiv_0.5_$1_a800.file" &
CUDA_VISIBLE_DEVICES=1 python train.py --dataset "Attack-ogbn-arxiv"  --attack  "heuristic-0.5" --mode $1 --model_name FilterMGAE >"./logs/FilterMGAE/FilterMGAE_heuristic_ogbn-arxiv_0.5_$1_a800.file" &