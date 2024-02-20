CUDA_VISIBLE_DEVICES=0
python train.py \
--dataset  'Attack-Citeseer' \
--attack   'Meta_Self-0.25' \
--task 'nodecls' \
--mode 'tranductive' \
--model_name 'FilterMGAE'
# --train_size 0.1 \
# --val_size 0.1 \
# --test_size 0.8 \
# --group 1 \
# --use_g1_split
