CUDA_VISIBLE_DEVICES=0
python train.py \
--task 'nodecls' \
--mode 'tranductive' \
--model_name 'FilterMGAE' \
--dataset  'Attack-Cora' \
--attack   'Meta_Self-0.25' \
--train_size 0.1 \
--val_size 0.1 \
--test_size 0.8 \
--group 1 \
--use_g1_split

# unit test
# --dataset                  'Unit-Cora_ml' \
# --adaptive_attack_model    'svd_gcn' \
# --split                    1 \
# --scenario                 'poisoning' \
# --budget                   684 \
# --unit_ptb                 0.1349


# common attack
# --dataset  'Attack-Cora' \
# --attack   'Meta_Self-0.25' \

# --train_size 0.1 \
# --val_size 0.1 \
# --test_size 0.8 \
# --group 1 \
# --use_g1_split
