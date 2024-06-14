CUDA_VISIBLE_DEVICES=1
python train.py \
--dataset  'Attack-Cora' \
--attack   'Meta_Self-0.25' \
--task 'nodecls' \
--mode 'tranductive' \
--model_name 'FilterMGAE'

# unit test
# --dataset                 'Unit-Cora_ml' \
# --adaptive_attack_model    'jaccard_gcn' \
# --split                    0 \
# --scenario                 'poisoning' \

# common attack
# --dataset  'Attack-Citeseer' \
# --attack   'Meta_Self-0.25' \

# --train_size 0.1 \
# --val_size 0.1 \
# --test_size 0.8 \
# --group 1 \
# --use_g1_split
