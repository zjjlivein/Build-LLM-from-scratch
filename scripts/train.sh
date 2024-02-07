#!/bin/bash    
    # --base_model '*/models/base_models/models--minlik--chinese-llama-plus-7b-merged/snapshots/f349700f2d5537b6500c6d9838eff2479902dbdb/' \
    # --base_model '*/models/chinese_llama_plus_lora_7b'\  
    # base 模型使用标准可以参考../reference/模型说明.md
WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=8 --master_port=1235 train.py \
    --base_model '/*/models/chinese_llama_plus_lora_7b/'\
    --data_path './data/train.json' \
    --output_dir './outputs/v7/' \
    --batch_size 256 \
    --micro_batch_size 8 \
    --num_epochs 10 \
    --learning_rate 0.0003 \
    --cutoff_len 1024 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, v_proj, k_proj, o_proj]' \
    --train_on_inputs True \
    --add_eos_token True \
    --wandb_project "MT-llama-pretraing" \
    --wandb_run_name "test_01" \
    --wandb_watch "all"\
    --wandb_log_model "" \
    --group_by_length True >pretrain.log 2>&1
    # --resume_from_checkpoint './outputs/v9/checkpoint-22600/'  选择是否热启