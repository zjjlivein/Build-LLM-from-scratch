#!/bin/bash
export WANDB_MODE=disabled # 禁用wandb

# 使用chinese-alpaca-plus-7b-merged模型在MT_instruction.json数据集上finetune
experiment_name="finetune/v1"
export NCCL_P2P_DISABLE=1
# 多卡数据并行
WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 finetune.py \
    --base_model "./models/merge_models/MT_base-7b/" \
    --data_path "./data/MT_instruction.json" \
    --output_dir "./outputs/"${experiment_name} \
    --batch_size 64 \
    --micro_batch_size 8 \
    --num_epochs 20 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \
    --val_set_size 0 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules "[q_proj, v_proj]" \
    --train_on_inputs True \
    --add_eos_token True \
    --group_by_length False \
    --wandb_project "" \
    --wandb_run_name "" \
    --wandb_watch ""\
    --wandb_log_model "" \
    --resume_from_checkpoint "./outputs/"${experiment_name} \
    --prompt_template_name "MT_template" \