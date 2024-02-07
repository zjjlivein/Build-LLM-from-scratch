#!/bin/bash

# 1.使用huggingface上已经训练好的模型
# python webui.py \
#     --load_8bit False \
#     --base_model 'minlik/chinese-alpaca-plus-7b-merged' \
#     --lora_weights 'entity303/lawgpt-lora-7b-v2' \
#     --prompt_template "law_template" \
#     --server_name "0.0.0.0" \
#     --share_gradio True \


# 2.使用预训练的lora, 把自己的模型放到对应目录即可
python webui.py \
    --load_8bit False \
    --base_model '/*/huggingface/hub/models--minlik--chinese-llama-plus-7b-merged/snapshots/f349700f2d5537b6500c6d9838eff2479902dbdb/' \
    --lora_weights './outputs/v5/checkpoint-24000' \
    --prompt_template "no_template" \
    --server_name "0.0.0.0" \
    --share_gradio True

# 3.使用fintune训练的lora
# python webui.py \
#     --load_8bit False \
#     --base_model '/ssd1/paddlenlp/huggingface/hub/models--minlik--chinese-llama-plus-7b-merged/snapshots/f349700f2d5537b6500c6d9838eff2479902dbdb/' \
#     --lora_weights './outputs/fintune/checkpoint-3000/' \
#     --prompt_template "paddle_template" \
#     --server_name "0.0.0.0" \
#     --share_gradio True