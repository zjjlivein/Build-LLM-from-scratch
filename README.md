# Build-LLM-from-scrartch
本项目的几点描述：

1、从数据构造，预训练，微调，DPO(RLHF),启动gradio服务推理一步步构建领域大模型
2、利用业务数据训练了一个基于领域数据的大模型MT，该模型能进行简单的业务问答，逻辑复杂prompt的效果不是很好，还在不断优化中。
3、主要代码参考了 https://github.com/pengxiao-song/LaWGPT/tree/main 

## 预训练模型、数据准备

### 下载预训练模型
代码参考：
tools/download_base_model.py

### 准备数据

1、数据格式参考：
train_data.json
finetune_data.json
infer_data.json

2、templates参考：
alpaca.json
law_template.json

3、扩展词表
训练词表模型： 参考https://zhuanlan.zhihu.com/p/630696264?utm_id=0
```
spm_train --input=1.txt --model_prefix=/*/LLM/LaWGPT/models/tokenizer/MT-tokenizer --vocab_size=4000 --character_coverage=0.9995 --model_type=bpe
```

4、合并词表参考：tools/merge_vocab.py


### 预训练模型
scripts/train.sh
### 精调模型
scripts/finetune.sh
### 推理
参考 infer.py
### 启动服务预测
```
python webui.py \
    --load_8bit False \
    --base_model '/*/huggingface/hub/models--minlik--chinese-llama-plus-7b-merged/snapshots/f349700f2d5537b6500c6d9838eff2479902dbdb/' \
    --lora_weights './outputs/v5/checkpoint-24000' \
    --prompt_template "no_template" \
    --server_name "0.0.0.0" \
    --share_gradio True
```



