import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

# export HF_HOME=/*/huggingface

# 1、chinese-LLaMA-7B
# tokenizer = LlamaTokenizer.from_pretrained('minlik/chinese-llama-plus-7b-merged')
# model = LlamaForCausalLM.from_pretrained('minlik/chinese-llama-plus-7b-merged').half().to('cuda')

# tokenizer = LlamaTokenizer.from_pretrained('minlik/chinese-llama-plus-7b-merged', cache_dir="/*/LLM/LaWGPT/models/base_models")
# model = LlamaForCausalLM.from_pretrained('minlik/chinese-llama-plus-7b-merged', cache_dir="/*/LLM/LaWGPT/models/base_models").half().to('cuda')

# 2、Chinese-Alpaca-7B
# tokenizer = LlamaTokenizer.from_pretrained('minlik/chinese-alpaca-plus-7b-merged')
# model = LlamaForCausalLM.from_pretrained('minlik/chinese-alpaca-plus-7b-merged').half().to('cuda')

# tokenizer = LlamaTokenizer.from_pretrained('models/base_models/llama-7b-lora-merged/')
# model = LlamaForCausalLM.from_pretrained('models/base_models/llama-7b-lora-merged/').half().to('cuda')

# tokenizer = LlamaTokenizer.from_pretrained("/*/LLM/LaWGPT/save_chinese/")
# print("resize the embedding size by the size of the tokenizer", len(tokenizer))
# model.resize_token_embeddings(len(tokenizer))

# 3、Chinese-LLaMA-2-7B
tokenizer = LlamaTokenizer.from_pretrained('ziqingyang/chinese-llama-2-7b')
model = LlamaForCausalLM.from_pretrained('ziqingyang/chinese-llama-2-7b').half().to('cuda')

model.eval()

text = "MT"

input_ids = tokenizer.encode(text,return_tensors='pt', max_length=4024).to('cuda')
            
with torch.no_grad():
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=200,
        temperature=0.9,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1
    ).cuda()
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output)
with open('./predict_chineses_llama_without_pretrain.txt', "a+", encoding='utf-8') as f:
    f.write("\n{}\n".format(output))