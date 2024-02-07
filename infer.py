from unittest import result
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel,PeftConfig

peft_model_id = "./outputs/v7/checkpoint-6000"
# load lora config
config = PeftConfig.from_pretrained(peft_model_id)
# load base model
model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path)
# merge lora & base model
model = PeftModel.from_pretrained(model, peft_model_id)

#1、load base tokenizer
tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path)

#2、load updated tokenizer models
# tokenizer = LlamaTokenizer.from_pretrained("/*/LaWGPT/save_chinese/")
# print("resize the embedding size by the size of the tokenizer", len(tokenizer))
# model.resize_token_embeddings(len(tokenizer))

# text = '''新建'''
# print(f'Tokenized by NEW LLaMA tokenizer:\n {tokenizer.tokenize(text)}')

device = "cuda"
model = model.to(device)
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()

if torch.__version__ >= "2":
    model = torch.compile(model)

input_text="如何新建MT任务？"
inputs = tokenizer(input_text, max_length=1024, return_tensors="pt", truncation=True)

# Inference: https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/text_generation
with torch.no_grad():
      outputs = model.generate(
        input_ids=inputs["input_ids"].to("cuda"), 
        max_new_tokens=200,
        temperature=0.8,
        top_k=40,
        top_p=0.8,
        num_beams=1,
        repetition_penalty=1,
        output_scores=True)
      results = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
      print(results)
      with open('./predict_chineses_llama_v4.txt', "a+", encoding='utf-8') as f:
          f.write("\n{}\n".format(results))
