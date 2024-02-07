from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as model
import sentencepiece as sp
import argparse
import os

if __name__ == '__main__':
    # Load arguments
    parser = argparse.ArgumentParser()
    # RuntimeError: Internal: [MASK] is already defined. Caused by version mismatch between model and the plus model.
    parser.add_argument('--load_path', default='/*/huggingface/hub/models--minlik--chinese-llama-plus-7b-merged/snapshots/f349700f2d5537b6500c6d9838eff2479902dbdb/tokenizer.model', type=str)
    parser.add_argument('--save_dir', default='../save_chinese/', type=str)
    parser.add_argument('--voc_path', default='../resources/MT_vocab.txt', type=str)
    args = parser.parse_args()


    LOAD_PATH = args.load_path
    SAVE_DIR = args.save_dir
    VOC_PATH = args.voc_path
    
    # Load pre-trained llama tokenizer and sentencepiece model
    llama_spm = model.ModelProto()
    llama_spm.ParseFromString(open(LOAD_PATH, "rb").read())

    # show size of llama's vocabulary
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
    print(f"Size of initial llama's vocabulary: {len(llama_spm_tokens_set)}")
    
    # Load custom vocabulary
    new_tokens = open(VOC_PATH, "r").read().split("\n")    
    for token in new_tokens:
        if token not in llama_spm_tokens_set:
            new_token = model.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            llama_spm.pieces.append(new_token)
    print(f"Size of merged llama's vocabulary: {len(llama_spm.pieces)}")

    # save
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_MODEL_PATH = os.path.join(SAVE_DIR, 'tokenizer.model')
    SAVE_VOCAB_PATH = os.path.join(SAVE_DIR, 'tokenizer.vocab')
    with open(SAVE_MODEL_PATH, 'wb') as f:
        f.write(llama_spm.SerializeToString())
    with open(SAVE_VOCAB_PATH, 'w')  as f:
        f.writelines([f'{token.piece} {token.score}\n' for token in llama_spm.pieces])
    tokenizer = LlamaTokenizer(SAVE_MODEL_PATH)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f'New llama tokenizer and spm has been saved to {SAVE_DIR}')

    # test
    llama_tokenizer_old = LlamaTokenizer.from_pretrained(LOAD_PATH)
    llama_tokenizer_new = LlamaTokenizer.from_pretrained(SAVE_DIR)
    text = '''MT'''
    
    print(f'Size of old vocabulary: {llama_tokenizer_old.vocab_size}')
    print(f'Size of new vocabulary: {llama_tokenizer_new.vocab_size}')
    print('All special tokens and ids in new llama:')
    print(llama_tokenizer_new.all_special_tokens)
    print(llama_tokenizer_new.all_special_ids)
    print(llama_tokenizer_new.special_tokens_map)

    print(f'Text:\n{text}')
    print(f'Tokenized by LLaMA tokenizer:\n {llama_tokenizer_old.tokenize(text)}')
    # print(f'Tokenized by NEW LLaMA tokenizer:\n\n {llama_tokenizer_old(text)}')
    print(f'Tokenized by NEW LLaMA tokenizer:\n {llama_tokenizer_new.tokenize(text)}')
    # print(f'Tokenized by NEW LLaMA tokenizer:\n {llama_tokenizer_new(text)}')

    # print(f'Tokenized by LLaMA tokenizer:\n {llama_tokenizer_old.convert_tokens_to_ids("百度")}')