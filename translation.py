from transformers import (
    AutoModel,
    AutoTokenizer,
)
from tqdm import trange
import json
import random


def translation(model, tokenizer, text):
    prompt = "你是一名资深的翻译，请将%s由英文翻译成中文。"
    max_length = 2048
    top_p = 0.8
    temperature = 0
    _input = prompt % text
    history = []
    response, history = model.chat(tokenizer, _input, history, max_length=max_length, top_p=top_p, temperature=temperature, do_sample=False)
    return response



if __name__ == "__main__":
    model_name_or_path = "/home/qianq/model/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()
    with open("/home/qianq/data/balance_mimic_pneumonia/train_metadata.jsonl", "r") as f:
        lines = f.readlines()
    res = []
    prompt_temp = [
        '通过这张胸部X光影像可以诊断出什么？',
        '这张图片的背景里有什么内容？',
        '详细描述一下这张图片',
        '看看这张图片并描述你注意到的内容',
        '请提供图片的详细描述',
        '你能为我描述一下这张图片的内容吗？'
    ]
    with open("/home/qianq/data/balance_mimic_pneumonia/train_metadata_final.jsonl", 'w+') as f:
        for i in trange(len(lines)):
            line = json.loads(lines[i])
            prompt = random.choice(prompt_temp)
            data = {
                "file_name": line["file_name"],
                'prompt': prompt,
                'label': translation(model, tokenizer, line['impression']),
                'Pneumonia': line['Pneumonia']
            }
            add_data = {
                "file_name": line["file_name"],
                'prompt': "通过这张胸部X光影像可以诊断出肺炎吗？请回答是或否：",
                'label': "是" if line['impression'] else "否",
                'Pneumonia': line['Pneumonia'],
            }
            f.write(json.dumps(data))
            f.write("\n")
            f.write(json.dumps(add_data))
            f.write("\n")

    with open("/home/qianq/data/balance_mimic_pneumonia/test_metadata.jsonl", "r") as f:
        lines = f.readlines()

    with open("/home/qianq/data/balance_mimic_pneumonia/test_metadata_final.jsonl", 'w+') as f:
        for i in trange(len(lines)):
        # for i in trange(10):
            line = json.loads(lines[i])
            prompt = random.choice(prompt_temp)
            data = {
                "file_name": line["file_name"],
                'prompt': prompt,
                'label': translation(model, tokenizer, line['impression']),
                'Pneumonia': line['Pneumonia']
            }
            f.write(json.dumps(data))
            f.write("\n")