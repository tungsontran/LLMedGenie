import pandas as pd
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
from prompt_template import *

df = pd.read_csv('~/dataset/medical_transcriptions/mtsamples.csv',encoding='utf-8').iloc[:,1:]
df = df[['medical_specialty', 'description', 'transcription', 'sample_name', 'keywords']]
df['medical_specialty'] = df['medical_specialty'].str.strip()
print(df['medical_specialty'].value_counts())
train_data, val_data = train_test_split(df.to_dict('records'), stratify=df['medical_specialty'], test_size=0.1, random_state=42)

# def format_instruct(data):
#     data_instruct = []
#     for record in data:          
#         record = {
#             'System': f"You are a medical record generator. The User is going to ask you to generate a synthetic medical record of type {record['medical_specialty']}, and you will provide the record in JSON format.",
#             'User': [record['medical_specialty']]+[label.strip() for label in record['medical_specialty'].replace('-', '/').split('/')],
#             'Answer': record
#         }
#         data_instruct.append(record)
#     return data_instruct

# train_data = format_instruct(train_data)
# val_data = format_instruct(val_data)
# train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_data))
# val_dataset = Dataset.from_pandas(pd.DataFrame(data=val_data))
train_data = prompt_template_training(train_data)
val_data = prompt_template_training(val_data)
train_dataset = Dataset.from_dict({"text": train_data})
print(train_dataset[0])
print(train_dataset[10])
print(train_dataset[20])

from transformers import AutoTokenizer
MODEL_NAME = "TheBloke/zephyr-7B-beta-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')

def tokenize_sample(sample):
    input = tokenizer(str(sample),
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=1000)

    input_ids = input["input_ids"][0]
    attention_mask = input["attention_mask"][0]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
# train_dataset = train_dataset.map(tokenize_sample)
# output = tokenize_sample(train_dataset[0])
# output = train_dataset[0]
# decoded_output = tokenizer.decode(output['input_ids'], skip_special_tokens=True)
# print(output['input_ids'])
# print(output)
# print(train_dataset[0])

