import pandas as pd
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

df = pd.read_csv('~/dataset/medical_transcriptions/mtsamples.csv',encoding='utf-8').iloc[:,1:]
df = df[['medical_specialty', 'description', 'transcription', 'sample_name', 'keywords']]
df['medical_specialty'] = df['medical_specialty'].str.strip()
# print(df['medical_specialty'].value_counts())
train_data, val_data = train_test_split(df.to_dict('records'), stratify=df['medical_specialty'], test_size=0.1, random_state=42)

def format_instruct(data):
    data_instruct = []
    for record in data:          
        record = {
            'System': f"You are a medical record generator. The User is going to ask you to generate a synthetic medical record of type {record['medical_specialty']}, and you will provide the record in JSON format.",
            'User': [record['medical_specialty']]+[label.strip() for label in record['medical_specialty'].replace('-', '/').split('/')],
            'Answer': record
        }
        data_instruct.append(record)
    return data_instruct

train_data = format_instruct(train_data)
val_data = format_instruct(val_data)

train_dataset = Dataset.from_pandas(pd.DataFrame(data=train_data))
val_dataset = Dataset.from_pandas(pd.DataFrame(data=val_data))

# print(train_dataset[0])
from transformers import AutoTokenizer
MODEL_NAME = "TheBloke/zephyr-7B-beta-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
input = {
            'System': f"Test system",
            'User': ['A / B','A','B'],
            'Answer': '{"medical_specialty": "A / B","description": "description","transcription": "transcription","sample_name": "sample_name","keywords": "keywords"}'
        }
def tokenize_sample(sample):
    user_prompt = sample["User"][0]
    system_info = str(sample["System"])
    answer_text = str(sample["Answer"])

    formatted_input = f"{user_prompt} | {system_info} | {answer_text}"

    input = tokenizer(formatted_input,
                         return_tensors="pt",
                         truncation=True,
                         padding=True,
                         max_length=4096)
    return input
    
output = tokenize_sample(input)
decoded_output = tokenizer.decode(output['input_ids'][0], skip_special_tokens=True)
print(decoded_output)
