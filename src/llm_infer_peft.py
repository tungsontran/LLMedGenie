from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    GPTQConfig,
    pipeline
)
from peft import (
    PeftConfig
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import Dataset
from utils import *
from prompt_template import prompt_template_inferrence

import pandas as pd
import json
import torch
import wandb
import os
import time
from datetime import datetime

start = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")

PROJECT_DIR_PATH = ''

adapters_path = f'./models/zephyr-7B-beta_medical_transcription'
peft_config = PeftConfig.from_pretrained(adapters_path)
gptq_config=GPTQConfig(bits=4,use_exllama=False)

model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, 
                                            torch_dtype=torch.bfloat16,
                                            device_map='auto',
                                            quantization_config=gptq_config,
                                            attn_implementation="flash_attention_2")


tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, padding_side='left',truncation_side='left')
tokenizer.pad_token = tokenizer.eos_token

pipeline = pipeline(
    'text-generation',
    model=model, 
    tokenizer=tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

prompt = prompt_template_inferrence("Cardiovascular / Pulmonary")
num_seq = 20

sequences = pipeline(
    prompt,
    do_sample=True,
    max_new_tokens=1000, 
    temperature=0.2, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=num_seq
)

end = start = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S")

with open("./output/output_peft_test.json", "a") as file:
    json.dump(sequences, file, ensure_ascii=False, indent=2)

# for i in range(num_seq):
#     print(sequences[i]['generated_text'])
print(f'Duration: {end-start}, Start: {start}, End: {end}')