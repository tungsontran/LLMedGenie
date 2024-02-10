from transformers import BitsAndBytesConfig
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer
from transformers import EarlyStoppingCallback
from transformers import DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import Dataset
from utils import *

import pandas as pd
import json
import torch
import wandb
import os

PROJECT_DIR_PATH = ''
OUTPUT_PATH = os.path.join(PROJECT_DIR_PATH, 'models')
PROJECT_NAME = 'zephyr-7b-beta_medical_transcription'
MODEL_NAME = "TheBloke/zephyr-7B-beta-GPTQ"

wandb.login()
wandb.init(project=PROJECT_NAME,mode="disabled")

###################################################################
# Model Preparation
###################################################################

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],                   
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto')
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# print_trainable_parameters(model)

# Hyperparameters.
MAXLEN=1600
BATCH_SIZE=6
GRAD_ACC=4
WARMUP=100
STEPS=1000
OPTIMIZER='paged_adamw_8bit'                                # Use paged optimizer to save memory
LR=4e-5                                                     # Use value slightly smaller than pretraining lr value & close to LoRA standard

# Setup Callbacks.
early_stop = EarlyStoppingCallback(10, 1.15)

training_args = TrainingArguments(  output_dir=OUTPUT_PATH,
                                    per_device_train_batch_size=BATCH_SIZE,
                                    gradient_accumulation_steps=GRAD_ACC,
                                    warmup_steps=WARMUP,
                                    max_steps=STEPS,
                                    optim=OPTIMIZER,
                                    learning_rate=LR,
                                    bf16=True,                           
                                    logging_steps=1,
                                    report_to='wandb',
                                    load_best_model_at_end=True,
                                    evaluation_strategy='steps',
                                    metric_for_best_model='eval_loss',
                                    greater_is_better=False,
                                    remove_unused_columns=False,
                                    eval_steps=10,
                                    save_steps=10,
                                    save_total_limit=2)

###################################################################
# Data Preparation
###################################################################

df = pd.read_csv('~/dataset/medical_transcriptions/mtsamples.csv',encoding='utf-8').iloc[:,1:]
df = df[['medical_specialty', 'description', 'transcription', 'sample_name', 'keywords']]
df['medical_specialty'] = df['medical_specialty'].str.strip()

train_data, val_data = train_test_split(df.to_dict('records'), stratify=df['medical_specialty'], test_size=0.1, random_state=42)

def format_instruct(data):
    data_instruct = []
    for record in data:          
        record = {
            'User': [record['medical_specialty']]+[label.strip() for label in record['medical_specialty'].replace('-', '/').split('/')],
            'System': f"You are a medical record generator. The User is going to ask you to generate a synthetic medical record of type {record['medical_specialty']}, and you will provide the record in JSON format.",            
            'Answer': record
        }
        data_instruct.append(record)
    return data_instruct

train_data = format_instruct(train_data)
val_data = format_instruct(val_data)

def tokenize_sample(sample):
    user_prompt = sample["User"][0]
    system_info = str(sample["System"])
    answer_text = str(sample["Answer"])

    formatted_input = f"{user_prompt} | {system_info} | {answer_text}"

    input = tokenizer(  formatted_input,
                        return_tensors="pt",
                        truncation=True,
                        padding='max_length',
                        pad_to_max_length=True,                                      
                        max_length=MAXLEN)
    input_ids = input["input_ids"][0]
    attention_mask = input["attention_mask"][0]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

train_dataset = Dataset.from_list([tokenize_sample(sample) for sample in train_data])
val_dataset = Dataset.from_list([tokenize_sample(sample) for sample in val_data])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

###################################################################
# Train
###################################################################
trainer = Trainer(  model=model,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=data_collator,
                    args=training_args,                    
                    callbacks=[early_stop])

model.config.use_cache = False  # Silence the warnings.
trainer.train()    
# print(train_dataset[0])


