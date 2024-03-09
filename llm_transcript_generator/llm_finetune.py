from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    GPTQConfig
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

from accelerate import Accelerator
device_index = Accelerator().process_index
device_map = {"": device_index}

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import Dataset
from llm_transcript_generator.utils import *
from llm_transcript_generator.prompt_template import prompt_template_training

import pandas as pd
import math
import logging
import json
import evaluate
import torch
import wandb
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

ROOT_DIR = get_project_root()
MODEL_DIR = os.path.join(ROOT_DIR,'models')
CHECKPOINT_DIR = os.path.join(ROOT_DIR,'checkpoints')
MODEL_NAME = "TheBloke/zephyr-7B-beta-GPTQ"

ADAPTER_PATH = os.path.join(MODEL_DIR,'zephyr-7B-beta_medical_transcription_2')
OUTPUT_PATH = os.path.join(CHECKPOINT_DIR, 'zephyr-7B-beta_medical_transcription_2')
OUTPUT_MODEL = os.path.join(MODEL_DIR, 'zephyr-7B-beta_medical_transcription_2')
PROJECT_NAME = 'zephyr-7b-beta_medical_transcription'

wandb.login()
wandb.init(project=PROJECT_NAME)

logger = logging.getLogger(__name__)

###################################################################
# Hyperparameters
###################################################################
MAXLEN=1500
BATCH_SIZE=6
GRAD_ACC=1
WARMUP=10
STEPS=1000
LR=4e-5        

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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                        padding_side = 'left',
                                        truncation_side='left',
                                        model_max_length=MAXLEN,
                                        attn_implementation="flash_attention_2")
tokenizer.pad_token = tokenizer.eos_token

gptq_config=GPTQConfig(bits=4,use_exllama=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=device_map, quantization_config=gptq_config)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(  output_dir=OUTPUT_PATH,
                                    per_device_train_batch_size=BATCH_SIZE,
                                    gradient_accumulation_steps=GRAD_ACC,
                                    gradient_checkpointing=True,
                                    gradient_checkpointing_kwargs={'use_reentrant':False},
                                    warmup_steps=WARMUP,                                    
                                    max_steps=STEPS,
                                    learning_rate=LR,
                                    weight_decay=0.001,
                                    bf16=True,                           
                                    report_to='wandb',                                    
                                    evaluation_strategy='steps',
                                    metric_for_best_model='eval_loss',
                                    greater_is_better=False,
                                    load_best_model_at_end=True,
                                    remove_unused_columns=False,
                                    logging_steps=50,
                                    eval_steps=50,
                                    save_steps=50,
                                    save_total_limit=5,
                                    do_train=True,
                                    do_eval=True)

###################################################################
# Data Preparation
###################################################################

df = pd.read_csv('~/dataset/medical_transcriptions/mtsamples.csv',encoding='utf-8').iloc[:,1:]
df = df[['medical_specialty', 'description', 'transcription', 'sample_name', 'keywords']]
df['medical_specialty'] = df['medical_specialty'].str.strip()

train_data, val_data = train_test_split(df.to_dict('records'), stratify=df['medical_specialty'], test_size=0.1, random_state=42)

train_data = prompt_template_training(train_data)
val_data = prompt_template_training(val_data)
train_dataset = Dataset.from_dict({"text": train_data})
val_dataset = Dataset.from_dict({"text": val_data})

def tokenize_sample(sample):
    input = tokenizer(str(sample),
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=MAXLEN)

    input_ids = input["input_ids"][0]
    attention_mask = input["attention_mask"][0]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

train_dataset = [tokenize_sample(sample) for sample in train_dataset]
val_dataset = [tokenize_sample(sample) for sample in val_dataset]

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

###################################################################
# Train
###################################################################
early_stop = EarlyStoppingCallback(10, 1.1)

trainer = Trainer(  model=model,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    data_collator=data_collator,
                    args=training_args,                    
                    callbacks=[early_stop])

model.config.use_cache = False  # Silence the warnings.

if training_args.do_train:
    train_result = trainer.train()   
    trainer.save_model(OUTPUT_MODEL)

    metrics = train_result.metrics

    max_train_samples = len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if training_args.do_eval:
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(val_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

