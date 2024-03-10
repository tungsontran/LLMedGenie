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
from llm_transcript_generator.project_path import ADAPTER_DIR, OUTPUT_DIR, CONFIG_DIR, CHECKPOINT_DIR, ROOT_DIR, DATASET_DIR
from llm_transcript_generator.prompt_template import prompt_template_training
from llm_transcript_generator.utils import get_model_name

import pandas as pd
import math
import logging
import json
import evaluate
import torch
import argparse
import wandb
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--model-name', type=str, help='Huggingface model name', default='TheBloke/zephyr-7B-beta-GPTQ')
    parser.add_argument('--max-len', type=int, help='Maximum sequence length', default=100)
    parser.add_argument('--batch-size', type=int, help='Batch size', default=6)
    parser.add_argument('--lr', type=float, help='Learning rate', default=4e-5)
    parser.add_argument('--warmup-steps', type=int, help='Number of warmup steps', default=10)
    parser.add_argument('--steps', type=int, help='Number of training steps', default=1000)
    parser.add_argument('--grad-acc', type=int, help='Number of gradient accumulation steps', default=1)

    return parser.parse_args()

###################################################################
# Model Preparation
###################################################################
def init_llm():
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],                   
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM")

    gptq_config=GPTQConfig(bits=4,use_exllama=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                device_map=device_map, 
                                                quantization_config=gptq_config)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                            padding_side = 'left',
                                            truncation_side='left',
                                            model_max_length=args.max_len,
                                            attn_implementation="flash_attention_2")
    tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(  output_dir=CHECKPOINT_PATH,
                                        per_device_train_batch_size=args.batch_size,
                                        gradient_accumulation_steps=args.grad_acc,
                                        gradient_checkpointing=True,
                                        gradient_checkpointing_kwargs={'use_reentrant':False},
                                        warmup_steps=args.warmup_steps,                                    
                                        max_steps=args.steps,
                                        learning_rate=args.lr,
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

    return model, tokenizer, training_args

###################################################################
# Data Preparation
###################################################################
def tokenize_sample(sample):
    input = tokenizer(str(sample),
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=args.max_len)

    input_ids = input["input_ids"][0]
    attention_mask = input["attention_mask"][0]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

def prepare_data():
    df = pd.read_csv(f'{DATASET_DIR}/mtsamples.csv',encoding='utf-8').iloc[:,1:]
    df = df[['medical_specialty', 'description', 'transcription', 'sample_name', 'keywords']]
    df['medical_specialty'] = df['medical_specialty'].str.strip()

    train_data, val_data = train_test_split(df.to_dict('records'), stratify=df['medical_specialty'], test_size=0.1, random_state=42)

    train_data = prompt_template_training(train_data)
    val_data = prompt_template_training(val_data)

    train_dataset = Dataset.from_dict({"text": train_data})
    val_dataset = Dataset.from_dict({"text": val_data})

    train_dataset = [tokenize_sample(sample) for sample in train_dataset]
    val_dataset = [tokenize_sample(sample) for sample in val_dataset]

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")

    return train_dataset, val_dataset, data_collator

###################################################################
# Train
###################################################################
def train():
    early_stop = EarlyStoppingCallback(10, 1.1)

    trainer = Trainer(  model=model,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        data_collator=data_collator,
                        args=training_args,                    
                        callbacks=[early_stop])

    model.config.use_cache = False  # Silence the warnings.

    logger = logging.getLogger(__name__)   

    if training_args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train()   
        trainer.save_model(ADAPTER_PATH)

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


if __name__ == '__main__':
    args = parse_arguments()

    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, get_model_name(args.model_name))
    ADAPTER_PATH = os.path.join(ADAPTER_DIR, get_model_name(args.model_name))
    PROJECT_NAME = f'{get_model_name(args.model_name)}-medical-transcription'

    wandb.login()
    wandb.init(project=PROJECT_NAME,mode="disabled")  

    model, tokenizer, training_args = init_llm()
    train_dataset, val_dataset, data_collator = prepare_data()
    train()

