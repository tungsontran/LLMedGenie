from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTQConfig,
    pipeline
)

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import Dataset
from llm_transcript_generator.utils import get_model_name
from llm_transcript_generator.prompt_template import prompt_template_inferrence
from llm_transcript_generator.project_path import ADAPTER_DIR, OUTPUT_DIR, CONFIG_DIR, CHECKPOINT_DIR, ROOT_DIR
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

import uvicorn
import tqdm
import pandas as pd
import json
import torch
import wandb
import os
import time
from datetime import datetime
import argparse

app = FastAPI()

##########################################################################################
# Parse Arguments
##########################################################################################


def parse_arguments():
    parser = argparse.ArgumentParser(description='LLM Backend Service')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for the application')
    parser.add_argument('--model-name', type=str, help='Huggingface model name',
                        default='TheBloke/zephyr-7B-beta-GPTQ')
    parser.add_argument('--use-adapter', type=bool,
                        help='Use PEFT adapter', default=False)

    return parser.parse_args()

##########################################################################################
# Init LLM
##########################################################################################


def init_llm():
    global pipeline
    gptq_config = GPTQConfig(bits=4, use_exllama=True)
    if args.use_adapter:
        ADAPTER_PATH = os.path.join(
            ADAPTER_DIR, get_model_name(args.model_name))
        peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            quantization_config=gptq_config,
            attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path,
                                                  padding_side='left',
                                                  truncation_side='left')
    else:
        MODEL_NAME = args.model_name
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map='auto',
                                                     quantization_config=gptq_config,
                                                     attn_implementation="flash_attention_2")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
                                                  padding_side='left',
                                                  truncation_side='left')

    tokenizer.pad_token = tokenizer.eos_token

    pipeline = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

##########################################################################################
# Inferrence
##########################################################################################


def perform_inference(pipeline, prompt, max_tokens=100, temperature=0.7, top_k=50, top_p=0.95, num_seq=1, output_file=None):
    prompt = prompt_template_inferrence(prompt)

    sequences = pipeline(
        prompt,
        do_sample=True,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_seq
    )

    if output_file:
        with open(f"{OUTPUT_DIR}/{output_file}", "w") as file:
            json.dump(sequences, file, ensure_ascii=False, indent=2)

    return sequences

##########################################################################################
# Inferrence request
##########################################################################################


class InferrenceRequest(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


@app.post("/inferrence")
async def process_inferrence_request(payload: InferrenceRequest):
    if pipeline is None:
        init_llm()

    try:
        sequences = perform_inference(
            pipeline=pipeline,
            prompt=payload.get('prompt'),
            max_tokens=payload.get('max_tokens'),
            temperature=payload.get('temperature'),
            top_k=payload.get('top_k'),
            top_p=payload.get('top_p'),
            num_seq=payload.get('num_seq'),
            output_file=payload.get('output_file')
        )
        return {'sequences': sequences}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == '__main__':
    args = parse_arguments()
    init_llm()
    uvicorn.run(app, host='127.0.0.1', port=args.port)
