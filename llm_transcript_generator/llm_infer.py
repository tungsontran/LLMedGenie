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
from llm_transcript_generator.prompt_template import prompt_template_inference
from llm_transcript_generator.project_path import ADAPTER_DIR, OUTPUT_DIR, CONFIG_DIR, CHECKPOINT_DIR, ROOT_DIR
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
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


def perform_inference(pipeline, prompt, max_tokens, temperature, top_k, top_p, num_seq, output_file):
    prompt = prompt_template_inference(prompt)

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


class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: float = 100
    temperature: float = 0.7
    top_k: float = 50
    top_p: float = 0.95
    num_seq: int = 1
    output_file: str = None


@app.get("/inference")
async def process_inference_request(payload: InferenceRequest):
    if pipeline is None:
        init_llm()

    try:
        start = time.perf_counter()

        sequences = perform_inference(
            pipeline=pipeline,
            prompt=payload.prompt,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
            top_p=payload.top_p,
            num_seq=payload.num_seq,
            output_file=payload.output_file
        )

        end = time.perf_counter()
        return {'Inference time': round(end_time - start_time, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

##########################################################################################

if __name__ == '__main__':
    args = parse_arguments()
    init_llm()
    uvicorn.run(app, host='127.0.0.1', port=args.port)
