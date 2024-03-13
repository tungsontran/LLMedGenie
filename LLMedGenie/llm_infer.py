from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTQConfig,
    pipeline
)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import Dataset
from LLMedGenie.utils import get_model_name
from LLMedGenie.prompt_template import prompt_template_inference
from LLMedGenie.project_path import ADAPTER_DIR, OUTPUT_DIR, CONFIG_DIR, CHECKPOINT_DIR, ROOT_DIR
from LLMedGenie.request_template import InferenceRequest
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

import uvicorn
import asyncio
import langchain
import tqdm
import pandas as pd
import json
import torch
import wandb
import os
import time
import queue
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


async def inference(pipeline, prompt, max_tokens, temperature, top_k, top_p, num_seq) -> List[dict]:
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

    return sequences


##########################################################################################
# Inferrence request
##########################################################################################


@app.post("/inference/batch")
async def batch_inference(payload: InferenceRequest):
    if pipeline is None:
        init_llm()

    try:
        start = time.perf_counter()

        sequences = await inference(
            pipeline=pipeline,
            prompt=payload.prompt,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_k=payload.top_k,
            top_p=payload.top_p,
            num_seq=payload.num_seq
        )

        end = time.perf_counter()
        if payload.output_file:
            with open(f"{OUTPUT_DIR}/{payload.output_file}", "w") as file:
                json.dump(sequences, file, ensure_ascii=False, indent=2)

        return {'Sequences': sequences, 'Inference time': round(end - start, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


##########################################################################################

if __name__ == '__main__':
    args = parse_arguments()
    init_llm()
    uvicorn.run(app, host='127.0.0.1', port=args.port)
