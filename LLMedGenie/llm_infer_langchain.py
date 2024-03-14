from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTQConfig,
    pipeline
)

from LLMedGenie.utils import get_model_name
from LLMedGenie.prompt_template import prompt_template_inference
from LLMedGenie.project_path import ADAPTER_DIR, OUTPUT_DIR, CONFIG_DIR, CHECKPOINT_DIR, ROOT_DIR
from LLMedGenie.request_template import InferenceRequest

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import Dataset
from pydantic import BaseModel
from typing import List
from datetime import datetime
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import uvicorn
import asyncio
import tqdm
import pandas as pd
import json
import torch
import wandb
import os
import time
import queue
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='LLM Backend Service')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port for the application')
    parser.add_argument('--model-name', type=str, help='Huggingface model name',
                        default='TheBloke/zephyr-7B-beta-GPTQ')
    parser.add_argument('--use-adapter', type=bool,
                        help='Use PEFT adapter', default=False)

    return parser.parse_args()


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
