import os
from llm_transcript_generator.utils import get_project_root

ROOT_DIR = get_project_root()
ADAPTER_DIR = os.path.join(ROOT_DIR,'adapters')
OUTPUT_DIR = os.path.join(ROOT_DIR,'output')
CONFIG_DIR = os.path.join(ROOT_DIR,'config')
CHECKPOINT_DIR = os.path.join(ROOT_DIR,'checkpoints')
DATASET_DIR = os.path.join(ROOT_DIR,'dataset')

