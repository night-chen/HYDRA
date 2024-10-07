import os
import warnings
warnings.simplefilter('ignore')
from transformers import logging
logging.set_verbosity_error()
import torch
import numpy as np
import argparse
from utils.util import load_config
from accelerate.utils import set_seed
from datasets import load_dataset
from data.dataset_loader import get_datasets
from tqdm import tqdm
from adapter.generate.generate_openai import generate_openai
from adapter.generate.generate import generate, generate_chat
from adapter.trainer.nop_trainer import orm_cls_trainer
from adapter.trainer.personalized_trainer import personalization_orm_cls_trainer

def set_seeds(seed):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/LaMP_2.yaml', type=str, help='Path to the config file')
    parser.add_argument('--debug', default='trainer', type=str, help='debug')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--trained', action='store_true', help='trained')
    parser.add_argument('--style', default='p', type=str, help='style')
    args = parser.parse_args()

    config_path = args.config
    assert os.path.isfile(config_path), f"Invalid config path: {config_path}"

    config = load_config(config_path)
    config['seed'] = args.seed
    # set seeds
    set_seeds(config['seed'])

    if args.debug == 'generation':    
        generator = generate_openai(config)
        generator.generate()
    elif args.debug == 'trainer':
        if args.style == 'nop':
            adapter_trainer = orm_cls_trainer(config)
        else:
            adapter_trainer = personalization_orm_cls_trainer(config)
        if args.trained:
            adapter_trainer.load_pretrained_weights()
        else:
            adapter_trainer.train()
        adapter_trainer.guided_inference()
    elif args.debug == 'direct':
        if args.style == 'nop':
            adapter_trainer = orm_cls_trainer(config)
        else:
            adapter_trainer = personalization_orm_cls_trainer(config)
        if args.trained:
            adapter_trainer.load_pretrained_weights()
        else:
            adapter_trainer.train()
        adapter_trainer.direct_inference()