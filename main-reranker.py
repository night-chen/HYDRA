import os
import warnings
warnings.simplefilter('ignore')
import argparse
from utils.util import load_config
from accelerate.utils import set_seed
import numpy as np
import torch
from reranker.trainer.personalized_trainer import personalization_orm_cls_trainer

def set_seeds(seed):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/reranker/LaMP_5.yaml', type=str, help='Path to the config file')
    parser.add_argument('--debug', default='generation', type=str, help='debug')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    parser.add_argument('--trained', action='store_true', help='trained')
    args = parser.parse_args()

    assert os.path.isfile(args.config), f"Invalid config path: {args.config}"
    config = load_config(args.config)
    config['seed'] = args.seed
    set_seeds(config['seed'])

    if args.debug == 'generation':
        if config['mode'] == 'train':
            from reranker.generator.train_data_gen import generate_openai
        else:
            from reranker.generator.test_data_gen import generate_openai
        generator = generate_openai(config)
        generator = generator.generate()
    elif args.debug == 'async_generation':
        if config['mode'] == 'train':
            from reranker.generator.train_data_gen_async import generate_openai
        else:
            from reranker.generator.test_data_gen_async import generate_openai
        generator = generate_openai(config)
        generator = generator.generate()
    elif args.debug == 'reranker':
        reranker_trainer = personalization_orm_cls_trainer(config)
        if args.trained:
            reranker_trainer.load_pretrained_weights()
        else:
            reranker_trainer.train()
        reranker_trainer.guided_inference()
    elif args.debug == 'inference':
        reranker_trainer = personalization_orm_cls_trainer(config)
        reranker_trainer.load_pretrained_weights()
        reranker_trainer.direct_inference()
    elif args.debug == 'direct':
        reranker_trainer = personalization_orm_cls_trainer(config)
        if args.trained:
            reranker_trainer.load_pretrained_weights()
        else:
            reranker_trainer.train()
        reranker_trainer.direct_inference2()
    elif args.debug == 'reranker-gen':
        reranker_trainer = personalization_orm_cls_trainer(config)
        if args.trained:
            reranker_trainer.load_pretrained_weights()
        else:
            reranker_trainer.train()
        reranker_trainer.pred_inference('train')
        reranker_trainer.guided_inference()
        reranker_trainer.pred_inference('test')