import os
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

from data.dataset_loader import get_datasets
from utils.util import load_config, get_data

def load_model(config):
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, config['trainer']["bnb_4bit_compute_dtype"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['trainer']["use_4bit"],
        bnb_4bit_quant_type=config['trainer']["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config['trainer']["use_nested_quant"],
    )

    if compute_dtype == torch.float16 and config['trainer']["use_4bit"]:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        config['trainer']["model_name"],
        device_map="auto",
        # quantization_config=bnb_config
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=config['trainer']["lora_alpha"],
        lora_dropout=config['trainer']["lora_dropout"],
        r=config['trainer']["lora_r"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['trainer']["model_name"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, peft_config

def train(config):
    model, tokenizer, peft_config = load_model(config)
    training_arguments = TrainingArguments(
        output_dir=config['trainer']["output_dir"],
        per_device_train_batch_size=config['trainer']["per_device_train_batch_size"],
        gradient_accumulation_steps=config['trainer']["gradient_accumulation_steps"],
        optim=config['trainer']["optim"],
        save_steps=config['trainer']["save_steps"],
        logging_steps=config['trainer']["logging_steps"],
        learning_rate=config['trainer']["learning_rate"],
        fp16=config['trainer']["fp16"],
        bf16=config['trainer']["bf16"],
        max_grad_norm=config['trainer']["max_grad_norm"],
        max_steps=config['trainer']["max_steps"],
        warmup_ratio=config['trainer']["warmup_ratio"],
        group_by_length=config['trainer']["group_by_length"],
        lr_scheduler_type=config['trainer']["lr_scheduler_type"],
        report_to=config['trainer']["report_to"],
    )

    
    dataset = get_data(config)
    
    train_text_list = []
    for i in range(len(dataset)):
        template = "Q: {question}\nA: {answer}\n\n"
        question = dataset[i]['source']
        answer = dataset[i]['target']
        train_text_list.append(template.format(question=question, answer=answer))

    temp_dataset = Dataset.from_dict({
                    "text": train_text_list,
                }).with_format("torch")


    trainer = SFTTrainer(
        model=model,
        train_dataset=temp_dataset,
        # peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=config['trainer']["max_seq_length"],
        tokenizer=tokenizer,
        args=training_arguments,
        packing=config['trainer']["packing"],
    )

    trainer.train()
    if not os.path.exists(config['trainer']["output_dir"]):
        os.makedirs(config['trainer']["output_dir"])
    trainer.model.save_pretrained(config['trainer']["output_dir"])

