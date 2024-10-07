from accelerate import Accelerator
from accelerate.utils import gather_object
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, time, json, os
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs
import warnings
warnings.filterwarnings("ignore")

from utils.util import get_data
kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
accelerator = Accelerator(kwargs_handlers=[kwargs])


def prepare_prompts(prompts, tokenizer, batch_size=4):
    batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    batches_tok = []
    tokenizer.padding_side = "left"
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding='longest',
                truncation=False,
                pad_to_multiple_of=8,
                add_special_tokens=False
            ).to("cuda")
        )
    tokenizer.padding_side = "right"
    return batches_tok

def generate(config):
    model_path = config["generator"]["model_name"]
    tokenizer_path = config["generator"]["tokenizer_name"]
    data_frac = config["generator"]["data_frac"]
    batch_size = config["generator"]["batch_size"]
    output_dir = Path(config["generator"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # credentials
    token = config["generator"]["token"]

    # load a base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": accelerator.process_index},
        torch_dtype=torch.bfloat16,
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    # load_data
    data = get_data(config)
    if config["generator"]["frac_len"] > 0:
        sub_len = config["generator"]["frac_len"]
        if sub_len * (data_frac + 1) > len(data):
            data = data[sub_len*data_frac:]
        else:
            data = data[sub_len*data_frac:sub_len*(data_frac+1)]
    
    print(data[0])

    # modification here
    prompts_all = [data[idx]['source'] for idx in range(len(data))]
    prompts_old = [data[int(idx/config["generator"]["num_return_sequences"])]['source'] for idx in range(config["generator"]["num_return_sequences"]*len(data))]
    corrects_all = [data[int(idx/config["generator"]["num_return_sequences"])]['target'] for idx in range(config["generator"]["num_return_sequences"]*len(data))]

    print(len(prompts_old), len(corrects_all))
    # sync GPUs and start the timer
    accelerator.wait_for_everyone()
    start = time.time()

    # divide the prompt list onto the avilable GPUs
    with accelerator.split_between_processes(prompts_all) as prompts:
        results = []
        prompt_batches = prepare_prompts(prompts, tokenizer, batch_size)
        for prompts_tokenized in tqdm(prompt_batches):
            # set max_new_tokens smaller for faster inference
            outputs_tokenized = model.generate(**prompts_tokenized, max_new_tokens=config["generator"]["max_length"], pad_token_id=tokenizer.eos_token_id, do_sample=True, num_return_sequences=config["generator"]["num_return_sequences"])
            inputs_tokenized = prompts_tokenized["input_ids"].repeat_interleave(config["generator"]["num_return_sequences"], dim=0)

            # remove prompt from gen. tokens
            outputs_tokenized = [tok_out[len(tok_in):] for tok_in, tok_out in zip(inputs_tokenized, outputs_tokenized)]

            outputs = tokenizer.batch_decode(outputs_tokenized)
            # print(outputs.shape)
            results.extend(outputs)

    # collect results from all the GPUs and remove paddings
    results_gathered = gather_object(results)
    results = [r.replace(tokenizer.eos_token, "").lstrip() for r in results_gathered]
    print(len(results))
    # input()
    if accelerator.is_local_main_process:
        timediff = time.time() - start
        print(f"Time elapsed: {timediff}")

        # collecting data
        for idx in range(len(corrects_all)):
            d = {"source": prompts_old[idx], "target": corrects_all[idx], "generation": results[idx]}
            file_name = f"{config['generator']['output_dir']}/{config['generator']['data_frac']}_{config['mode']}.jsonl"
            with open(file_name, 'a') as f:
                json.dump(d, f)
                f.write('\n')

import transformers

def generate_chat(config):
    data = get_data(config)
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    for idx in tqdm(range(len(data))):
        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": data[idx]["source"]},
        ]

        prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        generation = outputs[0]["generated_text"][len(prompt):]
        d = {"source": data[idx]['source'], "target": data[idx]['target'], "generation": generation}
        file_name = f"{config['generator']['output_dir']}/{config['generator']['data_frac']}_{config['mode']}_chat.jsonl"
        with open(file_name, 'a') as f:
            json.dump(d, f)
            f.write('\n')