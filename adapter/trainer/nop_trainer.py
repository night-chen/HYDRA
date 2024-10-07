import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.classification_metrics import postprocess_text_cls, postprocess_text_reg, extract_numbers
from metrics.generation_metrics import create_metric_bleu_rouge_meteor_chatgpt, postprocess_text

class orm_cls_trainer():
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["trainer"]["tokenizer_name"])
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_data(self, config, data_path):
        dataset = []
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data)
        samples = []
        labels = []
        if config['task'] in ['LaMP_1', 'LaMP_2', 'LaMP_2_movie', 'LaMP_3']:
            for idx in tqdm(range(len(dataset))):
                sample = dataset[idx]
                if config['task'] == 'LaMP_3':
                    answer, prediction = postprocess_text_reg([sample['target']], [sample['generation']])
                else:
                    answer, prediction = postprocess_text_cls([sample['target']], [sample['generation']])
                if answer[0] in prediction[0]:
                    label = 1
                else:
                    label = 0
                samples.append(sample['source']+sample['generation'])
                labels.append(label)
        elif config['task'] in ['LaMP_4', 'LaMP_5', 'LaMP_6', 'LaMP_7']:
            for idx in tqdm(range(0, len(dataset), config['generator']['num_return_sequences'])):
                for sample_idx in range(idx, idx+config['generator']['num_return_sequences']):
                    sample = dataset[sample_idx]
                    samples.append(sample['source']+sample['generation'])
                    labels.append(0)
                samples.append(sample['source']+sample['target'])
                labels.append(1)
        # convert samples into huggingface dataset with Dataset.from_dict
        dataset = Dataset.from_dict({"label": labels, "text": samples}).with_format("torch").shuffle(seed=config['seed'])
        return dataset

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)
    
    def train(self):
        self.train_dataset = self.prepare_data(self.config, self.config['train_dir'])
        self.train_dataset = self.train_dataset.map(self.preprocess_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        training_args = TrainingArguments(
            output_dir=self.config["trainer"]["output_dir"],
            learning_rate=self.config["trainer"]["learning_rate"],
            per_device_train_batch_size=self.config["trainer"]["per_device_train_batch_size"],
            num_train_epochs=self.config["trainer"]["num_train_epochs"],
            weight_decay=self.config["trainer"]["weight_decay"],
            save_strategy=self.config["trainer"]["save_strategy"],
            push_to_hub=self.config["trainer"]["push_to_hub"],
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(self.config["trainer"]["model_name"], num_labels=2)
        self.model.config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        self.model.config.eos_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        print(self.tokenizer.pad_token, self.tokenizer.encode(self.tokenizer.pad_token), self.model.config.pad_token_id, self.model.config.eos_token_id)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        if not os.path.exists(self.config["trainer"]["output_dir"]):
            os.makedirs(self.config["trainer"]["output_dir"])
        trainer.model.save_pretrained(self.config["trainer"]["output_dir"])
        self.model = trainer.model
    
    def get_reward_score(self):
        # pass the dataset through the reward model
        self.solution_scores = []
        num = 0
        for batch in self.dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits.detach().cpu() # B, 2
            # apply softmax on the logits
            logits = torch.nn.functional.softmax(logits, dim=-1)
            scores = logits[:,1]
            self.solution_scores.append(scores)
        self.solution_scores = torch.cat(self.solution_scores, dim=0)
        print(len(self.solution_scores), len(self.eval_dataset))
    
    def select_and_save(self):
        # select the solution with the highest score every num_return_sequences solutions
        selected_solutions = []
        for idx in range(0, len(self.eval_dataset), self.config["generator"]["num_return_sequences"]):
            if idx + self.config["generator"]["num_return_sequences"] < len(self.eval_dataset):
                max_idx = np.argmax(self.solution_scores[idx:idx+self.config["generator"]["num_return_sequences"]])
                selected_solutions.append(self.eval_dataset[idx+int(max_idx)])
            else:
                max_idx = np.argmax(self.solution_scores[idx:])
                selected_solutions.append(self.eval_dataset[idx+int(max_idx)])
        if not os.path.exists(self.config["generator"]["output_dir"]):
            os.mkdir(self.config["generator"]["output_dir"])
        solution_file = f'{self.config["generator"]["output_dir"]}/{self.config["seed"]}_selected.jsonl'
        
        for sol in selected_solutions:
            sol['generation'] = sol['orig_generation']
            sol['target'] = sol['orig_target']
            with open(solution_file, 'a') as f:
                json.dump(sol, f)
                f.write('\n')
    
    def prepare_eval_data(self, config, data_path):
        dataset = []
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data)
        samples = []
        ids = []
        targets = []
        samples_orig = []
        targets_orig = []
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            samples.append(sample['source']+sample['generation'])
            ids.append(sample['id'])
            targets.append(sample['source']+sample['target'])
            samples_orig.append(sample['generation'])
            targets_orig.append(sample['target'])
        # convert samples into huggingface dataset with Dataset.from_dict
        dataset = Dataset.from_dict({"id": ids, "generation": samples, "target": targets, "orig_generation": samples_orig, "orig_target": targets_orig}).with_format("torch")
        return dataset


    def guided_inference(self):
        self.eval_dataset = self.prepare_eval_data(self.config, self.config['eval_dir'])
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        def tokenized_dataset(data):
            return self.tokenizer(data["generation"], truncation=True)
        self.tokenized_dataset = self.eval_dataset.map(tokenized_dataset, batched=True)
        self.tokenized_dataset = self.tokenized_dataset.remove_columns(["id", "generation", "target", "orig_generation", "orig_target"])
        self.dataloader = DataLoader(self.tokenized_dataset, batch_size=self.config["trainer"]["per_device_eval_batch_size"], collate_fn=data_collator, shuffle=False)
        self.get_reward_score()
        self.select_and_save()

    def load_pretrained_weights(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config["trainer"]["output_dir"]).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]