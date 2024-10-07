import os
import json
import math
import numpy as np
import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import LongformerPreTrainedModel, LongformerModel
from transformers.utils import ModelOutput
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.classification_metrics import postprocess_text_cls, postprocess_text_reg, extract_numbers
from metrics.generation_metrics import create_metric_bleu_rouge_meteor_chatgpt, postprocess_text
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn import init
from utils.util import load_config

@dataclass
class LongformerSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class LongformerPersonalizedClsHead(nn.Module):
    def __init__(self, config, num_users):
        super().__init__()
        self.num_users = num_users
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.dense_W = nn.Parameter(torch.empty(self.num_users, config.hidden_size, config.hidden_size))
        self.dense_b = nn.Parameter(torch.empty(self.num_users, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_W = nn.Parameter(torch.empty(self.num_users, config.hidden_size, config.num_labels))
        self.out_proj_b = nn.Parameter(torch.empty(self.num_users, config.num_labels))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.dense_W, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.dense_W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.dense_b, -bound, bound)
        init.kaiming_uniform_(self.out_proj_W, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.out_proj_W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.out_proj_b, -bound, bound)

        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    def forward(self, hidden_states, user_mask, **kwargs):
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.dense(hidden_states)
        user_dense_W = torch.bmm(user_mask.unsqueeze(0).expand(self.hidden_size,-1,-1), self.dense_W.permute(1,0,2)).transpose(0,1)
        user_dense_b = torch.matmul(user_mask, self.dense_b)
        hidden_states = torch.bmm(hidden_states.unsqueeze(1), user_dense_W).squeeze() + user_dense_b
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # output = self.out_proj(hidden_states)
        user_out_proj_W = torch.bmm(user_mask.unsqueeze(0).expand(self.hidden_size,-1,-1), self.out_proj_W.permute(1,0,2)).transpose(0,1)
        user_out_proj_b = torch.matmul(user_mask, self.out_proj_b)
        output = torch.bmm(hidden_states.unsqueeze(1), user_out_proj_W).squeeze() + user_out_proj_b
        return output

class LongformerForPersonalizedCls(LongformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = LongformerPersonalizedClsHead(self.config, 531)
    
    def update_num_user(self, num_users):
        self.num_users = num_users
        self.classifier = LongformerPersonalizedClsHead(self.config, num_users)
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        user_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerSequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            # global attention on cls token
            global_attention_mask[:, 0] = 1

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, user_mask)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LongformerSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )

class personalization_orm_cls_trainer():
    def __init__(self, config):
        # super().__init__(self, config=config)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["trainer"]["tokenizer_name"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self, config, data_path):
        dataset = []
        self.id_dict = {}
        idx = 0
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data)
                if not data['id'] in self.id_dict:
                    self.id_dict[data['id']] = idx
                    idx += 1
        print(len(self.id_dict))
        samples = []
        labels = []
        user_masks = []
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
                user_mask = [0.0]*len(self.id_dict)
                user_mask[self.id_dict[sample['id']]] = 1.0
                user_masks.append(user_mask)
        elif config['task'] in ['LaMP_4', 'LaMP_5', 'LaMP_6', 'LaMP_7']:
            for idx in tqdm(range(0, len(dataset), config['generator']['num_return_sequences'])):
                # samples = dataset[idx:idx+config['generator']['num_return_sequences']]
                # print(samples)
                # for sample in samples:
                for sample_idx in range(idx, idx+config['generator']['num_return_sequences']):
                    sample = dataset[sample_idx]
                    samples.append(sample['source']+sample['generation'])
                    labels.append(0)
                    user_mask = [0.0]*len(self.id_dict)
                    user_mask[self.id_dict[sample['id']]] = 1.0
                    user_masks.append(user_mask)
                samples.append(sample['source']+sample['target'])
                labels.append(1)
                user_mask = [0.0]*len(self.id_dict)
                user_mask[self.id_dict[sample['id']]] = 1.0
                user_masks.append(user_mask)
        # convert samples into huggingface dataset with Dataset.from_dict
        dataset = Dataset.from_dict({"label": labels, "text": samples, "user_mask": user_masks}).with_format("torch").shuffle(seed=config['seed'])
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
        
        self.model = LongformerForPersonalizedCls.from_pretrained(self.config["trainer"]["model_name"], num_labels=2)
        self.model.update_num_user(num_users=len(self.id_dict))
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
        trainer.model.longformer.save_pretrained(self.config["trainer"]["output_dir"])
        # self.model = trainer.model
    
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
            temp = {}
            temp['id'] = sol['id']
            temp['generation'] = sol['orig_generation']
            temp['target'] = sol['orig_target']
            with open(solution_file, 'a') as f:
                json.dump(temp, f)
                f.write('\n')
    
    def prepare_eval_data(self, config, history_path, query_path):
        dataset = self.prepare_data(self.config, history_path)
        queries = []
        idx = 0
        with open(query_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                queries.append(data)
        query_samples = []
        query_ids = []
        query_targets = []
        query_samples_orig = []
        query_targets_orig = []
        query_user_masks = []
        for idx in tqdm(range(len(queries))):
            sample = queries[idx]
            query_samples.append(sample['source']+sample['generation'])
            query_ids.append(sample['id'])
            query_targets.append(sample['source']+sample['target'])
            query_samples_orig.append(sample['generation'])
            query_targets_orig.append(sample['target'])
            user_mask = [0.0]*len(self.id_dict)
            user_mask[self.id_dict[sample['id']]] = 1.0
            query_user_masks.append(user_mask)
        # convert samples into huggingface dataset with Dataset.from_dict
        queries = Dataset.from_dict({"id": query_ids, "generation": query_samples, "target": query_targets, "orig_generation": query_samples_orig, "orig_target": query_targets_orig, "user_mask": query_user_masks}).with_format("torch")
        return dataset, queries

    def fit_history(self):
        self.eval_history, self.eval_dataset = self.prepare_eval_data(self.config, self.config['eval_history'], self.config['eval_query'])
        self.eval_history = self.eval_history.map(self.preprocess_function, batched=True)
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
        self.model.update_num_user(num_users=len(self.id_dict))
        self.model.config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        self.model.config.eos_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
        for param in self.model.longformer.parameters():
            param.requires_grad = False
        print(self.tokenizer.pad_token, self.tokenizer.encode(self.tokenizer.pad_token), self.model.config.pad_token_id, self.model.config.eos_token_id)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.eval_history,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()
        if not os.path.exists(self.config["trainer"]["output_dir"]):
            os.makedirs(self.config["trainer"]["output_dir"])
        try:
            trainer.model.save_pretrained(self.config["trainer"]["output_dir"])
        except:
            raise ValueError("Model not saved")

    def guided_inference(self):
        self.fit_history()
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
        self.model = LongformerForPersonalizedCls.from_pretrained(self.config["trainer"]["model_name"]).to(self.device)
        self.model.config.pad_token_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
    
    def direct_inference(self):
        query_path = self.config['eval_query']
        queries = []
        idx = 0
        with open(query_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                queries.append(data)
        query_samples = []
        query_ids = []
        query_targets = []
        query_samples_orig = []
        query_targets_orig = []
        query_user_masks = []
        for idx in tqdm(range(len(queries))):
            sample = queries[idx]
            query_samples.append(sample['source']+sample['generation'])
            query_ids.append(sample['id'])
            query_targets.append(sample['source']+sample['target'])
            query_samples_orig.append(sample['generation'])
            query_targets_orig.append(sample['target'])
            user_mask = [0.0]*len(self.id_dict)
            user_mask[self.id_dict[sample['id']]] = 1.0
            query_user_masks.append(user_mask)
        # convert samples into huggingface dataset with Dataset.from_dict
        queries = Dataset.from_dict({"id": query_ids, "generation": query_samples, "target": query_targets, "orig_generation": query_samples_orig, "orig_target": query_targets_orig, "user_mask": query_user_masks}).with_format("torch")
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        def tokenized_dataset(data):
            return self.tokenizer(data["generation"], truncation=True)
        self.tokenized_dataset = self.eval_dataset.map(tokenized_dataset, batched=True)
        self.tokenized_dataset = self.tokenized_dataset.remove_columns(["id", "generation", "target", "orig_generation", "orig_target"])
        self.dataloader = DataLoader(self.tokenized_dataset, batch_size=self.config["trainer"]["per_device_eval_batch_size"], collate_fn=data_collator, shuffle=False)
        self.get_reward_score()
        self.select_and_save()