import os
import math
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from reward_model.orm.orm_data import prepare_orm_data
from transformers import LongformerPretrainedModel, LongformerModel
from transformers.utils import ModelOutput
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.nn import init

class LongformerSequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class LongformerPersonalizedClsHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_W = nn.Parameter(torch.empty(config.num_users, config.hidden_size, config.hidden_size))
        self.dense_b = nn.Parameter(torch.empty(config.num_users, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj_W = nn.Parameter(torch.empty(config.num_users, config.hidden_size, config.num_labels))
        self.out_proj_b = nn.Parameter(torch.empty(config.num_users, config.num_labels))
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.dense_W, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.dense_W)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output

class LongformerForPersonalizedCls(LongformerPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = LongformerPersonalizedClsHead(config)
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
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
        logits = self.classifier(sequence_output)

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
    
def vanilla_trainer(config, train_dataset):
    tokenizer = AutoTokenizer.from_pretrained(config["reward_model"]["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=config["reward_model"]["output_dir"],
        learning_rate=config["reward_model"]["learning_rate"],
        per_device_train_batch_size=config["reward_model"]["per_device_train_batch_size"],
        num_train_epochs=config["reward_model"]["num_train_epochs"],
        weight_decay=config["reward_model"]["weight_decay"],
        save_strategy=config["reward_model"]["save_strategy"],
        push_to_hub=config["reward_model"]["push_to_hub"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(config["reward_model"]["model_name"], num_labels=2)
    model.config.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    model.config.eos_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    print(tokenizer.pad_token, tokenizer.encode(tokenizer.pad_token), model.config.pad_token_id, model.config.eos_token_id)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    if not os.path.exists(config["reward_model"]["output_dir"]):
        os.makedirs(config["reward_model"]["output_dir"])
    trainer.model.save_pretrained(config["reward_model"]["output_dir"])