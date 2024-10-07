import os
from pathlib import Path
import json
from tqdm import tqdm
from utils.credentials import api_key_list
from openai import AzureOpenAI
import random

class generate_openai():
    def __init__(self, config=None):
        self.api_key_list = api_key_list(config["generator"]["openai_credentials"])
        self.api_idx = 0
        self.client = AzureOpenAI(
            api_key=self.api_key_list[self.api_idx]["api_key"], 
            api_version=self.api_key_list[self.api_idx]["api_version"],
            azure_endpoint=self.api_key_list[self.api_idx]["azure_endpoint"]
        )
        self.model = self.api_key_list[self.api_idx]["model"]
        self.config = config
        self.token_usage = {"input": 0, "output": 0}
        self.history_data, self.query_data = self.get_data(config)
        if config['task'] == 'LaMP_2':
            self.question_template = "Which category does this article relate to among the following categories? Just answer with the category name without further explanation. categories: [women, religion, politics, style & beauty, entertainment, culture & arts, sports, science & technology, travel, business, crime, education, healthy living, parents, food & drink] article: {text}"
        elif config['task'] == 'LaMP_2_movie':
            self.question_template = "Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {text}"
        elif config['task'] == 'LaMP_3':
            self.question_template = "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {text}"
        elif config['task'] == 'LaMP_4':
            self.question_template = "Generate a headline for the following article: {text}"
        elif config['task'] == 'LaMP_5':
            self.question_template = "Generate a title for the following abstract of a paper: {text}"

    def switch_api_key(self):
        self.api_idx = (self.api_idx + 1) % len(self.api_key_list)
        self.client = AzureOpenAI(
            api_key=self.api_key_list[self.api_idx]["api_key"],
            api_version=self.api_key_list[self.api_idx]["api_version"],
            azure_endpoint=self.api_key_list[self.api_idx]["azure_endpoint"]
        )
        self.model = self.api_key_list[self.api_idx]["model"]
    
    def query(self, prompt_chat, temp=None, n=None, stop=None, max_tokens=None,):
        # prompt_chat = [
        #     {"role": "user", "content": prompt}
        # ]
        flag = False
        num_trials = 0
        while not flag:
            try:
                raw_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt_chat,
                    max_tokens=self.config['generator']['max_length'] if max_tokens is None else max_tokens,
                    temperature=self.config['generator']['temperature'] if temp is None else temp,
                    frequency_penalty=self.config['generator']['frequency_penalty'],
                    presence_penalty=self.config['generator']['presence_penalty'],
                    stop=self.config['generator']['stop'] if stop is None else stop,
                    n=self.config['generator']['num_return_sequences'] if n is None else n,
                )
                self.token_usage["input"] += raw_response.usage.prompt_tokens
                self.token_usage["output"] += raw_response.usage.completion_tokens

                contents = [choice.message.content.strip() for choice in raw_response.choices]
                flag = True
                if len(contents) == 0:
                    flag = False
                    raise RuntimeError("No response from the API")
            except Exception as e:
                self.switch_api_key()
                flag = False
                num_trials += 1
                print(e)
            if num_trials > 3:
                flag = True
                contents = None
        return contents
    
    def generate(self):
        generation = []
        self.data = self.history_data
        for idx in tqdm(range(len(self.data))):
            prompt_msg = [
                {"role": "user", "content": self.question_template.format(text=self.data[idx]['retrieval'])},
                {"role": "assistant", "content": self.data[idx]['retr_ans']},
                {"role": "user", "content": self.question_template.format(text=self.data[idx]['source'])}
            ]
            # prompt_msg = template.format(source=self.data[idx]['source'], retrieval=self.data[idx]['retrieval'])
            responses = self.query(prompt_msg)
            if responses != None:
                for res in responses:
                    generation.append({"id": self.data[idx]['id'], "source": self.data[idx]['source'], "target": self.data[idx]['target'], "generation": res, "retrieval": self.data[idx]['retrieval'], "retr_ans": self.data[idx]['retr_ans']})
            # if idx == 10:
            #     for jj in range(idx):
            #         print(generation[jj])
            #     input()
        output_dir = Path(self.config['generator']["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        for gen in generation:
            file_name = f"{self.config['generator']['output_dir']}/{self.config['seed']}_{self.config['mode']}_history.jsonl"
            with open(file_name, 'a') as f:
                json.dump(gen, f)
                f.write('\n')
        print(self.token_usage)

        generation = []
        # template = "[CLS] {source} [SEP] {retrieval} [SEP]"
        self.data = self.query_data
        for idx in tqdm(range(len(self.data))):
            prompt_msg = [
                {"role": "user", "content": self.question_template.format(text=self.data[idx]['retrieval'])},
                {"role": "assistant", "content": self.data[idx]['retr_ans']},
                {"role": "user", "content": self.question_template.format(text=self.data[idx]['source'])}
            ]
            # prompt_msg = template.format(source=self.data[idx]['source'], retrieval=self.data[idx]['retrieval'])
            responses = self.query(prompt_msg)
            if responses != None:
                for res in responses:
                    generation.append({"id": self.data[idx]['id'], "source": self.data[idx]['source'], "target": self.data[idx]['target'], "generation": res, "retrieval": self.data[idx]['retrieval'], "retr_ans": self.data[idx]['retr_ans']})
            # if idx == 10:
            #     for jj in range(idx):
            #         print(generation[jj])
            #     input()
        output_dir = Path(self.config['generator']["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        for gen in generation:
            file_name = f"{self.config['generator']['output_dir']}/{self.config['seed']}_{self.config['mode']}_query.jsonl"
            with open(file_name, 'a') as f:
                json.dump(gen, f)
                f.write('\n')
        print(self.token_usage)

    def get_data(self, config):
        # get all the filtered training data ids
        user_id_list = []
        user_id_path = "data/LaMP-42/{}/42_test_history.jsonl".format(config['task'])
        with open(user_id_path, "r") as f:
            for line in f:
                user_id = json.loads(line)["id"]
                if user_id not in user_id_list:
                    user_id_list.append(user_id)
        print(len(user_id_list))

        # get all the answers to the query
        file_path = "data/LaMP/{}/{}/{}_rankings.json".format(config['task'], config['mode'], config['mode'])
        ranking_dict = {}
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
            for id in user_id_list:
                ranking_dict[id] = data[id] # "***": ["***", "***", "***", "***", "***"]

        # get all the answers to the query
        file_path = "data/LaMP/{}/{}/{}_outputs.json".format(config['task'], config['mode'], config['mode'])
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
        data = data['golds']
        solution_dict = {}
        for sol in data:
            if sol['id'] in user_id_list:
                solution_dict[sol['id']] = sol['output']
        print(len(solution_dict))

        
        file_path = "data/LaMP/{}/{}/{}_history_rankings_extended.json".format(config['task'], config['mode'], config['mode'])
        profile_ranking_dict = {}
        with open(file_path, 'r') as f:
            for line in f:
                profile_ranking_dict = json.loads(line)

        file_path = "data/LaMP/{}/{}/{}_questions.json".format(config['task'], config['mode'], config['mode'])
        history_data = []
        query_data = []
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
        print(user_id_list)
            # for user in data:
        for user in data:
            if user['id'] in user_id_list: # and (user['id'] not in existing_ids):
                if config['retriever']['num_retrieve'] != -1:
                    ranking = ranking_dict[user['id']][:config['retriever']['num_retrieve']]
                else:
                    ranking = ranking_dict[user['id']]
                for profile in user['profile']:
                    if profile['id'] in ranking:
                        if config['task'] == 'LaMP_2':
                            history_data.append({'id': user['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['text'], 'retr_ans': profile['category']})
                        elif config['task'] == 'LaMP_2_movie':
                            history_data.append({'id': user['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['description'], 'retr_ans': profile['tag']})
                        elif config['task'] == 'LaMP_3':
                            history_data.append({'id': user['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['text'], 'retr_ans': profile['score']})
                        elif config['task'] == 'LaMP_4':
                            history_data.append({'id': user['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['text'], 'retr_ans': profile['title']})
                        elif config['task'] == 'LaMP_5':
                            history_data.append({'id': user['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['abstract'], 'retr_ans': profile['title']})
                if config['retriever']['num_retrieve'] < len(user['profile']):
                    profile_data = random.sample(user['profile'], config['retriever']['num_retrieve'])
                else:
                    profile_data = user['profile']
                for profile in profile_data:
                    profile_id = profile['id']
                    profile_ranking = profile_ranking_dict[profile_id]
                    if config['retriever']['num_retrieve'] != -1:
                        profile_ranking = profile_ranking[:config['retriever']['num_retrieve']]
                    # temp = {}
                    for profile2 in user['profile']:
                        if profile2['id'] in profile_ranking:
                            if config['task'] == 'LaMP_2':
                                query_data.append({'id': user['id'], 'source': profile['text'], 'target': profile['category'], 'retrieval': profile2['text'], 'retr_ans': profile2['category']})
                            elif config['task'] == 'LaMP_2_movie':
                                query_data.append({'id': user['id'], 'source': profile['description'], 'target': profile['tag'], 'retrieval': profile2['description'], 'retr_ans': profile2['tag']})
                            elif config['task'] == 'LaMP_3':
                                query_data.append({'id': user['id'], 'source': profile['text'], 'target': profile['score'], 'retrieval': profile2['text'], 'retr_ans': profile2['score']})
                            elif config['task'] == 'LaMP_4':
                                query_data.append({'id': user['id'], 'source': profile['text'], 'target': profile['title'], 'retrieval': profile2['text'], 'retr_ans': profile2['title']})
                            elif config['task'] == 'LaMP_5':
                                query_data.append({'id': user['id'], 'source': profile['abstract'], 'target': profile['title'], 'retrieval': profile2['abstract'], 'retr_ans': profile2['title']})
        print(len(history_data), len(query_data))
        return history_data, query_data