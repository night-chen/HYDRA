import os
import time
from pathlib import Path
import json
from tqdm import tqdm
from utils.credentials import api_key_list
from openai import AzureOpenAI
import random
import random
import concurrent.futures

def append_to_jsonl(data, generation, filename: str) -> None:
    """Append a json paylaod to the end of a jsonl file"""
    for res in generation:
        json_dict = {"id": data['id'], "source": data['source'], "target": data['target'], "generation": res, "retrieval": data['retrieval'], "retr_ans": data['retr_ans']}
        json_string = json.dumps(json_dict)
        with open(filename, 'a') as f:
            f.write(json_string + '\n')

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
    
    def generate(self):
        generation = []
        self.data = self.history_data
        print('='*50)
        print('Start generating {} data...'.format(len(self.data)))
        p_bar = tqdm(range(len(self.data)))
        output_file = f"{self.config['generator']['output_dir']}/{self.config['seed']}_{self.config['mode']}_history_0731.jsonl"
        def openai_history(l_data):
            messages = [
                {"role": "user", "content": self.question_template.format(text=l_data['retrieval'])},
                {"role": "assistant", "content": l_data['retr_ans']},
                {"role": "user", "content": self.question_template.format(text=l_data['source'])}
            ]
            flag = False
            num_trials = 0
            max_trials = 3
            while num_trials < max_trials:
                try:
                    raw_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.config['generator']['max_length'],
                        temperature=self.config['generator']['temperature'],
                        frequency_penalty=self.config['generator']['frequency_penalty'],
                        presence_penalty=self.config['generator']['presence_penalty'],
                        stop=self.config['generator']['stop'],
                        n=self.config['generator']['num_return_sequences'],
                    )
                    self.token_usage["input"] += raw_response.usage.prompt_tokens
                    self.token_usage["output"] += raw_response.usage.completion_tokens
                    contents = [choice.message.content.strip() for choice in raw_response.choices]
                    append_to_jsonl(l_data, contents, output_file)
                    flag = True
                    if len(contents) == 0:
                        flag = False
                        raise RuntimeError("No response from the API")
                    p_bar.update(1)
                    break
                except Exception as e:
                    self.switch_api_key()
                    flag = False
                    num_trials += 1
                    print(e)
                    if num_trials > 3:
                        print(f"Retry exceed the max_retries {num_trials} times.")
                        break
                    time.sleep(10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(openai_history, self.data)
        print(self.token_usage)

        print('='*50)
        print('Start generating {} data...'.format(len(self.data)))
        p_bar = tqdm(range(len(self.data)))
        output_file = f"{self.config['generator']['output_dir']}/{self.config['seed']}_{self.config['mode']}_query.jsonl"

        def openai_query(l_data):
            messages = [
                {"role": "user", "content": self.question_template.format(text=l_data['retrieval'])},
                {"role": "assistant", "content": l_data['retr_ans']},
                {"role": "user", "content": self.question_template.format(text=l_data['source'])}
            ]
            flag = False
            num_trials = 0
            max_trials = 3
            while num_trials < max_trials:
                try:
                    raw_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=self.config['generator']['max_length'],
                        temperature=self.config['generator']['temperature'],
                        frequency_penalty=self.config['generator']['frequency_penalty'],
                        presence_penalty=self.config['generator']['presence_penalty'],
                        stop=self.config['generator']['stop'],
                        n=self.config['generator']['num_return_sequences'],
                    )
                    self.token_usage["input"] += raw_response.usage.prompt_tokens
                    self.token_usage["output"] += raw_response.usage.completion_tokens
                    contents = [choice.message.content.strip() for choice in raw_response.choices]
                    append_to_jsonl(l_data, contents, output_file)
                    flag = True
                    if len(contents) == 0:
                        flag = False
                        raise RuntimeError("No response from the API")
                    p_bar.update(1)
                    break
                except Exception as e:
                    self.switch_api_key()
                    flag = False
                    num_trials += 1
                    print(e)
                    if num_trials > 3:
                        print(f"Retry exceed the max_retries {num_trials} times.")
                        break
                    time.sleep(10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(openai_query, self.data)
        print(self.token_usage)

    def get_data(self, config):
        print('='*50)
        print('Start getting raw data...')
        start_time = time.time()
        # get all the filtered training data ids
        user_id_list = []
        file_path = "data/LaMP/{}/{}/{}_questions.json".format(config['task'], config['mode'], config['mode'])
        with open(file_path, 'r') as f:
            for line in f:
                data_list = json.loads(line)
            for line in data_list:
                if line['id'] not in user_id_list:
                    user_id_list.append(line['id'])
        user_id_list = user_id_list[:config['partial_data']['users']]
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

        
        file_path = "data-new/LaMP/{}/{}/{}_history_rankings_extended.json".format(config['task'], config['mode'], config['mode'])
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
        # print(user_id_list)
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
                            history_data.append({'id': user['id'], 'pid': profile['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['text'], 'retr_ans': profile['category']})
                        elif config['task'] == 'LaMP_2_movie':
                            history_data.append({'id': user['id'], 'pid': profile['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['description'], 'retr_ans': profile['tag']})
                        elif config['task'] == 'LaMP_3':
                            history_data.append({'id': user['id'], 'pid': profile['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['text'], 'retr_ans': profile['score']})
                        elif config['task'] == 'LaMP_4':
                            history_data.append({'id': user['id'], 'pid': profile['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['text'], 'retr_ans': profile['title']})
                        elif config['task'] == 'LaMP_5':
                            history_data.append({'id': user['id'], 'pid': profile['id'], 'source': user['input'], 'target': solution_dict[user['id']], 'retrieval': profile['abstract'], 'retr_ans': profile['title']})
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
                                query_data.append({'id': user['id'], 'pid': profile['id'], 'source': profile['text'], 'target': profile['category'], 'retrieval': profile2['text'], 'retr_ans': profile2['category']})
                            elif config['task'] == 'LaMP_2_movie':
                                query_data.append({'id': user['id'], 'pid': profile['id'], 'source': profile['description'], 'target': profile['tag'], 'retrieval': profile2['description'], 'retr_ans': profile2['tag']})
                            elif config['task'] == 'LaMP_3':
                                query_data.append({'id': user['id'], 'pid': profile['id'], 'source': profile['text'], 'target': profile['score'], 'retrieval': profile2['text'], 'retr_ans': profile2['score']})
                            elif config['task'] == 'LaMP_4':
                                query_data.append({'id': user['id'], 'pid': profile['id'], 'source': profile['text'], 'target': profile['title'], 'retrieval': profile2['text'], 'retr_ans': profile2['title']})
                            elif config['task'] == 'LaMP_5':
                                query_data.append({'id': user['id'], 'pid': profile['id'], 'source': profile['abstract'], 'target': profile['title'], 'retrieval': profile2['abstract'], 'retr_ans': profile2['title']})
        print(len(history_data), len(query_data))
        print('Finish getting raw data... Time elapsed: {}s'.format(time.time() - start_time))
        return history_data, query_data