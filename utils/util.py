import yaml
import json
import random
from datasets import Dataset

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_data(config):
    if config['mode'] == 'train':
        data_path = "data/LaMP/{}/{}/{}_new.json".format(config['task'], config['mode'], config['mode'])
    elif config['mode'] == 'dev':
        data_path = "data/LaMP/{}/{}/{}_new_history.json".format(config['task'], config['mode'], config['mode'])
    elif config['mode'] == 'test':
        data_path = "data/LaMP/{}/dev/dev_new_test.json".format(config['task'], config['mode'], config['mode'])
    source_list = []
    target_list = []
    id_list = []
    user_list = {}
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['id'] not in user_list:
                user_list[data['id']] = [data]
            else:
                user_list[data['id']].append(data)
    if config['partial_data']['users'] != -1:
        random_users = dict(random.sample(user_list.items(), config['partial_data']['users']))
    else:
        random_users = user_list
    for user in random_users.keys():
        for data in random_users[user]:
            source = data['source']
            target = data['target']
            id = data['id']
            source_list.append(source)
            target_list.append(target)
            id_list.append(id)
    dataset = Dataset.from_dict({'id': id_list, 'source': source_list, 'target': target_list}).with_format("torch").shuffle(seed=config['seed'])
    print(len(dataset))
    return dataset

def get_personalize_data(config):
    dict_path = "data/LaMP/{}/{}/{}_user_dict.json".format(config['task'], config['mode'], config['mode'])
    with open(dict_path, 'r') as f:
        for line in f:
            user_dict = json.loads(line)
    
    data_path = "data/LaMP/{}/{}/{}_new.json".format(config['task'], config['mode'], config['mode'])
    data_list = []
    source_list = []
    target_list = []
    id_list = []
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
            source = data['source']
            target = data['target']
            source_list.append(source)
            target_list.append(target)
            id_list.append(user_dict[data['id']])
    dataset = Dataset.from_dict({'source': source_list, 'target': target_list, 'id': id_list}).with_format("torch").shuffle(seed=config['seed'])
    return dataset