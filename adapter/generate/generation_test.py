import os
import json
import argparse
from openai import OpenAI, AzureOpenAI
from credentials import api_key_list
from datasets import Dataset
from tqdm import tqdm
class generate_openai():
    def __init__(self, data=None, output_path=None, file_name=None):
        self.api_key_list = api_key_list(args.openai_credentials)
        self.api_idx = 0
        self.client = AzureOpenAI(
            api_key=self.api_key_list[self.api_idx]["api_key"], 
            api_version=self.api_key_list[self.api_idx]["api_version"],
            azure_endpoint=self.api_key_list[self.api_idx]["azure_endpoint"]
        )
        self.model = self.api_key_list[self.api_idx]["model"]
        self.token_usage = {"input": 0, "output": 0}
        self.data = data
        self.output_path = output_path
        self.file_name = file_name

    def switch_api_key(self):
        self.api_idx = (self.api_idx + 1) % len(self.api_key_list)
        self.client = AzureOpenAI(
            api_key=self.api_key_list[self.api_idx]["api_key"],
            api_version=self.api_key_list[self.api_idx]["api_version"],
            azure_endpoint=self.api_key_list[self.api_idx]["azure_endpoint"]
        )
        self.model = self.api_key_list[self.api_idx]["model"]
    
    def query(self, prompt, temp=None, n=None, stop=None, max_tokens=None,):
        flag = False
        num_trials = 0
        while not flag:
            try:
                raw_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    max_tokens=64 if max_tokens is None else max_tokens,
                    temperature=1.0 if temp is None else temp,
                    frequency_penalty=0,
                    presence_penalty=0,
                    n=8 if n is None else n,
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
        # template = "Q: {question}\nA:"
        for idx in tqdm(range(len(self.data))):
            prompt_msg = self.data[idx]['prompt_msg']
            responses = self.query(prompt_msg)
            if responses != None:
                for res in responses:
                    generation.append({"id": self.data[idx]['id'], "source": self.data[idx]['prompt_msg'][-1]['content'], "target": self.data[idx]['target'], "generation": res})
        file_name = 'generation-' + self.file_name
        output_dir = os.path.join(self.output_path, file_name)
        for gen in generation:
            with open(output_dir, 'a') as f:
                json.dump(gen, f)
                f.write('\n')
        print(self.token_usage)

parser = argparse.ArgumentParser()
parser.add_argument("--task", default='LaMP_2', type=str, help='task')
parser.add_argument("--mode", default='train', type=str, help='mode')
parser.add_argument("--k", default=1, type=int, help='The number of few-shot examples')
parser.add_argument("--seed", default=42, type=int, help='seed')
parser.add_argument("--openai_credentials", default='chao-3.5', type=str, help='openai')
parser.add_argument("--retriever", default='bm25', type=str, help='retriever')
parser.add_argument("--prefix", default='history', type=str, help='prefix')
args = parser.parse_args()

test_questions = "data/LaMP-42/{}/42_test_history.jsonl".format(args.task, args.prefix)
test_ids = []
data_list = []
with open(test_questions, 'r') as f:
    for line in f:
        data = json.loads(line)
        test_ids.append(data['id'])
        data_list.append(data)
print(len(data_list))

ranking_path = "data/LaMP/{}/{}/{}_{}_rankings_extended.json".format(args.task, args.mode, args.mode, args.prefix)
ranking_dict = {}
with open(ranking_path, 'r') as f:
    for line in f:
        data = json.loads(line)
    # for id in test_ids:
        # ranking_dict[id] = data[id] # "***": ["***", "***", "***", "***", "***"]
    ranking_dict = data
print(len(ranking_dict))

if args.task == 'LaMP_2':
    template = "Which category does this article relate to among the following categories? Just answer with the category name without further explanation. categories: [women, religion, politics, style & beauty, entertainment, culture & arts, sports, science & technology, travel, business, crime, education, healthy living, parents, food & drink] article: {text}"
    keyword1 = "text"
    keyword2 = "category"
elif args.task == 'LaMP_2_movie':
    template = "Which tag does this movie relate to among the following tags? Just answer with the tag name without further explanation. tags: [sci-fi, based on a book, comedy, action, twist ending, dystopia, dark comedy, classic, psychology, fantasy, romance, thought-provoking, social commentary, violence, true story] description: {text}"
    keyword1 = "description"
    keyword2 = "tag"
elif args.task == 'LaMP_3':
    template = "What is the score of the following review on a scale of 1 to 5? just answer with 1, 2, 3, 4, or 5 without further explanation. review: {text}"
    keyword1 = "text"
    keyword2 = "score"
elif args.task == 'LaMP_4':
    template = "Generate a headline for the following article: {text}"
    keyword1 = "text"
    keyword2 = "title"
elif args.task == 'LaMP_5':
    template = "Generate a title for the following abstract of a paper: {text}"
    keyword1 = "abstract"
    keyword2 = "title"
elif args.task == 'LaMP_7':
    template = "Generate a title for the following abstract of a paper: {text}"

file_path = "data/LaMP/{}/{}/{}_{}_questions_extended.json".format(args.task, args.mode, args.mode, args.prefix)
question_dict = {}
answer_dict = {}
with open(file_path, 'r') as f:
    for line in f:
        data_list = json.loads(line)
    for data in data_list:
        question_dict[data['id']] = template.format(text=data['input'])
        for profile in data['profile']:
            question_dict[profile['id']] = profile[keyword1]
            answer_dict[profile['id']] = profile[keyword2]
print(len(question_dict))
# print(question_dict.keys())

file_path = "data/LaMP/{}/{}/{}_{}_outputs_extended.json".format(args.task, args.mode, args.mode, args.prefix)
# answer_dict = {}
with open(file_path, 'r') as f:
    for line in f:
        data = json.loads(line)
    data_list = data['golds']
    for data in data_list:
        answer_dict[data['id']] = data['output']
        # for profile in data['profile']:
        #     answer_dict[profile['id']] = profile[keyword2]
print(len(answer_dict))
# print(answer_dict.keys())

prompts = []
for id in ranking_dict.keys():
    query = question_dict[id]
    target = answer_dict[id]
    history = []
    if args.k < len(ranking_dict[id]):
        temp = args.k
    else:
        temp = len(ranking_dict[id])
    for idx in range(temp):
        ranking = ranking_dict[id][idx]
        history.append({"role": "user", "content": question_dict[ranking]})
        history.append({"role": "assistant", "content": answer_dict[ranking]})
    history.append({"role": "user", "content": query})
    prompts.append({"id": id, "prompt_msg": history, "target": target})

output_path = "data/LaMP/LaMP-RAG-k_{}/{}".format(args.k, args.task)
file_name = "{}_{}.json".format(args.mode, args.prefix)
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_dir = os.path.join(output_path, file_name)
with open(output_dir, 'w') as f:
    for prompt in prompts:
        json.dump(prompt, f)
        f.write('\n')

generator = generate_openai(prompts, output_path, file_name)
generator.generate()