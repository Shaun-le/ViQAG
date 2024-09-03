import fire
import json
import os
from tqdm.notebook import tqdm
from datasets import Dataset
import random

SEP_TOKEN = " [SEP] "

class QAGDataProcessor:
    def __init__(self):
        self.input_dir: str = 'data/examples'
        self.output_dir: str = 'data/processed_data'
        self.instruction_path: str = 'data/instructions.txt'

    def read_jsonl_file(self, file_path):
        contexts = []
        questions = []
        answers = []

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                contexts.append(data["context"])
                questions.append(data["question"])
                answers.append(data["answer"])

        return {"question": questions, "context": contexts, "answer": answers}

    def form(self, directory):
        train_file = os.path.join(directory, "train.jsonl")
        validation_file = os.path.join(directory, "validation.jsonl")
        test_file = os.path.join(directory, "test.jsonl")

        train_data = self.read_jsonl_file(train_file)
        validation_data = self.read_jsonl_file(validation_file)
        test_data = self.read_jsonl_file(test_file)

        return {
            "train": Dataset.from_dict(train_data),
            "validation": Dataset.from_dict(validation_data),
            "test": Dataset.from_dict(test_data)
        }

    def create_data(self, hf_data, instruction_path):
        df = hf_data.to_pandas()
        output = []
        with open(self.instruction_path, 'r') as f:
            instructions = f.read().split('\n')
        for paragraph, g in df.groupby("context"):
            example = {
                'instruction': random.choice(instructions),
                'paragraph': paragraph.replace(SEP_TOKEN, " "),
                'questions': [_g.replace(SEP_TOKEN, " ") for _g in g['question']],
                'answers': [_g.replace(SEP_TOKEN, " ") for _g in g['answer']],
            }
            example["questions_answers"] = SEP_TOKEN.join([f"question: {q}, answer: {a}" for q, a in zip(example["questions"], example["answers"])])
            output.append(example)
        return output

    def process_data(self, input_dir='data/examples', output_dir='data', instruction_path='data/instructions.txt'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.instruction_path = instruction_path

        data = self.form(self.input_dir)
        data_valid = self.create_data(data['validation'], self.instruction_path)
        data_train = self.create_data(data['train'], self.instruction_path)
        data_test = self.create_data(data['test'], self.instruction_path)
        data_all = {'train': data_train, 'validation': data_valid, 'test': data_test}

        os.makedirs(self.output_dir, exist_ok=True)
        for k, _data in data_all.items():
            with open('{}/{}.jsonl'.format(self.output_dir, k), 'w') as f:
                for single_data in tqdm(_data):
                    f.write(json.dumps(single_data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    fire.Fire(QAGDataProcessor)