import json
import os
import re
from glob import glob
from tqdm import tqdm
from typing import List, Dict
import spacy
import fire

SPLITTER = spacy.load('vi_core_news_lg')
HIGHLIGHT_TOKEN = '<hl>'

class DataProcessor:
    def __init__(self):
        self.train_path: str = 'data/train.jsonl'
        self.dev_path: str = 'data/dev.jsonl'
        self.test_path: str = 'data/test.jsonl'
        self.format_type: str = ''

    def get_sentence(self, document: str):
        return [str(s) for s in SPLITTER(document).sents]

    def jsonline_reader(self, filename: str):
        with open(filename, 'r') as f_reader:
            examples = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]
        return examples

    def process_single_data(self, data: Dict, output_dir: str):
        """ Convert single raw json data into QG format """
        example = {'question': data["question"], 'paragraph': data["context"], 'answer': data["answer"]}

        # get sentence
        position = (example['paragraph'].replace('\u202f', ' ')).find((example['answer'].replace('\u202f', ' ')))
        assert position != -1
        before_tmp = self.get_sentence(example['paragraph'][:position])
        if len(before_tmp) == 0:
            before = ''
            before_sentence = ''
        else:
            if before_tmp[-1].endswith('.'):
                before = ' '.join(before_tmp)
                before_sentence = ''
            else:
                before = ' '.join(before_tmp[:-1])
                before_sentence = before_tmp[-1]
                before_sentence = before_sentence if before_sentence.endswith(' ') else '{} '.format(before_sentence)
        after_tmp = self.get_sentence(example['paragraph'][position + len(example['answer']):])
        if len(after_tmp) == 0:
            after = ''
            after_sentence = ''
        else:
            after = ' '.join(after_tmp[1:])
            after_sentence = after_tmp[0]
            after_sentence = after_sentence if after_sentence.startswith(' ') else ' {}'.format(after_sentence)
        example['sentence'] = '{}{}{}'.format(before_sentence, example['answer'], after_sentence)

        # get paragraph_sentence
        before = '' if before == '' else '{} '.format(before)
        after = '' if after == '' else ' {}'.format(after)
        source_text = '{0}{1} {2} {1}{3}'.format(before, HIGHLIGHT_TOKEN, example['sentence'], after)
        example['paragraph_sentence'] = re.sub(r'\s+', ' ', source_text)

        # get paragraph_answer
        source_text = '{0}{1} {2} {1}{3}'.format(
            example['paragraph'][:position], HIGHLIGHT_TOKEN, example['answer'],
            example['paragraph'][position + len(example['answer']):])
        example['paragraph_answer'] = re.sub(r'\s+', ' ', source_text)

        # get sentence_answer
        if len(before_tmp) == 0 or before_tmp[-1].endswith('.'):
            before = ''
        else:
            before = before_tmp[-1] if before_tmp[-1].endswith(' ') else '{} '.format(before_tmp[-1])
        if len(after_tmp) == 0:
            after = ''
        else:
            after = after_tmp[0] if after_tmp[0].startswith(' ') else ' {}'.format(after_tmp[0])
        source_text = '{0}{1} {2} {1}{3}'.format(before, HIGHLIGHT_TOKEN, example['answer'], after)
        example['sentence_answer'] = re.sub(r'\s+', ' ', source_text)

        # Writing to file
        output_file = os.path.join(output_dir, 'processed_data.jsonl')
        with open(output_file, 'a') as f:
            f.write(json.dumps(example) + '\n')

    def process_data(self, input_dir: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = {'train': 'train.jsonl', 'dev': 'dev.jsonl', 'test': 'test.jsonl'}
        for _, v in path.items():
            json_data = []
            for _file in sorted(glob(os.path.join(input_dir, v))):
                json_data += self.jsonline_reader(_file)
            for single_data in tqdm(json_data):
                self.process_single_data(single_data, output_dir)


if __name__ == '__main__':
    fire.Fire(DataProcessor)