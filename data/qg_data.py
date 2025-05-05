import json
import os
import re
from glob import glob
from tqdm.notebook import tqdm
from typing import Dict
import spacy
import fire

SPLITTER = spacy.load('vi_core_news_lg')
HIGHLIGHT_TOKEN = '<hl>'

class QGDataProcessor:
    def __init__(self):
        self.input_dir: str = 'data/examples'
        self.output_dir: str = 'data/processed_data'

    def get_sentence(self, document: str):
        return [str(s) for s in SPLITTER(document).sents]

    def jsonline_reader(self, filename: str):
        with open(filename, 'r') as f_reader:
            examples = [json.loads(i) for i in f_reader.read().split('\n') if len(i) > 0]
        return examples

    def process_single_data(self, data: Dict):
        """ Convert single raw json data into QG format """
        example = {'question': data["question"], 'paragraph': data["context"], 'answer': data["answer"]}

        # get sentence
        position = example['paragraph'].find(example['answer'])
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

        return example

    def process_data(self, input_dir='data/examples', output_dir='data'):
        self.input_dir = input_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        path = {'train': f'{self.input_dir}/train.jsonl', 'dev': f'{self.input_dir}/validation.jsonl', 'test': f'{self.input_dir}/test.jsonl'}
        for k, v in path.items():
            json_data = []
            for _file in sorted(glob(v)):
                json_data += self.jsonline_reader(_file)
            with open('{}/{}.jsonl'.format(self.output_dir, k), 'w') as f:
                for single_data in tqdm(json_data):
                    single_data = self.process_single_data(single_data)
                    f.write(json.dumps(single_data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    fire.Fire(QGDataProcessor)