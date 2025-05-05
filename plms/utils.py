import os
import pandas as pd
import json
import re
from nltk.translate.bleu_score import sentence_bleu
import spacy
import evaluate
import numpy as np
#from datasets import load_metric

def save_result(path: str, result: dict[str, str]):
    file_mode = 'a' if os.path.exists(path) else 'w'
    with open(path, file_mode) as file:
        pd.DataFrame([result]).to_csv(file, index=False, header=(file_mode=='w'))
def jaccard_sim(
    docA: set[str],
    docB: list[set[str]]
):
    return [len(docA & e) / len(docA | e) for e in docB]

def read_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)
def post_process(refs_or_preds: list[str]):
    n_errors: int = 0
    total: int = 0

    results: list[str, list[str]] = {'question': [], 'answer': []}
    for e in refs_or_preds:
        pairs: list[str] = [i.strip() for i in re.split('\[SEP\]', e) if i.strip()]
        questions = []
        answers = []
        for qa in pairs:
            total += 1
            if qa.startswith('question: '):
                qa = qa.removeprefix('question: ')
                if ', answer: ' in qa:
                    q, *a = qa.split(', answer: ')
                    questions.append(q.strip())
                    answers.append(a[0].strip())
                else:
                    n_errors += 1
            else:
                n_errors += 1

        results['question'].append(questions)
        results['answer'].append(answers)

    results['qa'] = [[q + ' ' + a for q, a in zip(qas, ans)] for qas, ans in
                     zip(results['question'], results['answer'])]
    print('% error: ', round(n_errors / total * 100, 4))

    return results

class MetricsCalculator:
    def __init__(self):
        self.nlp = spacy.load('vi_core_news_lg')
        self.rouge_metrics = evaluate.load('rouge')
        self.meteor_metrics = evaluate.load('meteor')
        self.bert_score = evaluate.load('bertscore')

    def bleu(self, predict, goal):
        bleu_scores = {1: [], 2: [], 3: [], 4: []}

        for sent1, sent2 in zip(predict, goal):
            sent1_doc = self.nlp(sent1)
            sent2_doc = self.nlp(sent2)
            ws = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
            for n in range(1, 5):
                weights = ws[n-1]
                sent1_tokens = [token.text for token in sent1_doc]
                sent2_tokens = [token.text for token in sent2_doc]
                bleu_score = sentence_bleu([sent1_tokens], sent2_tokens, weights=weights)
                bleu_scores[n].append(bleu_score)
        result = {}
        for n in range(1, 5):
            avg_bleu_score = (sum(bleu_scores[n]) / len(bleu_scores[n]))*100
            result["BLEU{}".format(n)] = (sum(bleu_scores[n]) / len(bleu_scores[n]))*100
        return result

    def rouge(self, predict, goal):
        result = self.rouge_metrics.compute(predictions=predict, references=goal)
        return result

    def meteor(self, predict, goal):
        result = self.meteor_metrics.compute(predictions=predict, references=goal)
        return result

    def bert(self, predict, goal):
        score = self.bert_score.compute(predictions=predict, references=goal, lang='vi')
        result = np.array(score['f1']).mean()
        return result