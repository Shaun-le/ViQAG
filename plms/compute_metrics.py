import nltk
from .utils import post_process, jaccard_sim, MetricsCalculator
nltk.download("wordnet")
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
class Evaluate:
    def __init__(self,
                 result_file: str = ''):
        self.result_file = result_file
    def compute_metrics(self):
        if self.result_file.endswith('.csv'):
            df = pd.read_csv(self.result_file)
        elif self.result_file.endswith('.json'):
            df = pd.read_json(self.result_file)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .json file.")
        predictions, references = [], []
        for i in range(len(df)):
            predictions.append(str(df['prediction'][i]))
            references.append(str(df['reference'][i]))
        refs = post_process(references)
        preds = post_process(predictions)

        preds_aligned = []
        for ref, pred in zip(refs['qa'], preds['qa']):
            if len(pred) < len(ref):
                preds_aligned.append(pred)
                continue

            tmp = []
            pred_tmp = pred.copy()
            e = [set(sent.lower().split(' ')) for sent in ref]
            e1 = [set(sent.lower().split(' ')) for sent in pred]
            for s in e:
                scores = jaccard_sim(s, e1)
                chosen_idx = np.argmax(scores)
                s1 = pred_tmp.pop(chosen_idx)
                e1.pop(chosen_idx)

                tmp.append(s1)
            preds_aligned.append(tmp)

        final_refs = [' '.join(e) for e in refs['qa']]
        final_preds = [' '.join(e) for e in preds_aligned]

        metrics_calculator = MetricsCalculator()

        bleu = metrics_calculator.bleu(final_preds, final_refs)
        rouge = metrics_calculator.rouge(final_preds, final_refs)
        meteor = metrics_calculator.meteor(final_preds, final_refs)
        bert_score = metrics_calculator.bert(final_preds, final_refs)

        result = {'BLEU SCORE': bleu,
                  'ROUGE SCORE': rouge,
                  'METEOR SCORE': meteor,
                  'BERT SCORE': bert_score}

        for key, value in result.items():
            print(key, ':', value)