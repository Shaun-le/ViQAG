import nltk
nltk.download("wordnet")
import pandas as pd
import numpy as np
from utils import jaccard_sim, post_process
from utils import bleu, rouge, meteor, bert_score
class Evaluate:
    def __init__(self,
                 result_file: str = ''):
        self.result_file = result_file
    def compute_metrics(self):
        df = pd.read_csv(self.result_file)
        predictions, references = [], []
        for i in range(len(df)):
            predictions.append(df['prediction'][i])
            references.append(df['reference'][i])
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

        bleu = bleu(final_preds, final_refs)
        rouge = rouge(final_preds, final_refs)
        meteor = meteor(final_preds, final_refs)
        bert = bert_score(final_preds, final_refs)

        result = {'BLEU SCORE: ', bleu,
                  'ROUGE SCORE: ', rouge,
                  'METEOR SCORE: ', meteor,
                  'BERT SCORE: ', bert}
        print(result)
        return result