import json
import logging
import os
import re
import evaluate
from datasets import load_metric
import numpy as np
from itertools import chain
from datasets import load_dataset
from ..language_model import TransformersQG
import nltk
nltk.download("wordnet")
from nltk.translate.bleu_score import sentence_bleu
import spacy

nlp = spacy.load('vi_core_news_lg')

def bleu(predict, goal):
    bleu_scores = {1: [], 2: [], 3: [], 4: []}

    for sent1, sent2 in zip(predict, goal):
        sent1_doc = nlp(sent1)
        sent2_doc = nlp(sent2)
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
        result["BLEU{}".format(n)] = avg_bleu_score
    return result

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

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
        pairs: list[str] = [i.strip() for i in re.split('[^\S\n\t]?\[SEP\][^\S\n\t]?', e) if i.strip()]
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

class Evaluation:
    def __init__(self,
                 model: str = 'VietAI/vit5-base',
                 model_ae: str = None,
                 max_length: int = 512,
                 max_length_output: int = 256,
                 dataset_path: str = 'shnl/qg-example',
                 dataset_name: str = '',
                 test_split: str = 'test',
                 validation_split: str = 'validation',
                 n_beams: int = 8,
                 batch_size: int = 4,
                 language: str = 'vi',
                 use_auth_token: bool = True,
                 device_map: str = None,
                 low_cpu_mem_usage: bool = False,
                 export_dir: str = './result',
                 hyp_test: str = None,
                 hyp_dev: str = None,
                 overwrite_prediction: bool = True,
                 overwrite_metric: bool = True,
                 is_qg: bool = None,
                 is_ae: bool = None,
                 is_qag: bool = True,
                 use_reference_answer: bool = False):
        logging.info('QAG evaluation.')
        self.model = model
        self.model_ae = model_ae
        self.max_length = max_length
        self.max_length_output = max_length_output
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.test_split = test_split
        self.validation_split = validation_split
        self.n_beams = n_beams
        self.batch_size = batch_size
        self.language = language
        self.use_auth_token = use_auth_token
        self.device_map = device_map
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.export_dir = export_dir
        self.hyp_test = hyp_test
        self.hyp_dev = hyp_dev
        self.overwrite_prediction = overwrite_prediction
        self.overwrite_metric = overwrite_metric
        self.is_qg = is_qg
        self.is_ae = is_ae
        self.is_qag = is_qag
        self.use_reference_answer = use_reference_answer

    def load_model(self):
        os.makedirs(self.export_dir, exist_ok=True)
        if self.model is not None:
            _model = TransformersQG(self.model,
                                    is_ae=None if self.is_ae else True,
                                    is_qg=None if self.is_qg else True,
                                    is_qag=None if self.is_qag else True,
                                    model_ae=self.model_ae,
                                    skip_overflow_error=True,
                                    drop_answer_error_text=True,
                                    language=self.language,
                                    max_length=self.max_length,
                                    max_length_output=self.max_length_output)
            _model.eval()
            return _model
        raise ValueError("require `-m` or `--model`")

    def evaluation(self):
        if self.model_ae is not None:
            metric_file = f"{self.export_dir}/metric.first.answer.paragraph.questions_answers." \
                          f"{self.dataset_path.replace('/', '_')}.{self.dataset_name}." \
                          f"{self.model_ae.replace('/', '_')}.json"
        else:
            metric_file = f"{self.export_dir}/metric.first.answer.paragraph.questions_answers." \
                          f"{self.dataset_path.replace('/', '_')}.{self.dataset_name}.json"
        if os.path.exists(metric_file):
            with open(metric_file) as f:
                output = json.load(f)
        else:
            output = {}
        for _split, _file in zip([self.test_split, self.validation_split], [self.hyp_test, self.hyp_dev]):
            if _file is None:
                if self.model_ae is not None:
                    _file = f"{self.export_dir}/samples.{_split}.hyp.paragraph.questions_answers." \
                            f"{self.dataset_path.replace('/', '_')}.{self.dataset_name}." \
                            f"{self.model_ae.replace('/', '_')}.txt"
                else:
                    _file = f"{self.export_dir}/samples.{_split}.hyp.paragraph.questions_answers." \
                            f"{self.dataset_path.replace('/', '_')}.{self.dataset_name}.txt"

            logging.info(f'generate qa for split {_split}')
            if _split not in output:
                output[_split] = {}

            dataset = load_dataset(self.dataset_path, None if self.dataset_name == 'default' else self.dataset_name,
                                   split=_split, use_auth_token=True)
            df = dataset.to_pandas()

            # formatting data into qag format
            model_input = []
            gold_reference = []
            model_highlight = []
            for paragraph, g in df.groupby("paragraph"):
                model_input.append(paragraph)
                model_highlight.append(g['answer'].tolist())
                gold_reference.append(' | '.join([
                    f"question: {i['question']}, answer: {i['answer']}" for _, i in g.iterrows()
                ]))
            prediction = None
            if not self.overwrite_prediction and os.path.exists(_file):
                with open(_file) as f:
                    _prediction = f.read().split('\n')
                if len(_prediction) != len(gold_reference):
                    logging.warning(f"found prediction file at {_file} but length not match "
                                    f"({len(_prediction)} != {len(gold_reference)})")
                else:
                    prediction = _prediction
            if prediction is None:
                model = self.load_model()
                # model prediction
                if not self.use_reference_answer:
                    logging.info("model prediction: (qag model)")
                    prediction = model.generate_qa(
                        list_context=model_input,
                        num_beams=self.n_beams,
                        batch_size=self.batch_size)
                else:
                    logging.info("model prediction: (qg model, answer fixed by reference)")
                    model_input_flat = list(chain(*[[i] * len(h) for i, h in zip(model_input, model_highlight)]))
                    model_highlight_flat = list(chain(*model_highlight))
                    prediction_flat = model.generate_q(
                        list_context=model_input_flat,
                        list_answer=model_highlight_flat,
                        num_beams=self.n_beams,
                        batch_size=self.batch_size)
                    _index = 0
                    prediction = []
                    for h in model_highlight:
                        questions = prediction_flat[_index:_index+len(h)]
                        answers = model_highlight_flat[_index:_index+len(h)]
                        prediction.append(list(zip(questions, answers)))
                        _index += len(h)

                # formatting prediction
                prediction = [' | '.join([f"question: {q}, answer: {a}" for q, a in p]) if p is not None else "" for p in prediction]
                assert len(prediction) == len(model_input), f"{len(prediction)} != {len(model_input)}"

                meteor_metrics = evaluate.load('meteor')
                bert_score = evaluate.load('bertscore')

                refs = post_process(gold_reference)
                preds = post_process(prediction)

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

                bleu_result = bleu(final_preds, final_refs)

                metrics = load_metric('rouge')
                rouge_result = {k: (v.mid.fmeasure) * 100 for k, v in metrics.compute(predictions=final_preds, references=final_refs).items()}

                bert = bert_score.compute(predictions=final_preds, references=final_refs, lang=self.language)
                bert_result = np.array(bert['f1']).mean()

                meteor_result = meteor_metrics.compute(predictions=final_preds, references=final_refs)

                evaluation_result = {
                    "BLEU": bleu_result,
                    "ROUGE": rouge_result,
                    "BERT": bert_result,
                    "METEOR": meteor_result
                }

                with open(_file, 'w') as f:
                    f.write('\n'.join(prediction))
                with open(f'{_file}_ref.txt', 'w') as f:
                    f.write('\n'.join(gold_reference))
                with open(f'{_file}.json', 'w') as f:
                    json.dump(evaluation_result, f)