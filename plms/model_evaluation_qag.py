import json
import logging
import os
from itertools import chain
from datasets import load_dataset
from .language_model import TransformersQG
from .utils import save_result

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
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
        logging.info('QAG evaluator.')
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
                    _file = f"{self.export_dir}/samples.{_split}.questions_answers." \
                            f"{self.dataset_path.replace('/', '_')}.{self.dataset_name}." \
                            f"{self.model_ae.replace('/', '_')}"
                else:
                    _file = f"{self.export_dir}/samples.{_split}.questions_answers." \
                            f"{self.dataset_path.replace('/', '_')}.{self.dataset_name}"

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
                gold_reference.append(' [SEP] '.join([
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
                prediction = [' [SEP] '.join([f"question: {q}, answer: {a}" for q, a in p])
                              if p is not None else "" for p in prediction]
                assert len(prediction) == len(model_input), f"{len(prediction)} != {len(model_input)}"
                for i in range(len(prediction)):
                    save_result(path=f'{_file}.csv',
                                result={'prediction': prediction[i], 'reference': gold_reference[i]})