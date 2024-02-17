import json
import logging
import os
from collections import defaultdict
from os.path import join as pj
from typing import Dict, List
import torch
from .spacy_module import SpacyPipeline
from .language_model import TransformersQG
from .data import get_dataset

LANG_NEED_TOKENIZATION = ['ja', 'zh']

def evaluate(export_dir: str = '.',
             batch_size: int = 32,
             n_beams: int = 4,
             hypothesis_file_dev: str = None,
             hypothesis_file_test: str = None,
             model: str = None,
             max_length: int = 512,
             max_length_output: int = 64,
             dataset_path: str = 'shnl/qg-example',
             dataset_name: str = 'default',
             input_type: str = 'paragraph_answer',
             output_type: str = 'question',
             prediction_aggregation: str = 'first',
             prediction_level: str = None,
             data_caches: Dict = None,
             bleu_only: bool = False,
             overwrite: bool = False,
             use_auth_token: bool = False,
             torch_dtype=None,
             device_map: str = None,
             low_cpu_mem_usage: bool = False,
             language: str = 'en',
             test_split: str = 'test',
             validation_split: str = 'validation'):
    """ Evaluate question-generation model """
    reference_files = get_reference_files(dataset_path, dataset_name)
    if prediction_level is None:
        valid_prediction_level = [k.split('-')[0] for k in reference_files.keys()]
        if 'sentence' in valid_prediction_level:
            prediction_level = 'sentence'
        elif 'answer' in valid_prediction_level:
            prediction_level = 'answer'
        elif "questions_answers" in valid_prediction_level:
            prediction_level = 'questions_answers'
        else:
            raise ValueError(f"unexpected error: {valid_prediction_level}")
    path_metric = pj(export_dir, f'metric.{prediction_aggregation}.{prediction_level}.{input_type}.{output_type}.{dataset_path.replace("/", "_")}.{dataset_name}.json')
    metric = {}
    if not overwrite and os.path.exists(path_metric):
        with open(path_metric, 'r') as f:
            metric = json.load(f)
            if bleu_only:
                return metric
    os.makedirs(export_dir, exist_ok=True)

    if model is not None:
        lm = TransformersQG(model,
                            max_length=max_length,
                            max_length_output=max_length_output,
                            drop_overflow_error_text=False,
                            skip_overflow_error=True,
                            language=language,
                            use_auth_token=use_auth_token,
                            torch_dtype=torch_dtype,
                            device_map=device_map,
                            low_cpu_mem_usage=low_cpu_mem_usage)
        lm.eval()

        def get_model_prediction_file(split, _input_type, _output_type):
            path = pj(
                export_dir,
                f'samples.{split}.hyp.{_input_type}.{_output_type}.{dataset_path.replace("/", "_")}.{dataset_name}.txt')
            raw_input, _ = get_dataset(
                dataset_path,
                dataset_name,
                split=split,
                input_type=_input_type,
                output_type=_output_type,
                use_auth_token=use_auth_token)
            if os.path.exists(path) and not overwrite:
                with open(path) as f_read:
                    tmp = f_read.read().split('\n')
                if len(tmp) == len(raw_input):
                    return path
                logging.warning(f'recompute {path}: {len(tmp)} != {len(raw_input)}')
            prefix_type = None
            if lm.add_prefix:
                if _output_type == 'questions_answers':
                    prefix_type = 'qag'
                elif _output_type == 'question':
                    prefix_type = 'qg'
                elif _output_type == 'answer':
                    prefix_type = 'ae'
                else:
                    raise ValueError(f"prefix type is not determined for the output_type {_output_type}")
            output = lm.generate_prediction(
                raw_input,
                batch_size=batch_size,
                num_beams=n_beams,
                prefix_type=prefix_type,
                cache_path=None if data_caches is None else data_caches[split])
            with open(path, 'w') as f_writer:
                f_writer.write('\n'.join(output))
            return path

        try:
            hypothesis_file_test = get_model_prediction_file(test_split, input_type, output_type)
        except ValueError:
            hypothesis_file_test = None
        try:
            hypothesis_file_dev = get_model_prediction_file(validation_split, input_type, output_type)
        except ValueError:
            hypothesis_file_dev = None

    assert hypothesis_file_dev is not None or hypothesis_file_test is not None,\
        f'model ({model}) or file path ({hypothesis_file_dev}, {hypothesis_file_test}) is needed'

    with open(path_metric, 'w') as f:
        json.dump(metric, f)
    return metric