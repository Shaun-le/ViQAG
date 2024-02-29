import fire
from typing import List
import os
from plms.trainer import Trainer

class FineTuning:
    def __init__(self):
        pass

    def fine_tuning(
            self,
            checkpoint_dir: str = './cp',
            dataset_path: str = 'shnl/qg-example',
            dataset_name: str = 'default',
            input_types: List or str = ['paragraph_answer', 'paragraph_sentence'],
            output_types: List or str = ['question', 'answer'],
            prefix_types: List or str = ['qg','ae'],
            model: str = '',
            max_length: int = 512,
            max_length_output: int = 256,
            epoch: int = 10,
            batch: int = 4,
            lr: float = 1e-4,
            fp16: bool = False,
            random_seed: int = 42,
            gradient_accumulation_steps: int = 4,
            label_smoothing: float = None,
            disable_log: bool = False,
            config_file: str = 'trainer_config.json',
            use_auth_token: bool = False,
            torch_dtype=None,
            device_map: str = None,
            low_cpu_mem_usage: bool = False
    ):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(
                f"Training model with params:\n"
                f"checkpoint_dir = {checkpoint_dir}\n"
                f"dataset_path = {dataset_path}\n"
                f"dataset_name = {dataset_name}\n"
                f"input_types = {input_types}\n"
                f"output_types = {output_types}\n"
                f"prefix_types = {prefix_types}\n"
                f"model = {model}\n"
                f"max_length = {max_length}\n"
                f"max_length_output = {max_length_output}\n"
                f"epoch = {epoch}\n"
                f"batch = {batch}\n"
                f"lr = {lr}\n"
                f"fp16 = {fp16}\n"
                f"random_seed = {random_seed}\n"
                f"gradient_accumulation_steps = {gradient_accumulation_steps}\n"
                f"label_smoothing = {label_smoothing}\n"
                f"disable_log = {disable_log}\n"
                f"config_file = {config_file}\n"
                f"use_auth_token = {use_auth_token}\n"
                f"torch_dtype = {torch_dtype}\n"
                f"device_map = {device_map}\n"
                f"low_cpu_mem_usage = {low_cpu_mem_usage}\n"
            )
        assert (
            model
        ), "Please specify a --model, e.g. --model='VietAI/vit5-base'"
        trainer = Trainer(
            checkpoint_dir = checkpoint_dir,
            dataset_path = dataset_path,
            dataset_name = dataset_name,
            input_types = input_types,
            output_types = output_types,
            prefix_types = prefix_types,
            model = model,
            max_length = max_length,
            max_length_output = max_length_output,
            epoch = epoch,
            batch = batch,
            lr = lr,
            fp16 = fp16,
            random_seed = random_seed,
            gradient_accumulation_steps = gradient_accumulation_steps,
            label_smoothing = label_smoothing,
            disable_log = disable_log,
            config_file = config_file,
            use_auth_token = use_auth_token,
            torch_dtype = torch_dtype,
            device_map = device_map,
            low_cpu_mem_usage = low_cpu_mem_usage
        )
        trainer.train()

    def inst_tuning(self):
        return 'coming soon'

    def alpaca(self):
        return 'coming soon'

if __name__ == "__main__":
    fire.Fire(FineTuning)