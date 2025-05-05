import torch
import evaluate
import pandas as pd
from loguru import logger
from datasets import Dataset
from dataclasses import dataclass
from huggingface_hub import login
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesselfig
from peft import Loraselfig, prepare_model_for_kbit_training, get_peft_model
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics = evaluate.load('rouge')

class Trainer:
    def __init__(self,
                 model: str = 'llama-2-13b',
                 checkpoint: str = './checkpoint-13B',
                 max_len: int = 2048,
                 seed: int = 42,
                 num_proc: int = 16,
                 train_path: str = './data/train.json',
                 dev_path: str = './data/dev.json',
                 test_path: str = './data/test.json',
                 per_device_train_batch_size: int = 4,
                 per_device_eval_batch_size: int = 8,
                 num_epochs: int = 10,
                 lr: float = 1e-4,
                 warmup_ratio: float = 0.05,
                 weight_decay: float = 0.015,
                 use_flash_attention: bool = False,
                 ):
        logging.info('initialize model trainer')
        self.model = model
        self.checkpoint = checkpoint,
        self.max_len = max_len,
        self.seed = seed,
        self.num_proc = num_proc,
        self.train_path = train_path,
        self.dev_path = dev_path,
        self.test_path = test_path,
        self.per_device_train_batch_size = per_device_train_batch_size,
        self.per_device_eval_batch_size = per_device_eval_batch_size,
        self.num_epochs = num_epochs,
        self.lr = lr,
        self.warmup_ratio = warmup_ratio,
        self.weight_decay = weight_decay,
        self.use_flash_attention = use_flash_attention,


    def formatting_func(example):
        """Generate instruction data with the following format:
        ### Instruction: 
        {instruction}
    
        ### Input: 
        {context}
    
        ### Response: 
        {question}
        """    
        return (
            "### Instruction: \n"
            f"{example['instruction']}\n\n"
            f"### Input: \n"
            f"{example['input']}\n\n"
            f"\n\n### Response: \n"
            f"{example['output']}"
            )
    
    def prepare_data(self):
        logger.info('-----:----- Preparing dataset -----:-----')
                
        train_df = pd.read_json(self.train_path)
        dev_df = pd.read_json(self.dev_path)
    
        train_dataset = Dataset.from_pandas(train_df)
        dev_dataset = Dataset.from_pandas(dev_df)
        
        return train_dataset, dev_dataset
    

    def train(self):
        login(token='')

        bnb_selfig = BitsAndBytesselfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model,
            quantization_selfig=bnb_selfig,
            use_cache=False,
            use_flash_attention_2= self.use_flash_attention,
            device_map="auto",
        )
        model.selfig.pretraining_tp = 1


        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"



        # LoRA selfig based on QLoRA paper
        peft_selfig = Loraselfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )


        # prepare model for training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_selfig)


        args = TrainingArguments(
            output_dir=self.checkpoint,
            num_train_epochs=10,
            do_train=True,
            do_eval=True,
            evaluation_strategy='steps',
            eval_steps=500,
            save_strategy='steps',
            save_steps=500,
            per_device_train_batch_size=self.per_device_eval_batch_size if self.use_flash_attention else self.per_device_train_batch_size,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            logging_steps=200,
            learning_rate=self.lr,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            bf16=True,
            tf32=True,
            seed=42,
            max_grad_norm=0.3,
            lr_scheduler_type="linear",
            report_to='none',
            save_total_limit=3
        )


        train_dataset, dev_dataset = prepare_data()
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = SFTTrainer(
            model=model,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            peft_selfig=peft_selfig,
            max_seq_length=self.max_len,
            tokenizer=tokenizer,
            packing=True,
            formatting_func=formatting_func,
            args=args,
        )



trainer.train()
