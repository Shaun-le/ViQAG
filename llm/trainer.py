import re
import torch
import json, os
import evaluate
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from datasets import Dataset
from pyvi import ViTokenizer
from dataclasses import dataclass
from huggingface_hub import login
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from nltk.translate.bleu_score import corpus_bleu
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics = evaluate.load('rouge')
#os.makedirs('/viquad', mode=0o777, exist_ok=True)

@dataclass
class Config:
    pretrained_model_name_or_path: str = 'manhtt-079/llama-2-13b'
    checkpoint: str = './checkpoint-13B'
    max_len: int = 2048
    seed: int = 42
    num_proc: int = 16
    train_path: str = './data/train.json'
    dev_path: str = './data/dev.json'
    vico_qa: str = './data/ViCoQA/vico_test.json'
    vimmrc1: str = './data/ViMMRC1.0/vimmrc_test.json'
    vimmrc2: str = './data/ViMMRC1.0/vimmrc_test.json'
    vinews_qa: str = './data/ViNewsQA/vinewsqa_test.json'
    viquad: str = './data/ViQuAD/viquad_test.json'
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    num_epochs: int = 10
    lr: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.015
    use_flash_attention: bool = False
    
    
conf = Config()
    

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

def prepare_data():
    logger.info('-----:----- Preparing dataset -----:-----')
            
    train_df = pd.read_json(conf.train_path)
    dev_df = pd.read_json(conf.dev_path)
    # vico_qa = pd.read_json(conf.vico_qa)
    # vimmrc1 = pd.read_json(conf.vimmrc1)
    # vimmrc2 = pd.read_json(conf.vimmrc2)
    # vinews_qa = pd.read_json(conf.vinews_qa)
    # viquad = pd.read_json(conf.viquad)
    
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)
    # vico_qa_dataset = Dataset.from_pandas(vico_qa)
    # vimmrc1_dataset = Dataset.from_pandas(vimmrc1)
    # vimmrc2_dataset = Dataset.from_pandas(vimmrc2)
    # vinews_qa_dataset = Dataset.from_pandas(vinews_qa)
    # viquad_dataset = Dataset.from_pandas(viquad)
    
    # train_dataset = train_dataset.map(formatting_func, num_proc=conf.num_proc).remove_columns(['input', 'output', 'instruction'])
    # dev_dataset = dev_dataset.map(formatting_func, num_proc=conf.num_proc).remove_columns(['input', 'output', 'instruction'])
    # vico_qa_dataset = vico_qa_dataset.map(formatting_func, num_proc=conf.num_proc).remove_columns(['input', 'output', 'instruction'])
    # vimmrc1_dataset = vimmrc1_dataset.map(formatting_func, num_proc=conf.num_proc).remove_columns(['input', 'output', 'instruction'])
    # vimmrc2_dataset = vimmrc2_dataset.map(formatting_func, num_proc=conf.num_proc).remove_columns(['input', 'output', 'instruction'])
    # vinews_qa_dataset = vinews_qa_dataset.map(formatting_func, num_proc=conf.num_proc).remove_columns(['input', 'output', 'instruction'])
    # viquad_dataset = viquad_dataset.map(formatting_func, num_proc=conf.num_proc).remove_columns(['input', 'output', 'instruction'])

    
    return train_dataset, dev_dataset


login(token='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')
# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    conf.pretrained_model_name_or_path,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=conf.use_flash_attention,
    device_map="auto",
)
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(conf.pretrained_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"



# LoRA config based on QLoRA paper
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)


args = TrainingArguments(
    output_dir=conf.checkpoint,
    num_train_epochs=10,
    do_train=True,
    do_eval=True,
    evaluation_strategy='steps',
    eval_steps=500,
    save_strategy='steps',
    save_steps=500,
    per_device_train_batch_size=conf.per_device_eval_batch_size if conf.use_flash_attention else conf.per_device_train_batch_size,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=200,
    learning_rate=conf.lr,
    warmup_ratio=conf.warmup_ratio,
    weight_decay=conf.weight_decay,
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
    peft_config=peft_config,
    max_seq_length=conf.max_len,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=formatting_func,
    args=args,
)



trainer.train()
