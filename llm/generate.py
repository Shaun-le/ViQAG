import torch
import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from plms.utils import save_result

# load base LLM model and tokenizer
model_checkpoint = '/home/int2-user/qag/checkpoint-13B/checkpoint-10000'
model = AutoPeftModelForCausalLM.from_pretrained(
    model_checkpoint,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
df = pd.read_json('/home/int2-user/qag/data/ViMMRC2.0/vimmrc2.0_test.json')

preds = []
for idx in range(len(df)):
    sample = df.iloc[idx]
    
    prompt: str = ("### Instruction: \n"
    f"{sample['instruction']}\n\n"
    "### Input: \n"
    f"{sample['input']}\n\n"
    f"### Response: \n"           
    )
    # prompt = f"""### Instruction:
    # {sample['instruction']}

    # ### Input:
    # {sample['input']}

    # ### Response:
    # """

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3096).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, top_p=0.75, temperature=0.1)

    pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    # print(f"Prompt:\n{prompt}\n")
    print(f"Generated repsonse:\n{pred}")
    print(f"Ground truth:\n{sample['output']}\n\n")
    
    preds.append(pred)
    save_result(path='/home/int2-user/qag/tune/13B/vimmrc2.0_4.csv', result={'prediction': pred, 'reference': sample['output']})