from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
#from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from peft import AutoPeftModelForCausalLM
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os

app = Flask(__name__)
CORS(app)

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

model_map = {
    'pipelinevit5': ['namngo/pipeline_vit5_ae','namngo/pipeline_vit5_qg'],
    'pipelinebartpho':'',
    'multitaskvit5':'',
    'multitaskbartpho':'',
    'end2endvit5': '',
    'end2endbartpgo': '',
    'inllama13': 'shnl/llama2-13B-QAG-10000',
    'inllama7': 'shnl/llama2-7B-QAG-17500'
}

template = """Context: {context}"""
prompt = PromptTemplate(template=template, input_variables=["context"])
def create_llm_chain(model_id, llm_type='plms'):
    if llm_type == 'plms':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    elif llm_type == 'llms':
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=True,)
    else:
        raise ValueError("Invalid LLM type. Choose either 'plms' or 'llms'.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024)
    local_llm = HuggingFacePipeline(pipeline=pipe)
    llm_chain = LLMChain(prompt=prompt, llm=local_llm)
    return llm_chain

@app.route('/gen', methods=['POST'])
def receive_data():
    try:
        data = request.json.get('data')
        name = request.json.get('name')
        if name == 'inllama7' or name == 'inllama13':
            llm_type = 'llms'
        else:
            llm_type = 'plms'
        llm_chain = create_llm_chain(model_map[name], llm_type)
        qag = llm_chain.run(context=data)
        stext = qag.split(' [*] ')
        formatted_text = '\n\n'.join(item.replace(', answer', '\nanswer') for item in stext if item)
        print(formatted_text)
        return jsonify({'lowercase_data': formatted_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)