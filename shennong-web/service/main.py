
import os
import uvicorn
import json
import traceback
import uuid
import argparse
import sys
import torch
from peft import PeftModel
import transformers

import argparse
import warnings
import os

from os.path import abspath, dirname
from loguru import logger
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from message_store import MessageStore
from transformers import AutoModel, AutoTokenizer
from errors import Errors
import gen_data

log_folder = os.path.join(abspath(dirname(__file__)), "log")
logger.add(os.path.join(log_folder, "{time}.log"), level="INFO")


DEFAULT_DB_SIZE = 100000

massage_store = MessageStore(db_path="message_store.json", table_name="chatgpt", max_size=DEFAULT_DB_SIZE)
# Timeout for FastAPI
# service_timeout = None

app = FastAPI()


stream_response_headers = {
		"Content-Type": "application/octet-stream",
		"Cache-Control": "no-cache",
}


@app.post("/config")
async def config():
		return JSONResponse(content=dict(
				message=None,
				status="Success",
				data=dict()
		))

def generate_prompt(instruction, input=None):
		if input:
				return f"""The following is a conversation between an Pharmacologist called Shennong and a human user called User.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
		else:
				return f"""The following is a conversation between an Pharmacologist called Shennong and a human user called User.

### Instruction:
{instruction}

### Response:"""


async def process(prompt, options, params, message_store, is_knowledge, history=None,**kwargs,):
		"""

		"""

		if not prompt:
				logger.error("Prompt is empty.")
				yield Errors.PROMPT_IS_EMPTY.value
				return


		try:
				chat = {"role": "user", "content": prompt}


				if options:
						parent_message_id = options.get("parentMessageId")
						messages = message_store.get_from_key(parent_message_id)
						if messages:
								messages.append(chat)
						else:
								messages = []
				else:
						parent_message_id = None
						messages = [chat]


				messages = messages[-params['memory_count']:]


				history_formatted = []
				if options is not None:
						history_formatted = []

						for i, old_chat in enumerate(messages):
								if  old_chat['role'] == "user":
										history_formatted.append("User: "+old_chat['content'])
								elif old_chat['role'] == "AI":
										history_formatted.append("AI: "+old_chat['content'])
								else:
										continue
				
				input="\n".join(history_formatted)+'\n+User: '+prompt
				input=generate_prompt(input)
				inputs = tokenizer(input, return_tensors="pt")
				input_ids = inputs["input_ids"].to(device)



				uid = "chatglm"+uuid.uuid4().hex
				footer=''
				# yield footer

				temperature=0.1
				top_p=0.75
				top_k=40
				num_beams=1
				max_new_tokens=128
				repetition_penalty=1.0
				max_memory=256
                                
				generation_config = GenerationConfig(
							                                temperature=0.1,
                                top_p=0.75,
                                top_k=40,
                                num_beams=1,
                                **kwargs,
                                )				
				with torch.no_grad():
						generation_output = model.generate(
						input_ids=input_ids,
						generation_config=generation_config,
						return_dict_in_generate=True,
						output_scores=True,
						max_new_tokens=max_new_tokens,
						repetition_penalty=float(repetition_penalty),
						)
				s = generation_output.sequences[0]
				output = tokenizer.decode(s)
				output = output.split("### Response:")[1].strip()
				output = output.replace("Belle", "Vicuna")
				if 'User:' in output:
						output = output.split("User:")[0]
				message = json.dumps(dict(
					role="AI",
					id=uid,
					parentMessageId=parent_message_id,
					text=output + footer,
				))
				yield "data: " + message
		except:
				err = traceback.format_exc()
				logger.error(err)
				yield Errors.SOMETHING_WRONG.value
				return

		try:
				# save to cache
				chat = {"role": "AI", "content": output}
				messages.append(chat)
				parent_message_id = uid
				message_store.set(parent_message_id, messages)
		except:
				err = traceback.format_exc()
				logger.error(err)


@app.post("/chat-process")
async def chat_process(request_data: dict):
		prompt = request_data['prompt']
		max_length = request_data['max_length']
		top_p = request_data['top_p']
		temperature = request_data['temperature']
		options = request_data['options']
		if request_data['memory'] == 1 :
				memory_count = 5
		elif request_data['memory'] == 50:
				memory_count = 20
		else:
				memory_count = 999

		if 1 == request_data["top_p"]:
				top_p = 0.2
		elif 50 == request_data["top_p"]:
				top_p = 0.5
		else:
				top_p = 0.9
		if temperature is None:
				temperature = 0.9
		if top_p is None:
				top_p = 0.7
		is_knowledge = request_data['is_knowledge']
		params = {
				"max_length": max_length,
				"top_p": top_p,
				"temperature": temperature,
				"memory_count": memory_count
		}
		answer_text = process(prompt, options, params, massage_store, is_knowledge)
		return StreamingResponse(content=answer_text, headers=stream_response_headers, media_type="text/event-stream")


if __name__ == "__main__":
		parser = argparse.ArgumentParser(description='Simple API server for ChatGLM-6B')
		parser.add_argument('--device', '-d', help='使用设备，cpu或cuda:0等', default='cpu')
		parser.add_argument('--quantize', '-q', help='量化等级。可选值：16，8，4', default=16)
		parser.add_argument('--host', '-H', type=str, help='监听Host', default='0.0.0.0')
		parser.add_argument('--port', '-P', type=int, help='监听端口号', default=3002)


		assert (
			"LlamaTokenizer" in transformers._import_structure["models.llama"]
		), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
		from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


		parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
		parser.add_argument("--lora_path", type=str, default="./lora-Vicuna/checkpoint-final")
		parser.add_argument("--use_local", type=int, default=1)
		args = parser.parse_args()

		tokenizer = LlamaTokenizer.from_pretrained(args.model_path)

		LOAD_8BIT = True
		BASE_MODEL = args.model_path
		LORA_WEIGHTS = args.lora_path

		# fix the path for local checkpoint
		lora_bin_path = os.path.join(args.lora_path, "adapter_model.bin")
		print(lora_bin_path)
		if not os.path.exists(lora_bin_path) and args.use_local:
				pytorch_bin_path = os.path.join(args.lora_path, "pytorch_model.bin")
				print(pytorch_bin_path)
				if os.path.exists(pytorch_bin_path):
						os.rename(pytorch_bin_path, lora_bin_path)
						warnings.warn("The file name of the lora checkpoint'pytorch_model.bin' is replaced with 'adapter_model.bin'")
				else:
						assert ('Checkpoint is not Found!')
		if torch.cuda.is_available():
				device = "cuda"
		else:
				device = "cpu"

		try:
				if torch.backends.mps.is_available():
						device = "mps"
		except:
				pass

		if device == "cuda":
				model = LlamaForCausalLM.from_pretrained(
					BASE_MODEL,
					load_in_8bit=LOAD_8BIT,
					torch_dtype=torch.float16,
					device_map="auto",
				)
				model = PeftModel.from_pretrained(
					model,
					LORA_WEIGHTS,
					torch_dtype=torch.float16,
					device_map={'': 0}
				)
		elif device == "mps":
				model = LlamaForCausalLM.from_pretrained(
					BASE_MODEL,
					device_map={"": device},
					torch_dtype=torch.float16,
				)
				model = PeftModel.from_pretrained(
					model,
					LORA_WEIGHTS,
					device_map={"": device},
					torch_dtype=torch.float16,
				)
		else:
				model = LlamaForCausalLM.from_pretrained(
					BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
				)
				model = PeftModel.from_pretrained(
					model,
					LORA_WEIGHTS,
					device_map={"": device},
				)
if not LOAD_8BIT:
		model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
		model = torch.compile(model)

uvicorn.run(app, host=args.host, port=args.port)

