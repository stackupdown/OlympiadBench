import json
from evaluators.evaluator import Evaluator
from time import sleep
import re, os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import SamplingParams

class Qwen_Evaluator(Evaluator):
    def __init__(self, model_name, cuda_device_id=0, k=-1):
        super(Qwen_Evaluator, self).__init__(model_name, k)
        # model_name = "../../deepseek-math-7b-rl"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=f'cuda:{cuda_device_id}', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        print("model.generation_config.pad_token_id: ", self.model.generation_config.pad_token_id)
        # self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = 151645
        print("model.generation_config.eos_token_id: ", self.model.generation_config.eos_token_id)

    def make_input(self, prompt, question_content):
        content = prompt + '\n' + question_content + '\n'
        # Adding the prompt recommended in Deepseek-Math's huggingface repository
        if self.is_chinese:
            content += '请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。'
        else:
            content += 'Please reason step by step, and put your final answer within \\boxed{}.'
        messages = [{
            'role': 'user',
            'content': content
        }]
        return messages

    def get_answer(self, input):
        # print(input)
        input_tensor = self.tokenizer.apply_chat_template(input, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(input_tensor.to(self.model.device), max_new_tokens=8192) # 2048
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        # print(result)
        return result

from vllm import LLM, SamplingParams

class Qwen_VLLM_Evaluator(Evaluator):
    # TODO: 未完成
    def __init__(self, model_name, cuda_device_id=0, k=-1):
        super(Qwen_VLLM_Evaluator, self).__init__(model_name, k)
        # 加载Tokenizer（与原始代码一致）
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = LLM(
            model=model_name,
            tokenizer=model_name,
            trust_remote_code=True,
            temperature=0.0,
            dtype="bfloat16",
            tensor_parallel_size=2,
            # gpu_memory_utilization=0.9,  # 可选内存配置
        )

        self.sampling_params = SamplingParams(
            top_k=self.k if self.k > 0 else -1,  # 处理k=-1的情况
            stop_token_ids=[151645], max_tokens=2048, temperature=0.0
        )

        print("Tokenizer pad_token_id:", self.tokenizer.pad_token_id)
        print("Sampling stop_token_ids:", self.sampling_params.stop_token_ids)

    def make_input(self, prompt, question_content):
        content = prompt + '\n' + question_content + '\n'
        # Adding the prompt recommended in Deepseek-Math's huggingface repository
        if self.is_chinese:
            content += '请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。'
        else:
            content += 'Please reason step by step, and put your final answer within \\boxed{}.'
        messages = [{
            'role': 'user',
            'content': content
        }]
        return messages

    def get_answer(self, input):
        # print(input)
        input_tensor = self.tokenizer.apply_chat_template(input, add_generation_prompt=True, return_tensors="pt")
        prompt_ids = self.tokenizer.encode(input, add_special_tokens=False)['input_ids']
        results = self.model.generate(
            prompt_token_ids=[prompt_ids], sampling_params=self.sampling_params, use_tqdm=False
        )
        completions = []
        for result in results:
            for output in result.outputs:
                completions.append(output.text)
        # print(result)
        return completions[0]
