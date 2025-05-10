import os, sys, re
import json
from tqdm import tqdm
import argparse

sys.path.append('code')
from math_judger import MathJudger
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('[%(levelname)s][%(asctime)s-%(name)s]-[%(filename)s:%(lineno)d]-%(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def extract_answer(is_chinese, model_output, is_deepseek=False):
	# deepseekmath has special answering format
	if is_deepseek:
		if is_chinese:
			matches = re.findall('## 解题答案(.*)', model_output)
		else:
			matches = re.findall('The answer is: (.*)', model_output)
    
		# 检测是否至少找到一个匹配，如果没有就直接整个送进去找\boxed{}
		if matches:
			# 如果找到多个匹配，取最后一个
			model_answer = matches[-1].strip()
			return model_answer
		else:
			return model_output
		
	if is_chinese:
		matches = re.findall('所以最终答案是(.*)', model_output)
	else:
		matches = re.findall('So the final answer is (.*)', model_output)

	# 检测是否至少找到一个匹配，如果没有就直接整个送进去找\boxed{}
	if matches:
		# 如果找到多个匹配，取最后一个
		model_answer = matches[-1].strip()
		return model_answer
	else:
		return model_output


def judge_result(args):
	judger = MathJudger()
	# os.path.join('generated', dataset)
	for dataset in os.listdir(args.path):
		print('-'*10 + dataset + '-'*10)
		if "TP" in dataset:
			print("Warning: Theorem proving problems cannot currently be automatically assessed.")
			continue
		is_chinese = True if 'zh' in dataset else False # 也没有这个属性了
		dataset_path = os.path.join(args.path, dataset)
		for model in os.listdir(dataset_path):
			is_deepseek = True if 'deepseek' in model else False
			logger.info(model)
			results_path = os.path.join(args.path, dataset, model)
			logger.info(results_path)
			if os.path.exists(results_path):
				full_num = 0
				machine_scored_num = 0
				correct_num = 0
				available_id_list = []	# deduplication
				merged_result = []
				for single_result in tqdm(os.listdir(results_path)):
					if single_result[-5:] != '.json':
						continue
					single_result_path = os.path.join(results_path, single_result) # 具体的某个结果文件
					single_result_dict = []
					with open(single_result_path, 'r', encoding='utf-8') as f:
						single_result_dict = json.load(f)
						for id, question in enumerate(single_result_dict):
							if (len(question['model_output'][model]['raw_output'])>0 and question['model_output'][model]['raw_output'] != '<Inappropriate content in response>' and question['model_output'][model]['raw_output']!='<No response>' and ('code:' not in question['model_output'][model]['raw_output'] or 'message:' not in question['model_output'][model]['raw_output'])):
								if question['id'] in available_id_list:	# 重复数据
									continue
								else:
									available_id_list.append(question['id'])
							full_num += 1 # 这俩没用到
							machine_scored_num += 1 # 这俩没用到
							
							model_answer = question['model_output'][model]['raw_output']
							model_answer = extract_answer(is_chinese, model_answer, is_deepseek)

							answer_type = question['answer_type']
							if 'Tuple' in answer_type: # 目前可机评的数据中 没有 need_human_evaluate
								judge_result = judger.judge(model_answer, question['final_answer'][0])
							else:
								if question['error']:
									if ',' in question['error']:
										precisions = question['error'].split(',')
										precisions = [float(p) if p else 1e-8 for p in precisions]
										judge_result = judger.judge(model_answer, question['final_answer'][0], precisions)
									else:
										precision = float(question['error'])
										judge_result = judger.judge(model_answer, question['final_answer'][0], precision)
								else:
									judge_result = judger.judge(model_answer, question['final_answer'][0])

							if judge_result:
								correct_num += 1 # 貌似也没用到
							single_result_dict[id]['model_output'][model]['answer'] = model_answer
							single_result_dict[id]['model_output'][model]['correctness'] = judge_result
						merged_result += single_result_dict # 保留所有的处理结果

				if not os.path.exists(os.path.join('merged_result', model)):
					os.makedirs(os.path.join('merged_result', model))
				with open(os.path.join('merged_result', model, f'{dataset}.json'), 'w', encoding='utf-8') as f:
					json.dump(merged_result, f, ensure_ascii=False, indent=4)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', type=str, required=True) # generated
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	judge_result(args)
