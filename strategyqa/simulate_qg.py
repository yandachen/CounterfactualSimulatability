import json
import random
import sys
sys.path.append('../..')
from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task
import pickle as pkl
from copy import deepcopy


def simulate_qg(model, orig_inputs, orig_tm_preds, top_p, num_samples):
	# orig_tm_preds = List[{'pred_ans': str, 'pred_expl': str}]
	# orig_inputs = List[str]
	assert len(orig_inputs) == len(orig_tm_preds)
	num_examples = len(orig_inputs)
	prompts = get_prompts_by_task('strategyqa-simqg',
								  [{'orig_qn': orig_input['question'], 'orig_qa_tm_expl': orig_tm_pred['pred_expl']}
								   for orig_input, orig_tm_pred in zip(orig_inputs, orig_tm_preds)])
	# repeat the prompts for self.num_samples times
	prompts = [prompt for prompt in prompts for _ in range(num_samples)]
	assert len(prompts) == num_examples * num_samples
	responses = multiprocess_api(model=model, prompts=prompts, bsz=20, num_processes=16 if model =='gpt3' else 10,
											  temperature=1, top_p=top_p, max_tokens=50, stop='\n\n')
	sim_inputs = []
	# for response in responses:
	# 	lines = response.split("\n")
	# 	if len(lines) != 2 or (not lines[0].strip().endswith("?")) or (
	# 			not lines[0].strip().startswith('Assistant: here is my response.')):
	# 		sim_inputs.append(None)
	# 	else:
	# 		line = lines[0].strip()
	# 		question = line[len('Assistant: here is my response.'):].strip()
	# 		sim_inputs.append({'question': question.strip()})

	for response in responses:
		lines = response.split("\n")
		if len(lines) != 2 or not lines[0].strip().endswith("?"):
			sim_inputs.append(None)
		else:
			sim_inputs.append({'question': lines[0].strip()})

	# group the generated outputs by examples
	assert len(sim_inputs) == num_examples * num_samples
	example_siminputs = []
	for ex_idx in range(num_examples):
		ex_sim_inputs = [sim_input for sim_input in sim_inputs[ex_idx * num_samples: (ex_idx + 1) * num_samples]
			 if sim_input is not None]
		seen_questions = set()
		unique_idxs = []
		for idx in range(len(ex_sim_inputs)):
			qn = ex_sim_inputs[idx]['question']
			if qn not in seen_questions:
				seen_questions.add(qn)
				unique_idxs.append(idx)
		ex_sim_inputs = [ex_sim_inputs[idx] for idx in unique_idxs]
		example_siminputs.append(ex_sim_inputs)
	assert len(example_siminputs) == num_examples
	return example_siminputs


def _check_two_dict_same(dict1, dict2):
	if dict1.keys() != dict2.keys():
		return False
	for key in dict1:
		if dict1[key] != dict2[key]:
			return False
	return True


def mix_sim_inputs(model1_siminputs, model2_siminputs, sample_num):
	mixed_samples = []
	model1_siminputs, model2_siminputs = deepcopy(model1_siminputs), deepcopy(model2_siminputs)
	for sample_idx in range(sample_num):
		add_sample = None
		if len(model1_siminputs) == 0 and len(model2_siminputs) == 0:
			return mixed_samples
		elif len(model1_siminputs) > 0 and (sample_idx % 2 == 0 or len(model2_siminputs) == 0):
			add_sample = random.sample(model1_siminputs, 1)[0]
			mixed_samples.append(add_sample)
		else:
			assert len(model2_siminputs) > 0 and (sample_idx % 2 == 1 or len(model1_siminputs) == 0)
			add_sample = random.sample(model2_siminputs, 1)[0]
			mixed_samples.append(add_sample)
		# remove duplicates
		model1_siminputs = [ex for ex in model1_siminputs if not _check_two_dict_same(ex, add_sample)]
		model2_siminputs = [ex for ex in model2_siminputs if not _check_two_dict_same(ex, add_sample)]
	return mixed_samples
