import json
import sys
sys.path.append('../..')
from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task
from copy import deepcopy
import random


def simulate_qg(model, orig_inputs, orig_tm_preds, top_p, num_samples, with_context):
	# orig_tm_preds = List[{'pred_ans': str, 'pred_expl': str}]
	# orig_inputs = List[str]
	assert len(orig_inputs) == len(orig_tm_preds)
	num_examples = len(orig_inputs)
	prompts = get_prompts_by_task('shp-simqg-withcontext' if with_context else 'shp-simqg-nocontext',
								  [{'starter_context': orig_input['context'],
									'starter_response0': orig_input['options'][0],
									'starter_response1': orig_input['options'][1],
									'starter_preferred_idx_plus_1':
										orig_tm_pred['pred_ans'] + 1 if orig_tm_pred['pred_ans'] is not None
										else 'Neither',
									'starter_reason': orig_tm_pred['pred_expl']}
								   for orig_input, orig_tm_pred in zip(orig_inputs, orig_tm_preds)])
	# repeat the prompts for self.num_samples times
	prompts = [prompt for prompt in prompts for _ in range(num_samples)]
	assert len(prompts) == num_examples * num_samples
	responses = multiprocess_api(model=model, prompts=prompts, bsz=8, num_processes=12,
								 temperature=1, top_p=top_p, max_tokens=512)
	assert len(responses) == num_examples * num_samples
	# parse generated inputs
	sim_inputs = []
	for response in responses:
		response = response.strip()
		context_start_idx = response.find("Context:")
		response0_start_idx = response.find("Candidate Response 1:")
		response1_start_idx = response.find("Candidate Response 2:")
		if not ((context_start_idx == 0) and (response0_start_idx > context_start_idx) and (response1_start_idx > response0_start_idx)):
			# print(context_start_idx, response0_start_idx, response1_start_idx)
			# print(response)
			# print('\n\n\n\n\n\n\n')
			sim_inputs.append(None)
		else:
			context = response[context_start_idx+len("Context:"): response0_start_idx].strip()
			response0 = response[response0_start_idx+len("Candidate Response 1:"): response1_start_idx].strip()
			response1 = response[response1_start_idx+len("Candidate Response 2:"):].strip()
			if 'Human:' in response1: # the next example already started
				# print('Warning')
				response1 = response1[: response1.index('Human:')].strip()
			sim_inputs.append({'context': context, 'options': [response0, response1]})
	# group the generated outputs by examples
	assert len(sim_inputs) == num_examples * num_samples
	example_siminputs = []
	for ex_idx in range(num_examples):
		example_siminputs.append(
			[sim_input for sim_input in sim_inputs[ex_idx * num_samples: (ex_idx + 1) * num_samples]
			 if sim_input is not None])
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


