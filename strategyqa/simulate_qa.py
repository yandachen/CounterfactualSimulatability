import json
import random
import sys
from collections import Counter
import numpy as np
sys.path.append('../..')
from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task
import pickle as pkl
from copy import deepcopy


def extract_sim_qa_ans(sim_qa_expl):
	cannot_guess = 'I cannot guess' in sim_qa_expl
	guess_yes = 'answer yes' in sim_qa_expl
	guess_no = 'answer no' in sim_qa_expl
	if not (cannot_guess + guess_yes + guess_no == 1):
		return 'neither'
	elif cannot_guess:
		return 'unknown'
	elif guess_yes:
		return 'yes'
	elif guess_no:
		return 'no'
	else:
		raise NotImplementedError


def simulate_qa(model, orig_inputs, orig_tm_preds, sim_inputs_list, include_expl=True,
				majority_vote=None,
				annotated_examples=None):
	assert len(orig_inputs) == len(orig_tm_preds) == len(sim_inputs_list)
	num_examples = len(orig_inputs)

	if annotated_examples is None:
		if include_expl:
			prompts = get_prompts_by_task('strategyqa-simqa-withexpl-fix',
										  [{'orig_qn': orig_input['question'], 'orig_qa_tm_expl': orig_tm_pred['pred_expl'], 'sim_qn': sim_input['question']}
										   for orig_input, orig_tm_pred, sim_inputs in zip(orig_inputs, orig_tm_preds, sim_inputs_list)
										   for sim_input in sim_inputs])
		else:
			prompts = get_prompts_by_task('strategyqa-simqa-noexpl',
										  [{'orig_qn': orig_input['question'],
											'orig_qa_tm_expl': orig_tm_pred['pred_expl'],
											'sim_qn': sim_input['question']}
										   for orig_input, orig_tm_pred, sim_inputs in
										   zip(orig_inputs, orig_tm_preds, sim_inputs_list)
										   for sim_input in sim_inputs])
	else:
		assert len(orig_inputs) == len(annotated_examples)
		task_prompt_with_expl = json.load(open('../../prompts/prompts.json'))['strategyqa-simqa-withexpl-mentioncot']
		task_prompt_no_expl = json.load(open('../../prompts/prompts.json'))['strategyqa-simqa-noexpl']
		assert task_prompt_with_expl['instruction'] == task_prompt_no_expl['instruction']
		prompt_prefix = str(task_prompt_with_expl['instruction'])

		dem_examples_prefix = []
		if include_expl:
			for dem in task_prompt_with_expl['dem_examples']:
				dem_examples_prefix.append(task_prompt_with_expl['template_with_label'].format(**dem))
		else:
			for dem in task_prompt_no_expl['dem_examples']:
				dem_examples_prefix.append(task_prompt_no_expl['template_with_label'].format(**dem))

		prompts = []
		for orig_input, orig_tm_pred, sim_inputs, annotated_exs in \
				zip(orig_inputs, orig_tm_preds, sim_inputs_list, annotated_examples):
			# append the annotater-annotated examples
			formatted_output_annotated_exs = []
			for annotated_ex in annotated_exs:
				annotated_ex_copy = {key: annotated_ex[key] for key in annotated_ex if key != 'sim_qa_ans'}
				annotated_ex_copy['sim_qa_ans'] = {'yes': 'The robot will likely answer yes.',
												   'no': 'The robot will likely answer no.',
												   'unknown': "I cannot guess the robot's answer to the follow-up "
															  "question based on its response to the starter question."
												   }[annotated_ex['sim_qa_ans']]
				formatted_output_annotated_exs.append(annotated_ex_copy)
			annotated_exs = formatted_output_annotated_exs
			ex_dem_examples_prefix = dem_examples_prefix + [task_prompt_no_expl['template_with_label'].format(**annotated_ex)
														for annotated_ex in annotated_exs]
			random.shuffle(ex_dem_examples_prefix)
			ex_prompt_prefix = prompt_prefix + ''.join(ex_dem_examples_prefix)
			for sim_input in sim_inputs:
				test_example = {'orig_qn': orig_input['question'], 'orig_qa_tm_expl': orig_tm_pred['pred_expl'],
								'sim_qn': sim_input['question']}
				if include_expl:
					ex_prompt = ex_prompt_prefix + task_prompt_with_expl['template_no_label'].format(**test_example)
				else:
					ex_prompt = ex_prompt_prefix + task_prompt_no_expl['template_no_label'].format(**test_example)
				prompts.append(ex_prompt)

	# deduplicate the prompts before calling the API to save time
	deduplicated_prompts = list(set(prompts))
	if majority_vote is None or majority_vote == 1:
		pred_expls = multiprocess_api(model=model, prompts=deduplicated_prompts, bsz=16,
									  num_processes=8,
									  # num_processes=16,
									  temperature=0, max_tokens=200, stop='\n', n=1)
	else:
		pred_expls = multiprocess_api(model=model, prompts=deduplicated_prompts, bsz=4,
									  num_processes=4,
									  # num_processes=20,
									  temperature=1, max_tokens=200, stop='\n', n=majority_vote)
	assert len(pred_expls) == len(deduplicated_prompts)
	# add duplicate prompts back
	prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
	pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
	assert len(pred_expls) == len(prompts)

	# extract answers
	if majority_vote is None or majority_vote == 1:
		preds = []
		for pred_expl in pred_expls:
			preds.append({'pred_ans': extract_sim_qa_ans(pred_expl), 'pred_expl': pred_expl})
	else:
		preds = []
		for pred_expl_samples in pred_expls:
			ex_preds = [{'pred_ans': extract_sim_qa_ans(pred_expl), 'pred_expl': pred_expl}
						  for pred_expl in pred_expl_samples]
			ex_pred_answers = [pred['pred_ans'] for pred in ex_preds]
			counter = Counter(ex_pred_answers)
			max_count = np.max([counter[item] for item in counter])
			most_frequent_answers = [ans for ans in counter if counter[ans] == max_count]
			majority_ans = random.sample(most_frequent_answers, 1)[0]
			preds.append({'pred_ans': majority_ans, 'majority_vote_details': ex_preds})

	# regroup preds according to examples (multiple simulation questions correspond to each original question)
	assert len(preds) == len(prompts)
	example_preds = []
	cur = 0
	for ex_idx in range(num_examples):
		example_preds.append(preds[cur: cur + len(sim_inputs_list[ex_idx])])
		cur += len(sim_inputs_list[ex_idx])
	assert cur == len(preds)
	return example_preds

