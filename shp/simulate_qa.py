import json
import sys
sys.path.append('../..')
from api_wrapper.api_wrapper import multiprocess_api
from prompts.load_prompt import get_prompts_by_task


def simulate_qa(model, orig_inputs, orig_tm_preds, sim_inputs_list):
	assert len(orig_inputs) == len(orig_tm_preds) == len(sim_inputs_list)
	num_examples = len(orig_inputs)
	prompts = get_prompts_by_task('shp-simqa-fix',
								  [{'starter_context': orig_input['context'],
									'starter_response0': orig_input['options'][0],
									'starter_response1': orig_input['options'][1],
									'starter_preferred_idx_plus_1':
										orig_tm_pred['pred_ans'] + 1 if orig_tm_pred['pred_ans'] is not None
									   else 'Neither',
									'starter_reason': orig_tm_pred['pred_expl'],
									'followup_context': sim_input['context'],
									'followup_response0': sim_input['options'][0],
									'followup_response1': sim_input['options'][1]}
								   for orig_input, orig_tm_pred, sim_inputs in
								   zip(orig_inputs, orig_tm_preds, sim_inputs_list)
								   for sim_input in sim_inputs])
	# deduplicate the prompts before calling the API to save time
	deduplicated_prompts = list(set(prompts))
	pred_expls = multiprocess_api(model=model, prompts=deduplicated_prompts,
								  bsz=8, num_processes=12,
								  temperature=0, max_tokens=100, stop='\n\n')
	assert len(pred_expls) == len(deduplicated_prompts)
	# add duplicate prompts back
	prompt2pred_expl = {prompt: pred_expl for prompt, pred_expl in zip(deduplicated_prompts, pred_expls)}
	pred_expls = [prompt2pred_expl[prompt] for prompt in prompts]
	assert len(pred_expls) == len(prompts)
	# extract answers
	pred_answers = []
	for pred_expl in pred_expls:
		if 'No, I cannot confidently guess' in pred_expl:
			pred_answers.append('unknown')
		elif 'Yes, I can confidently guess' in pred_expl:
			select0 = "I would guess that the robot will choose Candidate Response 1" in pred_expl
			select1 = "I would guess that the robot will choose Candidate Response 2" in pred_expl
			if select0 == select1:
				pred_answers.append('neither')
			elif select0:
				pred_answers.append(0)
			elif select1:
				pred_answers.append(1)
		else:
			pred_answers.append('neither')
	assert len(pred_answers) == len(pred_expls)
	preds = [{'pred_ans': pred_ans, 'pred_expl': pred_expl} for pred_ans, pred_expl in zip(pred_answers, pred_expls)]
	# regroup preds according to examples (multiple simulation questions correspond to each original question)
	example_preds = []
	cur = 0
	for ex_idx in range(num_examples):
		example_preds.append(preds[cur: cur + len(sim_inputs_list[ex_idx])])
		cur += len(sim_inputs_list[ex_idx])
	assert cur == len(preds)
	return example_preds

