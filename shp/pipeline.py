from task_qa import task_qa, task_qa_sim_inputs_list
from simulate_qg import simulate_qg, mix_sim_inputs
from simulate_qa import simulate_qa
import json
import time
import os
import pickle as pkl
from tqdm import trange


def run_task_save_results(task_function, out_file, ex_idxs, **kwargs):
	all_preds = {}
	if os.path.exists(out_file):
		all_preds = pkl.load(open(out_file, 'rb'))
	ex_idxs = [ex_idx for ex_idx in ex_idxs if ex_idx not in all_preds]
	for key in kwargs:
		if type(kwargs[key]) == list or type(kwargs[key]) == dict:
			kwargs[key] = [kwargs[key][ex_idx] for ex_idx in ex_idxs]
	preds = task_function(**kwargs)
	assert type(preds) == list and len(preds) == len(ex_idxs)
	for pos, ex_idx in enumerate(ex_idxs):
		all_preds[ex_idx] = preds[pos]
	assert out_file.endswith('.pkl')
	pkl.dump(all_preds, open(out_file, 'wb'))


if __name__ == '__main__':
	f_log = open('log.txt', 'w')
	timestamp = time.time()

	EX_IDXS = range(0, 500)

	# TaskQA
	for taskqa_model in ['gpt3', 'gpt4']:
		test_inputs = json.load(open('../data/data.json'))['test']
		for taskqa_expl_type in ['cot', 'posthoc']:
			out_file = f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}_test.pkl'
			run_task_save_results(task_function=task_qa, out_file=out_file, ex_idxs=EX_IDXS,
									model=taskqa_model, expl_type=taskqa_expl_type, inputs=test_inputs)
			f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type} {(time.time() - timestamp)//60} minutes\n')
			timestamp = time.time()

	# SimQG
	for taskqa_model in ['gpt3', 'gpt4']:
		for taskqa_expl_type in ['cot', 'posthoc']:
			for simqg_model in ['gpt3', 'gpt4']:
				for with_context in [True, False]:
					for top_p in [1.0]:
						out_file = f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}_test.pkl'
						orig_inputs = json.load(open('../data/data.json'))['test']
						orig_tm_preds = pkl.load(open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}.pkl', 'rb'))
						run_task_save_results(task_function=simulate_qg, ex_idxs=EX_IDXS, out_file=out_file,
												model=simqg_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
												top_p=top_p, num_samples=6, with_context=with_context)
						f_log.write(f'SimQG-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context} {(time.time() - timestamp)//60} minutes\n')
						timestamp = time.time()

	# mix GPT-3 and GPT-4 outputs
	for taskqa_model in ['gpt3', 'gpt4']:
		for taskqa_expl_type in ['cot', 'posthoc']:
			for with_context in [True, False]:
				for top_p in [1.0]:
					simqg_model2sim_inputs = {}
					for simqg_model in ['gpt3', 'gpt4']:
						simqg_model2sim_inputs[simqg_model] = pkl.load(
							open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}.pkl', 'rb'))
					out_file = f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_mix_{top_p}_{with_context}.pkl'
					if os.path.exists(out_file):
						ex_idx2mixed_sim_inputs = pkl.load(open(out_file, 'rb'))
					else:
						ex_idx2mixed_sim_inputs = {}
					for ex_idx in EX_IDXS:
						if ex_idx not in ex_idx2mixed_sim_inputs: # should not re-run for already computed mix ones because this process is random!
							ex_idx2mixed_sim_inputs[ex_idx] = mix_sim_inputs(simqg_model2sim_inputs['gpt3'][ex_idx],
																				simqg_model2sim_inputs['gpt4'][ex_idx],
																				sample_num=6)
					pkl.dump(ex_idx2mixed_sim_inputs, open(out_file, 'wb'))

	# SimQA
	for taskqa_model in ['gpt3', 'gpt4']:
		for taskqa_expl_type in ['cot', 'posthoc']:
			for simqg_model in ['mix']: # expl
				for with_context in [True]:
					for top_p in [1.0]:
						for simqa_model in ['gpt4', 'gpt3']:
							out_file = f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}-simqa_{simqa_model}_fix_test.pkl'
							orig_inputs = json.load(open('../data/data.json'))['test']
							orig_tm_preds = pkl.load(open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}.pkl', 'rb'))
							sim_inputs_list = pkl.load(open(
								f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}.pkl', 'rb'))
							run_task_save_results(task_function=simulate_qa, ex_idxs=EX_IDXS, out_file=out_file,
												model=simqa_model, orig_inputs=orig_inputs, orig_tm_preds=orig_tm_preds,
												sim_inputs_list=sim_inputs_list)
							f_log.write(f'SimQA-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context}-{simqa_model} {(time.time() - timestamp)//60} minutes\n')
						timestamp = time.time()

	# TaskQA on SimInputs
	for taskqa_model in ['gpt3', 'gpt4']:
		for taskqa_expl_type in ['cot', 'posthoc']:
			for simqg_model in ['mix']:
				for with_context in [True, False]:
					for top_p in [1.0]:
						out_file = f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}' \
									f'-taskqa_{taskqa_model}_{taskqa_expl_type}.pkl'
						sim_inputs_list = pkl.load(open(
							f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_{simqg_model}_{top_p}_{with_context}.pkl', 'rb'))
						run_task_save_results(task_function=task_qa_sim_inputs_list, ex_idxs=EX_IDXS, out_file=out_file,
												model=taskqa_model, expl_type=taskqa_expl_type, sim_inputs_list=sim_inputs_list)
						f_log.write(f'TaskQA-{taskqa_model}-{taskqa_expl_type}-{simqg_model}-{top_p}-{with_context} {(time.time() - timestamp)//60} minutes\n')
						timestamp = time.time()
