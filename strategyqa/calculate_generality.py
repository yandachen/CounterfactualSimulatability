import pickle as pkl
import sys
sys.path.append('../../utils')
from diversity_util import calculate_diversity
import numpy as np
import random
from scipy.stats import ttest_ind, ttest_rel

setting2exidx2qns_aggann = pkl.load(open('../amt/expl-eval-aggann-good-turkers.pkl', 'rb'))
metrics = ['bleu', 'cosine', 'jaccard']
ex_idxs = range(200)

# load simulatable inputs
setting2exidx2simulatableinputs = {}
simqg_model = 'mix'
for taskqa_model in ['gpt3', 'gpt4']:
	for taskqa_expl_type in ['cot', 'posthoc']:
			setting = (taskqa_model, taskqa_expl_type)
			setting2exidx2simulatableinputs[setting] = {}
			sim_inputs = pkl.load(open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}'
									   f'-simqg_{simqg_model}_1.0.pkl', 'rb'))
			for ex_idx in ex_idxs:
				ex_sim_inputs = sim_inputs[ex_idx]
				ex_sim_ans = setting2exidx2qns_aggann[(taskqa_model, taskqa_expl_type, simqg_model)][ex_idx]
				assert len(ex_sim_inputs) == len(ex_sim_ans)
				simulatable_inputs = [ex_sim_inputs[idx]['question'] for idx in range(len(ex_sim_inputs))
									  if ex_sim_ans[idx] != 'unknown']
				setting2exidx2simulatableinputs[setting][ex_idx] = simulatable_inputs

# calculate diversity
setting2divs = {}
for taskqa_model in ['gpt3', 'gpt4']:
	for taskqa_expl_type in ['cot', 'posthoc']:
		for simqg_model in ['gpt3', 'gpt4', 'gpt-mix', 'pj']:
			divs = []
			for ex_idx in ex_idxs:
				setting = (taskqa_model, taskqa_expl_type)
				divs.append(calculate_diversity(setting2exidx2simulatableinputs[setting][ex_idx]))
			setting2divs[(taskqa_model, taskqa_expl_type)] = np.array(divs)

for div_metric in range(3):
	print(metrics[div_metric])
	setting2scores = {setting: setting2divs[setting][:, div_metric].tolist() for setting in setting2divs}
	setting2exidxs_nonempty = {setting: [ex_idx for ex_idx in range(len(setting2scores[setting]))
										if not np.isnan(setting2scores[setting][ex_idx])]
							   for setting in setting2scores}
	setting_exidxs_nonempty = [setting2exidxs_nonempty[setting] for setting in setting2exidxs_nonempty]
	nonempty_exidxs_for_all_simqg = [ex_idx for ex_idx in setting_exidxs_nonempty[0]
									 if np.all(
			[ex_idx in exidxs_nonemtpy for exidxs_nonemtpy in setting_exidxs_nonempty[1:]])]
	print(len(nonempty_exidxs_for_all_simqg))
	setting2scores = {setting: [setting2scores[setting][ex_idx] for ex_idx in nonempty_exidxs_for_all_simqg]
					  for setting in setting2scores}
	setting2score = {setting: np.mean(setting2scores[setting]) for setting in setting2scores}
	settings = [(taskqa_model, taskqa_expl_type) for taskqa_model in ['gpt3', 'gpt4']
				for taskqa_expl_type in ['cot', 'posthoc']]
	print(settings)
	print(','.join([str(round(setting2score[setting], 3)) for setting in settings]))
	for setting1 in settings:
		for setting2 in settings:
			print(setting1, setting2, ttest_rel(setting2scores[setting1], setting2scores[setting2])[1])

