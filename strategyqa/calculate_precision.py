import sys
from collections import Counter
from copy import deepcopy
import numpy as np
import pickle as pkl
from scipy.stats import ttest_ind, ttest_rel
sys.path.append('../../amt')


if __name__ == '__main__':
	setting2exidx2qns_aggann = pkl.load(open('../amt/expl-eval-aggann-good-turkers.pkl', 'rb'))

	setting2exidx2precision = {}
	for taskqa_model in ['gpt3', 'gpt4']:
		for taskqa_expl_type in ['cot', 'posthoc']:
			setting2exidx2precision[(taskqa_model, taskqa_expl_type)] = {}
			exidx2qns_aggann = setting2exidx2qns_aggann[(taskqa_model, taskqa_expl_type, 'mix')]
			taskqa_preds = pkl.load(
				open(f'../outputs/taskqa_{taskqa_model}_{taskqa_expl_type}-simqg_mix_1.0-taskqa_{taskqa_model}_{taskqa_expl_type}.pkl', 'rb'))
			for exidx in exidx2qns_aggann:
				assert len(taskqa_preds[exidx]) == len(exidx2qns_aggann[exidx])
				ex_simulatable_count, ex_correct_simul_count = 0, 0
				for qnidx in range(len(taskqa_preds[exidx])):
					simqa_ann = exidx2qns_aggann[exidx][qnidx]
					taskqa_pred = taskqa_preds[exidx][qnidx]['pred_ans']
					if simqa_ann in ['yes', 'no']:
						ex_simulatable_count += 1
						if simqa_ann == taskqa_pred:
							ex_correct_simul_count += 1
				if ex_simulatable_count != 0:
					setting2exidx2precision[(taskqa_model, taskqa_expl_type)][exidx] = ex_correct_simul_count / ex_simulatable_count
	pkl.dump(setting2exidx2precision, open('../outputs/setting2exidx2precision.pkl', 'wb'))

	all_settings_exidxs = [setting2exidx2precision[setting].keys() for setting in setting2exidx2precision]
	exidxs_in_all_settings = [exidx for exidx in all_settings_exidxs[0] if np.all([exidx in exidxs for exidxs in all_settings_exidxs])]
	print(exidxs_in_all_settings)
	print(len(exidxs_in_all_settings))
	setting2scores = {setting: [setting2exidx2precision[setting][exidx] for exidx in exidxs_in_all_settings]
					  for setting in setting2exidx2precision}
	for setting in setting2scores:
		print(' '.join(setting), round(np.mean(setting2scores[setting]) * 100, 1))

	# calculate p-values
	settings = list(setting2scores.keys())
	print(settings)
	for setting1 in settings:
		pvalues = [str(ttest_rel(setting2scores[setting1], setting2scores[setting2])[1]) for setting2 in settings]
		print(','.join(pvalues))

	gpt3_scores = np.mean([setting2scores[('gpt3', 'cot')], setting2scores[('gpt3', 'posthoc')]], axis=0)
	gpt4_scores = np.mean([setting2scores[('gpt4', 'cot')], setting2scores[('gpt4', 'posthoc')]], axis=0)
	print(ttest_rel(gpt3_scores, gpt4_scores, alternative='less'))
