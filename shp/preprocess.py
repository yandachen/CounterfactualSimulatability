import numpy as np
from datasets import load_dataset
import json
from collections import defaultdict
import random
import re
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer


def clean(text):
	return re.sub(' +', ' ', text.strip()).strip().\
		replace(' , ',',').replace(' .','.').replace(' !','!')\
		.replace(' ?','?').replace(' : ',': ').replace(' \'', '\'')


if __name__ == '__main__':
	dataset = load_dataset("stanfordnlp/shp")
	split2data = {'train': dataset['train'], 'test': dataset['test']}
	tokenizer = AutoTokenizer.from_pretrained('gpt2')
	split2selecteddata = {}
	for split in ['train', 'test']:
		context2examples = defaultdict(list)
		for ex in tqdm(split2data[split]):
			context2examples[ex['history']].append(ex)
		# for each context, choose the selected response to be the response with highest score
		# extract all inferior responses to that selected response in the dataset (this guarantee timestamp ordering)
		# sort by highest vote to lowest vote, choose the negative unselected response randomly from the second half
		data = []
		for context in tqdm(context2examples):
			# figure out the top-score answer
			examples = context2examples[context]
			preferred_answer_scores = [max(ex['score_A'], ex['score_B']) for ex in examples]
			highest_score_examples = [examples[idx] for idx in range(len(examples))
									  if preferred_answer_scores[idx] == max(preferred_answer_scores)]
			lowest_scores = [min(ex['score_A'], ex['score_B']) for ex in highest_score_examples]
			assert len(highest_score_examples) == len(lowest_scores)
			median_lowest_score = np.median(lowest_scores)
			select_from_examples = [highest_score_examples[idx] for idx in range(len(highest_score_examples))
									if lowest_scores[idx] < median_lowest_score]

			if len(select_from_examples) > 0:
				selected_ex = random.sample(select_from_examples, 1)[0]
				if len(tokenizer.tokenize(clean(selected_ex['history']))) + len(
						tokenizer.tokenize(clean(selected_ex['human_ref_A']))) + len(
						tokenizer.tokenize(clean(selected_ex['human_ref_B']))) <= 1024:
					data.append({'context': clean(selected_ex['history']),
								 'options': [clean(selected_ex['human_ref_A']), clean(selected_ex['human_ref_B'])],
								 'preferred_idx': 1 - selected_ex['labels'],
								 'domain': clean(selected_ex['domain']),
								 'scores': [selected_ex['score_A'], selected_ex['score_B']],
								 'post_id': selected_ex['post_id']},
								)
		split2selecteddata[split] = data

	json.dump(split2selecteddata, open('data/data.json', 'w'), indent=4)
