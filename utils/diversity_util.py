import numpy as np
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer
import torch
import random
import string
import re

from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')

def calculate_pairwise_bleu(texts, texts2=None):
	if len(texts) <= 1 or (texts2 is not None and len(texts2) <= 1):
		return np.nan
	bleu = BLEU(effective_order=True)
	if texts2 is None:
		return np.mean([bleu.sentence_score(texts[idx1], [texts[idx2]]).score / 100
						for idx1 in range(len(texts)) for idx2 in range(len(texts)) if idx1 != idx2])
	else:
		return np.mean([bleu.sentence_score(texts[idx1], [texts2[idx2]]).score / 100
						for idx1 in range(len(texts)) for idx2 in range(len(texts2))])


def calculate_pairwise_cosine(texts, texts2=None):
	if len(texts) <= 1 or (texts2 is not None and len(texts2) <= 1):
		return np.nan
	if texts2 is None:
		encoding = torch.tensor(model.encode(texts, batch_size=256, show_progress_bar=False))
		scores = torch.nn.functional.cosine_similarity(encoding[:,:,None], encoding.t()[None,:,:]) # calculate pairwise cosine similarity
		assert scores.shape == (len(texts), len(texts))
		# fill the diagonal with nans
		scores.fill_diagonal_(torch.nan)
		return torch.nanmean(scores).item()
	else:
		encoding = torch.tensor(model.encode(texts, batch_size=256, show_progress_bar=False))
		encoding2 = torch.tensor(model.encode(texts2, batch_size=256, show_progress_bar=False))
		scores = torch.nn.functional.cosine_similarity(encoding[:, :, None],
													   encoding2.t()[None, :, :])  # calculate pairwise cosine similarity
		assert scores.shape == (len(texts), len(texts2))
		return torch.mean(scores).item()


def calculate_pairwise_bow_jaccard(texts, stopwords, texts2=None):
	def tokenize_bow(text, stopwords):
		text = re.sub(' +', ' ', text.translate(str.maketrans('', '', string.punctuation)).strip())
		text_tokens = text.split(' ')
		# remove stopwords
		nonstopword_tokens = [token for token in text_tokens if token not in stopwords]
		return set(nonstopword_tokens)

	if len(texts) <= 1 or (texts2 is not None and len(texts2) <= 1):
		return np.nan

	texts_bow = [tokenize_bow(text, stopwords=stopwords) for text in texts]
	if texts2 is None:
		return np.mean([len(texts_bow[idx1].intersection(texts_bow[idx2])) / len(texts_bow[idx1].union(texts_bow[idx2]))
						for idx1 in range(len(texts_bow)) for idx2 in range(len(texts_bow)) if idx1 != idx2])
	else:
		texts2_bow = [tokenize_bow(text, stopwords=stopwords) for text in texts2]
		return np.mean([len(texts_bow[idx1].intersection(texts2_bow[idx2])) / len(texts_bow[idx1].union(texts2_bow[idx2]))
						for idx1 in range(len(texts_bow)) for idx2 in range(len(texts2_bow))])

def count_distinct_ngrams(texts, n):
	# remove punctuations and then remove consecutive spaces
	texts = [re.sub(' +', ' ', text.translate(str.maketrans('', '', string.punctuation)).strip()) for text in texts]
	ngrams = set()
	for text in texts:
		tokens = text.split(' ')
		if len(tokens) >= n:
			for idx in range(len(tokens) - n + 1):
				ngram = tuple(tokens[idx: idx + n])
				assert len(ngram) == n
				ngrams.add(ngram)
	return len(ngrams)


def calculate_diversity(texts, texts2=None):
	return 1 - np.array([calculate_pairwise_bleu(texts, texts2=texts2),
						 calculate_pairwise_cosine(texts, texts2=texts2),
						 calculate_pairwise_bow_jaccard(texts, stopwords=en_stopwords, texts2=texts2)])

