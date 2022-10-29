from enum import Enum

VERSION = 'DPGrading1'

DATASETS = {
	# https://www.kaggle.com/smiles28/semeval-2013-2-and-3-way
	'SEB2': {
		'source_folder': 'semeval2013-Task7-2and3way',
		'processed_folder': 'SEB2'
	},
	'SEB3': {
		'source_folder': 'semeval2013-Task7-2and3way',
		'processed_folder': 'SEB3'
	},
	# https://www.kaggle.com/smiles28/semeval-2013-5-way
	'SEB5': {
		'source_folder': 'semeval2013-Task7-5way',
		'processed_folder': 'SEB5'
	},
	'USCIS': {
		'source_folder': 'Powergrading-1.0-Corpus',
		'processed_folder': 'USCIS'
	},
	'USCIS_with_100': {
		'source_folder': 'Powergrading-1.0-Corpus',
		'processed_folder': 'USCIS_include_100'
	},
	'Mobley': {
		'source_folder': 'ShortAnswerGrading_v2.0',
		'processed_folder': 'Mobley'
	},
	'CAK': {
		'source_folder': 'chakrabortyAndKonar',
		'processed_folder': 'cak'

	},
	'ASAP': {
		'source_folder': 'asap-aes',
		'processed_folder': 'ASAP'
	}
}

ENCODERS = {
	'skip_thoughts': {
		'default_model': 'skip_thoughts_uni_2017_02_02',
		'default_checkpoint_name': 'model.ckpt-501424'
	},
	'google_universal_sentence_encoder': {
		'default_model': '4'
	},
	'glove': {
		# 'default_model': 'glove.twitter.27B.200d',
		# 'default_model': 'glove.840B.300d'
		'default_model': 'glove.840B.300d/1_3_min_df_0'
	},
	'bert': {
		# 'default_model': 'bert_24_1024_16',
		# 'default_model': 'bert_24_1024_16/1_3_min_df_0',
		# 'default_dataset_name': 'book_corpus_wiki_en_cased'
		'default_model': 'bert_24_1024_16/book_corpus_wiki_en_cased'
	},
	'fasttext': {
		'default_model': 'wiki-news-300d-1M'
	},
	'tfidf': {
		'default_model': '1_3_min_df_2'
	},
	'count': {
		'default_model': '1_3_min_df_2'
	},
	'lsa': {
		'default_model': '1_3_min_df_2'
	},
	'jaccard_similarity': {
		'default_model': 'nl_nr'
	}
}

ALGORITHMS = ['gal', 'dbscan', 'birch', 'kmeans', 'gaussian_mixture',
	'spectral_clustering', 'hdbscan', 'affinity_propagation', 'optics']

VALIDATION_INDICES = [
	'WB', 'VC'
]

NON_APPLICABLE_CLUSTER = -1
UNCLASSIFIED_CLUSTER = 0

class ObjectiveType(Enum):
	SCORE = 1
	CLUSTER_NUM = 2

class ConvergeDirection(Enum):
	MINIMIZE = -1
	MAXIMIZE = 1
