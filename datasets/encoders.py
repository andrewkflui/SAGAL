import os, random, re, pickle
from pathlib import Path

from abc import ABC, ABCMeta, abstractmethod

import nltk, spacy, ssl
# from nltk.corpus import stopwords
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

# from gensim.models import Word2Vec
from gensim import corpora
from gensim.models import LsiModel

import gluonnlp
import mxnet as mx

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

import numpy as np

from core import utils
from basis import config
from basis.logger import logger
from datasets.datasets import EncodedDataset

# nltk.data.path.append('{}/data/models/pretrained/nltk_data'.format(config.ROOT_PATH))
nltk.data.path.append(str(Path(config.ROOT_PATH) / 'data/models/pretrained/nltk_data'))

# download the package if punkt not exists
try:
	nltk.data.find('tokenizers/punkt')
except Exception as e:
	try:
		_create_unverified_https_context = ssl._create_unverified_context
	except AttributeError:
		pass
	else:
		ssl._create_default_https_context = _create_unverified_https_context
	
	nltk.download('punkt')
	nltk.download('stopwords')
	import nltk
finally:
	from nltk.corpus import stopwords

class TextEncoder(ABC):
	def __init__(self, name, model, min_value=None, max_value=None, file_name=None):
		self.name = name
		self.model = model if file_name is None else os.path.splitext(file_name)[0]
		# self.model_path = '{}/../data/models/pretrained/{}/{}'.format(os.path.abspath(os.path.dirname(__file__)), name, model)
		# self.model_path = '{}/data/models/pretrained/{}/{}'.format(config.ROOT_PATH, name, model)
		self.model_path = str(Path(config.ROOT_PATH) / 'data/models/pretrained' / name / model)
		if file_name is not None:
			self.model_path += '/{}'.format(file_name)
		self.min_value = min_value
		self.max_value = max_value

		self.stop_words = stopwords.words('english')
		self.stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

		# download en_core_web
		# self.nlp = spacy.load('en_core_web_sm', disable=['parser',' ner'])

	def tokenize(self, text, remove_stop_words):
		# download the package if punkt not exists
		# try:
		# 	nltk.data.find('tokenizers/punkt')
		# except Exception as e:
		# 	nltk.download('punkt')
		# 	nltk.download('stopwords')
		# 	import nltk

		# vector
		sentences = nltk.sent_tokenize(text)
		sentence_words = [nltk.word_tokenize(sentence) for sentence in sentences]

		if remove_stop_words:
			for i in range(len(sentence_words)):
				sentence_words[i] = [word for word in sentence_words[i] if word not in stopwords.words('english')]

		return sentence_words

	def lemmatization(self, words, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
		# nlp_text = self.nlp(' '.join(words))
		nlp_text = nlp(' '.join(words))
		# return [n.lemma_ for n in nlp_text if n.pos_ in allowed_postags]
		# return [n.lemma_ if n.pos_ in allowed_postags else '' for n in nlp_text]
		return [n.lemma_ if n.pos_ in allowed_postags else str(n) for n in nlp_text]

	def compare_encoding_in_dictionary(self, encoding, dictionary):
		if len(dictionary) > 0:
			values = np.array(list(dictionary.values()))
			logger.debug('TextEncoder.compare_encoding_in_dictionary, values.shape; {}'.format(values.shape))

			differences = values - np.array(encoding)
			# for value in dictionary.values():
			# 	if np.sum(abs(np.array(encoding) - np.array(value))) == 0.:
			# 		return True
			for difference in differences:
				if np.allclose(difference, np.zeros(shape=encoding.shape)):
					return True

		return False

	def get_encoding_from_oov(self, name, dimension, word, dictionary):
		# path = utils.create_directories('{}/../data/models/oov/{}/{}'.format(os.path.abspath(os.path.dirname(__file__)), name, dimension))
		path = utils.create_directories('{}/data/models/oov/{}/{}'.format(config.ROOT_PATH, name, dimension))
		file_name = 'oov.txt'

		oov_dict = utils.load(path / file_name) if os.path.exists(path / file_name) else dict()
		if word in oov_dict:
			return oov_dict[word]

		while True:
			encoding = np.array([random.random() * (self.max_value - self.min_value) - self.min_value for i in range(dimension)])
			exists = not self.compare_encoding_in_dictionary(encoding, oov_dict) and not self.compare_encoding_in_dictionary(encoding, dictionary)

			if not exists:
				break

		oov_dict[word] = encoding
		utils.save(oov_dict, path, file_name)

		return encoding

	@abstractmethod
	def encode(self, dataset, skip_encoded=False):
		raise NotImplementedException('encode function not implemented')

# class Gensim(TextEncoder):
# 	def __init__(self, dataset):
# 		super().__init__(dataset)
			
# 	def text_to_vector(self, answer_texts):
# 		# download the package if punkt not exists
# 		# nltk.download('punkt')
# 		# nltk.download('stopwords')

# 		vector
# 		for text in answer_texts:
# 			sentences = nltk.sent_tokenize(text)
# 			sentence_words = [nltk.word_tokenize(sentence) for sentence in sentences]

# 			for i in range(len(sentence_words)):
# 				sentence_words[i] = [word for word in sentence_words[i] if word not in stopwords.words('english')]

# 			# min_count specifies the minimum number of times a word should appear in the model
# 			# model = Word2Vec(text, min_count=1, size=100, window=5)

# 		model = Word2Vec(sentence_words, min_count=1)
# 		vocalbulary = model.wv.vocab
# 		print('vocalbulary: {}'.format(vocalbulary))

# 	def encode(self, question_id=None):
# 		question = self.dataset.get_question(question_id)
# 		if question is None:
# 			raise Exception('INVALID QUESTION ID {}'.format(question_id))

# 		answers = question.answers if question is not None else None
# 		vectors = self.text_to_vector([answer.text.lower() for answer in answers])

class SkipThoughts(TextEncoder):
	def __init__(self, model='skip_thoughts_uni_2017_02_02', checkpoint_name='model.ckpt-501424'):
		super().__init__(name='skip_thoughts', model=model, min_value=-1.0, max_value=1.0)
		# self.model = model
		self.checkpoint_name = checkpoint_name

		# self.model_path = '{}/../data/models/pretrained/skip_thoughts/{}'.format(os.path.abspath(os.path.dirname(__file__)), model)
		self.vocab_file = '{}/vocab.txt'.format(self.model_path)
		self.embedding_matrix_file = '{}/embeddings.npy'.format(self.model_path)
		self.checkpoint_path = '{}/{}'.format(self.model_path, self.checkpoint_name)

	def encode(self, dataset):
		from skip_thoughts import configuration, encoder_manager

		encoder = encoder_manager.EncoderManager()
		encoder.load_model(configuration.model_config(bidirectional_encoder=False),
			vocabulary_file=self.vocab_file,
			embedding_matrix_file=self.embedding_matrix_file,
			checkpoint_path=self.checkpoint_path
		)

		for question in dataset.questions:
			logger.debug('{}, encode, question id: {}'.format(self.name, question.id))
			# answers = sorted(question.answers, key=lambda x: x.is_reference, reverse=True)
			answers = question.answers
			# encodings = encoder.encode([answer.text for answer in answers])
			encodings = encoder.encode([answer.text for answer in answers if not answer.is_reference])

			for i in range(len(encodings)):
				answers[i].encoding = encodings[i]

		# self.save(data=encodings, output_path='datasets/encoded/{}/{}'.format(self.dataset.name, self.name))
		# self.dataset.save(output_path='datasets/encoded/{}'.format(self.dataset.name))
		return EncodedDataset(dataset=dataset, encoder=self)

class GoogleUniversalSentenceEncoder(TextEncoder):
	def __init__(self, model='4', multilingual=False):
		name = 'google_universal_sentence_encoder' + ('_multilingual' if multilingual else '')
		super().__init__(name=name, model=model, min_value=-1.0, max_value=1.0)

	
	def encode(self, dataset, skip_encoded=False):
		import tensorflow as tf
		import tensorflow_hub as hub
		import tensorflow_text

		embed = hub.load(self.model_path)
		
		for question in dataset.questions:
			logger.debug(f'{self.name}, encode, question id: {question.id}')
			# answers = sorted(question.answers, key=lambda x: x.is_reference, reverse=True)
			
			answers = question.answers if not skip_encoded else [a for a in question.answers if a.encodings is None]
			encodings = embed([answer.text.lower() for answer in answers])
			for i in range(len(encodings)):
				answers[i].encoding = encodings[i].numpy()

		return EncodedDataset(dataset=dataset, encoder=self)
	
	# def encode(self, dataset):
	# 	import tensorflow as tf
	# 	import tensorflow_hub as hub

	# 	embed = hub.load(self.model_path)

	# 	for question in dataset.questions:
	# 		logger.debug('{}, encode, question id: {}'.format(self.name, question.id))
	# 		# answers = sorted(question.answers, key=lambda x: x.is_reference, reverse=True)
			
	# 		answers = question.answers
	# 		encodings = embed([answer.text.lower() for answer in answers])
	# 		for i in range(len(encodings)):
	# 			answers[i].encoding = encodings[i].numpy()

	# 	return EncodedDataset(dataset=dataset, encoder=self)

# class Bert(TextEncoder):
# 	def __init__(self, model, dataset_name, oov_handler, remove_stop_words=False):
# 		super().__init__(name='bert{}'.format('_without_stop_words' if remove_stop_words else ''),
# 			model=model)
# 		self.dataset_name = dataset_name
# 		self.oov_handler = oov_handler
# 		self.remove_stop_words = remove_stop_words

# 	def encode(self, dataset):
# 		from bert_embedding import BertEmbedding
# 		# bert = BertEmbedding(model=self.model, dataset_name=self.dataset_name, ctx=mx.gpu(0))
# 		bert = BertEmbedding(model=self.model, dataset_name=self.dataset_name)

# 		for question in dataset.questions:
# 		# for question in [dataset.get_question('1'), dataset.get_question('8')]:
# 			print('question id: {}'.format(question.id))

# 			answers = question.answers
# 			if not self.remove_stop_words:
# 				embeddings = bert([answer.text for answer in answers])
# 			else:
# 				embeddings = bert([' '.join(self.tokenize(answer.text.lower(), self.remove_stop_words)[0]) for answer in answers])

# 			for i in range(len(embeddings)):
# 				embedding = embeddings[i]
# 				print('words: {}'.format(embedding[0]))
# 				print('embeddings: {}'.format(np.array(embedding[1]).shape))

# 				answers[i].words = embedding[0]
# 				answers[i].encoding = np.sum(embedding[1], axis=0)

# 		return EncodedDataset(dataset=dataset, encoder=self)

# class FastText(TextEncoder):
# 	def __init__(self, model='wiki-news-300d-1M', remove_stop_words=False):
# 		super().__init__(name='fasttext{}'.format('' if not remove_stop_words else '_without_stop_words'),
# 			model=model, min_value=-1., max_value=1.)
# 		self.remove_stop_words = remove_stop_words

# 	def train_model(self, raw_file_name='enwiki-latest-pages-articles', model='skipgram'):
# 		import fasttext

# 		# path = '{}/../data/models/pretrained/{}/{}'.format(os.path.abspath(os.path.dirname(__file__)), self.name, raw_file_name)
# 		path = '{}/data/models/pretrained/{}/{}'.format(config.ROOT_PATH, self.name, raw_file_name)
# 		source_path = '{}/{}.xml'.format(path, raw_file_name)

# 		output_path = utils.create_directories(path) / '{}.bin'.format(model)

# 		fasttext_model = fasttext.train_unsupervised(source_path, model=model)
# 		fasttext_model.save_model(output_path)

# 	def load_vectors(self):
# 		# cite vectors
# 		# @inproceedings{mikolov2018advances,
# 		# 	title={Advances in Pre-Training Distributed Word Representations},
# 		# 	author={Mikolov, Tomas and Grave, Edouard and Bojanowski, Piotr and Puhrsch, Christian and Joulin, Armand},
# 		# 	booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
# 		# 	year={2018}
# 		# }

# 		import io
# 		file = io.open('{}.vec'.format(self.model_path), 'r', encoding='utf-8', newline='\n', errors='ignore')
# 		number_of_data, dimension = map(int, file.readline().split())
# 		print('number_of_data: {}, dimension: {}'.format(number_of_data, dimension))
# 		vectors = dict()
# 		for line in file:
# 			tokens = line.rstrip().split(' ')
# 			vectors[tokens[0]] = list(map(float, tokens[1:]))
# 		return dimension, vectors


# 	def encode(self, dataset):
# 		dimension, vectors = self.load_vectors()

# 		# for question in dataset.questions:
# 		for question in [dataset.get_question('8')]:
# 			print('question id: {}'.format(question.id))

# 			answers = question.answers
# 			for answer in answers:
# 				if not self.remove_stop_words:
# 					text = self.tokenize(answer.text.lower(), self.remove_stop_words)[0]
# 				else:
# 					text = answer.text.lower().split(' ')

# 				answer.words = text
# 				encodings = []
# 				for word in text:
# 					encodings.append(vectors[word] if word in vectors else self.get_encoding_from_oov('fasttext', dimension, word, vectors))
# 				answer.encoding = np.sum(encodings, axis=0)
# 				print('encoding: {}'.format(answer.encoding))

# 		return EncodedDataset(dataset=dataset, encoder=self)

class WordEmbedding(TextEncoder):
	__meta__ = ABCMeta

	def __init__(self, name, ngram_range=(1, 3), min_frequency=0., lemmatize=True,
		remove_stop_words=True, retrain=False):
		super().__init__(name, model='{}_{}_min_df_{}'.format(ngram_range[0], ngram_range[1],
			int(min_frequency)))
		self.ngram_range = ngram_range
		self.min_frequency = min_frequency
		self.lemmatize = lemmatize
		self.remove_stop_words = remove_stop_words
		self.retrain = retrain

		self.feature_name_dict = dict()
		self.model_path_root = self.model_path.replace('pretrained', 'trained')
		self.vocabulary_frequency_dict_root = '{}/data/models/vocabulary/{}_{}'.format(config.ROOT_PATH,
			self.ngram_range[0], self.ngram_range[1])

	def text_to_words(self, text, nlp, lemmatize=True, remove_stop_words=False,
		reserve_stop_words_space=True):
		words = nltk.word_tokenize(text)
		lemmatized_text = text
		
		if lemmatize:
			words = self.lemmatization(words, nlp=nlp)
			lemmatized_text = ' '.join(words)

		if remove_stop_words:
			if not reserve_stop_words_space:
				words = [word for word in words if word not in self.stop_words and word != '']
			else:
				words = [word if word not in self.stop_words else '' for word in words]
		return lemmatized_text, words

	def clean_text(self, text_list):
		nlp = spacy.load('en_core_web_sm', disable=['parser',' ner'])
		lemmatized_text_list, word_list = [], []
		for text in text_list:
			text = re.sub('[,\.!?]', '', text).lower() 
			lemmatized_text, words = self.text_to_words(text, nlp, lemmatize=self.lemmatize,
				remove_stop_words=self.remove_stop_words)
			lemmatized_text_list.append(lemmatized_text)
			word_list.append(words)
		return lemmatized_text_list, word_list

	def plot_wordcloud(self, question, cleaned=True):
		from wordcloud import WordCloud
		import matplotlib.pyplot as plt
		text_list = [a.cleaned_text if cleaned else a.text for a in question.answers]
		string = ','.join(text_list)
		wordcloud = WordCloud(background_color='white', max_words=5000, contour_width=3,
			contour_color='steelblue')
		wordcloud.generate(string)
		
		plt.figure()
		plt.imshow(wordcloud, interpolation="bilinear")
		plt.axis("off")
		plt.show()

	def get_vocabulary_frequency_dict_path(self, dataset_name, question_id=None):
		# path = Path('{}/{}/{}'.format(self.vocabulary_frequency_dict_root, dataset_name, question_id))
		path = Path(os.path.join(self.vocabulary_frequency_dict_root, dataset_name))
		if question_id is not None:
			path /= question_id
		return path, 'vocabulary_frequency_dict.pickle'

	def save_vocabulary_frequency_dict(self, vocabulary_frequency_dict, dataset_name, question_id):
		path, file_name = self.get_vocabulary_frequency_dict_path(dataset_name, question_id)
		utils.save(vocabulary_frequency_dict, str(path), file_name)

	def load_vocabulary_frequency_dict(self, dataset_name, question_id):
		path, file_name = self.get_vocabulary_frequency_dict_path(dataset_name, question_id)
		path = path / file_name
		if self.retrain or not os.path.exists(path):
			return dict()
		return utils.load(path)

	def extend_vocabulary(self, word_list, vocabulary_frequency_dict):
		train_data, padded_word_list = padded_everygram_pipeline(self.ngram_range[1], word_list)
		for line in train_data:
			for t in line:
				if '' in t:
					continue
				w = ' '.join(list(t)).strip()
				if w.replace('<s>', '').replace('</s>', '').strip() != '':
					if w not in vocabulary_frequency_dict:
						vocabulary_frequency_dict[w] = 0
					vocabulary_frequency_dict[w] +=1
		return vocabulary_frequency_dict

	def encode(self, dataset, skip_encoded=False):
		for question in dataset.questions:
			logger.debug('{}, encode, question id: {}'.format(self.name, question.id))
			
			answers = question.answers if not skip_encoded else [a for a in question.answers if a.encodings is not None]
			# lemmatized_text_list, word_list = self.clean_text([a.text for a in question.answers])
			lemmatized_text_list, word_list = self.clean_text([a.text for a in answers])
			vocabulary_frequency_dict = self.load_vocabulary_frequency_dict(dataset.name, question.id)

			if self.retrain or len(vocabulary_frequency_dict) <= 0:
				vocabulary_frequency_dict = self.extend_vocabulary(word_list, vocabulary_frequency_dict)
				self.save_vocabulary_frequency_dict(vocabulary_frequency_dict, dataset.name, question.id)
			
			feature_names, vectors = self.get_vectors(lemmatized_text_list, word_list,
				vocabulary_frequency_dict)
			self.feature_name_dict[question.id] = feature_names
			if vocabulary_frequency_dict is not None:
				logger.debug('Vocabulary Count: {}, Feature Count: {}'.format(len(vocabulary_frequency_dict),
					len(vectors[0])))
			
			for i in range(len(vectors)):
				answers[i].lemmatized_text = lemmatized_text_list[i]
				answers[i].words = word_list[i] 
				answers[i].encoding = vectors[i]

				# print('text: ', question.answers[i].text)
				# for j in range(len(vectors[i])):
				# 	if vectors[i][j] > 0:
				# 		print('j: {}, feature_name: {}, {}'.format(j ,feature_names[j], vectors[i][j]))
				# print('')
		return EncodedDataset(dataset=dataset, encoder=self)

	# def encode(self, dataset. skip_encoded=False):
	# 	for question in dataset.questions:
	# 		logger.debug('{}, encode, question id: {}'.format(self.name, question.id))
	
	# 		lemmatized_text_list, word_list = self.clean_text([a.text for a in question.answers])
	# 		vocabulary_frequency_dict = self.load_vocabulary_frequency_dict(dataset.name, question.id)

	# 		if self.retrain or len(vocabulary_frequency_dict) <= 0:
	# 			vocabulary_frequency_dict = self.extend_vocabulary(word_list, vocabulary_frequency_dict)
	# 			self.save_vocabulary_frequency_dict(vocabulary_frequency_dict, dataset.name, question.id)
			
	# 		feature_names, vectors = self.get_vectors(lemmatized_text_list, word_list,
	# 			vocabulary_frequency_dict)
	# 		# lemmatized_text_list, word_list, vocabulary_frequency_dict, feature_names, vectors = self.encode_data(
	# 		# 	[a.text for a in question.answers], dataset.name, question.id)
	# 		self.feature_name_dict[question.id] = feature_names
	# 		if vocabulary_frequency_dict is not None:
	# 			logger.debug('Vocabulary Count: {}, Feature Count: {}'.format(len(vocabulary_frequency_dict),
	# 				len(vectors[0])))
			
	# 		for i in range(len(vectors)):
	# 			question.answers[i].lemmatized_text = lemmatized_text_list[i]
	# 			question.answers[i].words = word_list[i] 
	# 			question.answers[i].encoding = vectors[i]

	# 			# print('text: ', question.answers[i].text)
	# 			# for j in range(len(vectors[i])):
	# 			# 	if vectors[i][j] > 0:
	# 			# 		print('j: {}, feature_name: {}, {}'.format(j ,feature_names[j], vectors[i][j]))
	# 			# print('')
	# 	return EncodedDataset(dataset=dataset, encoder=self)

	@abstractmethod
	def get_vectors(self, text_list, word_list, vocabulary_frequency_dict):
		raise NotImplementedException('get_vectors not implemented')

class Count(WordEmbedding):
	def __init__(self, ngram_range=(1, 3), min_frequency=0., retrain=False):
		super().__init__(name='count', ngram_range=ngram_range, min_frequency=min_frequency, retrain=retrain)

	def get_vectors(self, lemmatized_text_list, word_list, vocabulary_frequency_dict):
		vocabulary = [k for k, v in vocabulary_frequency_dict.items() if v >= self.min_frequency]
		model = CountVectorizer(analyzer='word', vocabulary=vocabulary, ngram_range=self.ngram_range,
			stop_words=self.stop_words, min_df=self.min_frequency)
		return model.get_feature_names(), np.array(model.fit_transform(lemmatized_text_list).toarray())

class TFIDF(WordEmbedding):
	def __init__(self, ngram_range=(1, 3), min_frequency=0., retrain=False):
		super().__init__(name='tfidf', ngram_range=ngram_range, min_frequency=min_frequency, retrain=retrain)

	def get_vectors(self, lemmatized_text_list, word_list, vocabulary_frequency_dict):
		vocabulary = [k for k, v in vocabulary_frequency_dict.items() if v >= self.min_frequency]
		model = TfidfVectorizer(vocabulary=vocabulary, ngram_range=self.ngram_range,
			stop_words=self.stop_words, min_df=self.min_frequency, use_idf=True)
		return model.get_feature_names(), np.array(model.fit_transform(lemmatized_text_list).toarray())

class JaccardSimilarity(WordEmbedding):
	def __init__(self, lemmatize=True, remove_stop_words=True):
		super().__init__(name='jaccard_similarity', lemmatize=lemmatize,
			remove_stop_words=remove_stop_words, retrain=False)
		self.model = '{}_{}'.format('l' if lemmatize else 'nl', 'r' if remove_stop_words else 'nr')
	# def __init__(self, ngram_range=(1, 1), min_frequency=0., retrain=False):
	# 	super().__init__(name='jaccard_similarity', ngram_range=ngram_range, min_frequency=min_frequency, retrain=retrain)

	def save_vocabulary_frequency_dict(self, vocabulary_frequency_dict, dataset_name, question_id):
		pass

	def load_vocabulary_frequency_dict(self, dataset_name, question_id):
		return None

	def extend_vocabulary(self, word_list, vocabulary_frequency_dict):
		pass

	def get_vectors(self, lemmatized_text_list, word_list, vocabulary_frequency_dict):
		return None, [[w for w in l if w != ''] for l in word_list]

class LSA(WordEmbedding):
	def __init__(self, ngram_range=(1, 3), min_frequency=0., n_components=100, retrain=False):
		super().__init__(name='lsa', ngram_range=ngram_range, min_frequency=min_frequency, retrain=retrain)
		self.n_components = n_components

	def get_vectors(self, lemmatized_text_list, word_list, vocabulary_frequency_dict):
		vocabulary = [k for k, v in vocabulary_frequency_dict.items() if v >= self.min_frequency]
		# id2word = gensim.corpora.Dictionary(vocabulary)
		# corpus = [id2word.doc2bow(v) for v in lemmatized_text_list]
		
		tfidf = TfidfVectorizer(vocabulary=vocabulary, ngram_range=self.ngram_range,
			stop_words=self.stop_words, min_df=self.min_frequency, use_idf=True, smooth_idf=True)
		n_components = min(len(vocabulary) - 1, self.n_components)
		# svd = TruncatedSVD(n_components=n_components, algorithm='arpack', tol=1)
		svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=0)
		# fit svd twice to resolve "sign indeterminacy" problem'
		model = Pipeline([('tfidf', tfidf), ('svd', svd)])
		model.fit(lemmatized_text_list)
		vectors = model.transform(lemmatized_text_list)
		return None, vectors

class PretrainedWordEmbedding(WordEmbedding):
	__meta__ = ABCMeta

	def __init__(self, name, source_model, ngram_range=(1, 3), min_frequency=0., lemmatize=True,
		remove_stop_words=True):
		super().__init__(name, ngram_range=ngram_range, min_frequency=min_frequency, lemmatize=lemmatize,
			remove_stop_words=remove_stop_words, retrain=False)
		self.source_model = source_model
		self.source_model_path = str(Path(config.ROOT_PATH) / 'data/models/pretrained')
		self.model = '{}/{}'.format(self.source_model, self.model)

class Glove(PretrainedWordEmbedding):
	def __init__(self, source_model, ngram_range=(1, 3), min_frequency=0., lemmatize=True,
		remove_stop_words=True):
		super().__init__(name='glove', source_model=source_model, ngram_range=ngram_range,
			min_frequency=min_frequency, lemmatize=lemmatize, remove_stop_words=remove_stop_words)

	def get_vectors(self, lemmatized_text_list, word_list, vocabulary_frequency_dict):
		count_dict = {k: v for k, v in vocabulary_frequency_dict.items() if v >= self.min_frequency}
		# glove = gluonnlp.embedding.create('GloVe', source=self.source_model)
		glove = gluonnlp.embedding.create('GloVe', source=self.source_model,
			embedding_root=self.source_model_path)
		vocab = gluonnlp.Vocab(count_dict)
		vocab.set_embedding(glove)
		vectors = []
		for text in lemmatized_text_list:
			words = text.split(' ')
			vector = mx.ndarray.mean(vocab.embedding[words], 0).asnumpy()
			vectors.append(vector)
		vectors = np.array(vectors)
		return None, vectors

class Bert(PretrainedWordEmbedding):
	def __init__(self, source_model, dataset_name, ngram_range=(1, 3), min_frequency=0., lemmatize=True,
		remove_stop_words=True, oov_handler='avg'):
		super().__init__(name='bert{}'.format('_no_stop_word_removal' if not remove_stop_words else ''),
			source_model=source_model, ngram_range=ngram_range, min_frequency=min_frequency,
			lemmatize=lemmatize, remove_stop_words=remove_stop_words)
		self.dataset_name = dataset_name
		self.oov_handler = oov_handler
		self.model = '{}/{}'.format(self.source_model, self.dataset_name)

	def get_vectors(self, lemmatized_text_list, word_list, vocabulary_frequency_dict):
		bert, vocabulary = gluonnlp.model.get_model(self.source_model,
			dataset_name=self.dataset_name, pretrained=True, use_pooler=False, use_decoder=False,
			use_classifier=False)
		tokenizer = gluonnlp.data.BERTTokenizer(vocabulary, lower=True)
		transform = gluonnlp.data.BERTSentenceTransform(tokenizer,
			max_seq_length=1024, pair=False, pad=False)

		vectors = []
		for lemmatized_text in lemmatized_text_list:
			token_ids, valid_length, token_types = transform([lemmatized_text])
			encoding = bert(mx.nd.array([token_ids]), mx.nd.array([token_types]),
				mx.nd.array([valid_length]))[0].asnumpy()

			if self.oov_handler == 'avg':
				encoding = np.mean(encoding, 0)
			elif eslf.oov_handler == 'sum':
				encoding = np.sum(encoding, 0)
			vectors.append(encoding)
		vectors = np.array(vectors)
		logger.debug('{}, get_vectors, vectors.shape: {}'.format(self.name, vectors.shape))
		
		# for words in word_list:
		# 	print('words', words)
		# 	words = [w for w in words if w != '']
		# 	print('words', words)
		# 	token_ids = self.vocabulary[words]
		# 	print('token_ids', token_ids)
		# 	encoding = self.bert(mx.nd.array([token_ids]), mx.nd.array([len(token_ids)])).asnumpy()
		# 	print('encoding.shape', encoding.shape)
		# 	print('encoding', encoding)
		
		return None, vectors

