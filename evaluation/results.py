import os, csv, random, copy, datetime
from pathlib import Path

import numpy as np

from abc import ABC, abstractmethod

from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from core import utils
from basis import config
from basis.constants import VERSION, ENCODERS, DATASETS, UNCLASSIFIED_CLUSTER
from datasets.datasets import LABELS_2WAY

def load_encoded_dataset(name, encoder, model, mapping):
	source_path = encoder
	model = model if model is not None else ENCODERS[encoder]['default_model']
	source_path += '/{}/{}'.format(model, DATASETS[name]['processed_folder'])
	source_path = Path('data/datasets/encoded') / source_path
	source_path = Path(config.ROOT_PATH) / source_path
	# encoded_dataset = utils.load(source_path / 'dataset.txt')
	if os.path.exists(source_path / 'dataset.zip'):
		encoded_dataset = utils.load_zip(source_path / 'dataset.zip', 'dataset.txt')
	else:
		encoded_dataset = utils.load(source_path / 'dataset.txt')
	return encoded_dataset, source_path

def load_dataset(name, question_id, encoder, model=None, mapping=None):
	encoded_dataset, source_path = load_encoded_dataset(name, encoder, model, mapping)

	if mapping is not None:
		labels_path = source_path / question_id / mapping
	else:
		labels_path = source_path / question_id

	rows = utils.read_csv_file(labels_path / 'labels.tsv', delimiter='\t')
	labels, values = [], []
	for row in rows:
		values.append(row[0])
		labels.append(row[1])

	reference_answer_labels, reference_answer_values = [], []
	rows = utils.read_csv_file(labels_path / 'reference_answer_labels.tsv', delimiter='\t')
	for row in rows:
		reference_answer_values.append(row[0])
		reference_answer_labels.append(row[1])
	return encoded_dataset.get_subset(question_id=question_id), labels, values, reference_answer_labels,\
		reference_answer_values

def handle_result(name, question_id, encoder, model, compress_factor, compress_method, mapping,
	algorithm, result, time_used, save_mode=1, plot=None, folder_name=None):
	dataset, labels, text_list, reference_answer_labels, reference_answers_text_iist = load_dataset(name, question_id, encoder, model, mapping=mapping)
	dataset.set_data(algorithm.data)
	
	result_set = ResultSet(dataset, algorithm, time_used=time_used)

	result_set.handle_result(solutions=result, labels=labels)
	folder_name = print_result_lines(result_set, algorithm, labels, save_mode, folder_name=folder_name)

	if plot is not None:
		plot_graphs(algorithm, result_set, labels, plot)

	return result_set, folder_name

def print_result_lines(result_set, algorithm, labels, save_mode, folder_name=None, path=None,
	file_name_suffix=None):
	lines = []
	lines.append(str(result_set))

	if result_set.dataset.name == 'Dummy':
		regions = result_set.dataset.regions
		for i in range(len(regions)):
			lines.append('Region #{}: {} => {}'.format(i+1, regions[i]['bounds'], regions[i]['label']))
		lines.append('Variance: {}'.format(result_set.dataset.variance))
	lines.append('')

	common_lines = copy.copy(lines)

	for result in result_set.selected_results.values():
		lines.append(str(result))

	if save_mode == 0:
		list(map(print, lines))
		return None
	else:
		if folder_name is not None:
			time = folder_name
		else:
			time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
		
		file_name = algorithm.name

		# if algorithm.name == 'gal':
		# 	file_name += '_m_{}_rdg_{}_ga'.format(result_set.results[0].solution.problem.density_method,
		# 		result_set.results[0].solution.problem.radius_grade,
		# 		result_set.results[0].solution.problem.grading_actions)

		if path is None:
			if result_set.dataset.name == 'Dummy':
				path = '{}/data/results/{}/{}/{}/{}'.format(config.ROOT_PATH, result_set.dataset.name,
					result_set.dataset.question.id, result_set.algorithm, time)
			elif algorithm.name != 'gal':
				path = '{}/data/results/{}/{}/{}/{}/{}'.format(config.ROOT_PATH, result_set.dataset.name,
					result_set.dataset.question.id, result_set.dataset.encoder.name,
					result_set.algorithm_name, time)
			else:
				path = '{}/data/results/{}/{}/multiple/{}/{}'.format(config.ROOT_PATH, result_set.dataset.name,
					result_set.dataset.question.id, result_set.algorithm_name, time)

		if algorithm.name == 'dbscan' or algorithm.name == 'hdbscan':
			if algorithm.name == 'dbscan':
				file_name += '_eps_{}'.format(algorithm.eps)
				file_name += '_metric_{}',format(algorithm.metric_name)
			else:
				file_name += '_metric_{}'.format(algorithm.clusterer.metric_name)

			if algorithm.count_unclassified:
				file_name += '_count_unclassified'
		
		elif (algorithm.name == 'birch' or algorithm.name == 'kmeans') \
			and algorithm.n_clusters is not None:
			file_name += '_{}'.format(algorithm.n_clusters)

		if config.RANDOM_SEED is not None:
			file_name += '_seed_{}'.format(config.RANDOM_SEED)

		if file_name_suffix is not None:
			file_name += file_name_suffix

		utils.write(path, '{}.txt'.format(file_name), lines)

		if save_mode == 2:
			data = {'result_set': result_set, 'algorithm': algorithm}
			utils.save(data, path, '{}.dat'.format(file_name))

		if algorithm.name == 'dbscan':
			rows = []
			rows.append([
				'Question ID', 'Encoder', 'Dimension', 'TP', 'TN', 'FP', 'FN',
				'Precision', 'Recall', 'Cluster Number', 'Accuracy', 'F1-Score', 'Unclassified Percentage'
			])

			for _, result in result_set.selected_results.items():
				rows.append([
					result_set.dataset.question.id, result_set.dataset.encoder.name,
					result_set.dataset.data_dim,
					result.tp, result.tn, result.fp, result.fn,
					result.precision, result.recall, result.solution.count_cluster,
					result.accuracy, result.f1_score,
					'{}%'.format(result.solution.unclassified_num * 100 / result_set.dataset.num_data)
				])
			
			utils.write_csv_file(path, '{}.csv'.format(file_name), rows)
		elif algorithm.name == 'gal':
			rows = []
			# rows.append([
			# 	'Question ID', 'Encoder', 'Dimension', 'Density Method', 'Radius Grade',
			# 	'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall', 'Clusters', 'Accuracy', 'F1-Score'
			# ])

			# for _, result in result_set.selected_results.items():
			# 	rows.append([
			# 		result_set.dataset.question.id, result_set.dataset.encoder.name,
			# 		result_set.dataset.data_dim, result.solution.problem.density_method,
			# 		result.solution.problem.radius_grade,result.tp, result.tn, result.fp, result.fn,
			# 		result.precision, result.recall, result.solution.count_cluster,
			# 		result.accuracy, result.f1_score
			# 	])

			headers = ['Question ID', 'Encoder', 'Dimension', 'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall',
				'Clusters', 'Accuracy', 'F1-Score']

			result_values = []
			for _, result in result_set.selected_results.items():
				values = [
					result_set.dataset.question.id, result_set.dataset.encoder.name,
					result_set.dataset.data_dim, result.tp, result.tn, result.fp, result.fn,
					result.precision, result.recall, result.solution.count_cluster,
					result.accuracy, result.f1_score
				]
				for key, value in result.solution.get_printable_params().items():
					if not key in headers:
						headers.append(key)
					values.append(value)
				result_values.append(values)
			
			rows.append(headers)
			rows += result_values

			utils.write_csv_file(path, '{}.csv'.format(file_name), rows)
		else:
			rows = []
			rows.append([
				'Question ID', 'Encoder', 'Dimension', 'TP', 'TN', 'FP', 'FN',
				'Precision', 'Recall', 'Clusters', 'Accuracy', 'F1-Score'
			])

			for _, result in result_set.selected_results.items():
				rows.append([
					result_set.dataset.question.id, result_set.dataset.encoder.name,
					result_set.dataset.data_dim,
					result.tp, result.tn, result.fp, result.fn,
					result.precision, result.recall, result.solution.count_cluster,
					result.accuracy, result.f1_score
				])
			
			utils.write_csv_file(path, '{}.csv'.format(file_name), rows)

		return time

def plot_graphs(algorithm, result_set, labels, plot='2d'):
	dataset = result_set.dataset

	if result_set.dataset.data_dim > 2:
		tsne_data = utils.get_tsne_data(result_set.dataset.data, mode=plot, perplexity=5.)
	else:
		tsne_data = result_set.dataset.data

	utils.plot(tsne_data, np.array(labels), title='Dataset, Question {}, Dimension: {}'.format(dataset.question.id, dataset.data_dim))
	r = list(result_set.selected_results.values())
	for i in range(len(r)):
		if algorithm.name != 'gal':
			utils.plot(tsne_data, r[i].solution.labels, title='Solution #{}, Cluster Count: {}, Accuracy: {:.2f}%'.format(i+1, r[i].solution.count_cluster, r[i].accuracy * 100))
		else:
			utils.plot(tsne_data, r[i].solution.labels if not result_set.reduced else r[i].solution.reduced_labels,
				title='{}{} {}, Cluster Count: {}, Accuracy: {:.2f}%'.format('(Reduced) ' if result_set.reduced else '',
				r[i].solution.density_method.upper(), r[i].solution.radius_grade, r[i].solution.count_cluster, r[i].accuracy * 100))

	plt.show()

def compute_inertia(a, X):
	W = [np.mean(pairwise_distances(X[a == c, :])) for c in np.unique(a)]
	return np.mean(W)

def compute_gap(data, k_max=5, n_references=5, clustering=KMeans()):
	if len(data.shape) == 1:
		data = data.reshape(-1, 1)
	reference = np.random.rand(*data.shape)
	reference_inertia = []
	
	for k in range(1, k_max+1):
		local_inertia = []
		for _ in range(n_references):
			clustering.n_clusters = k
			assignments = clustering.fit_predict(reference)
			local_inertia.append(compute_inertia(assignments, reference))
		reference_inertia.append(np.mean(local_inertia))

	ondata_inertia = []
	for k in range(1, k_max+1):
		clustering.n_clusters = k
		assignments = clustering.fit_predict(data)
		ondata_inertia.append(compute_inertia(assignments, data))

	gap = np.log(reference_inertia)-np.log(ondata_inertia)
	return gap, np.log(reference_inertia), np.log(ondata_inertia)

class Cluster(object):
	def __init__(self, centroid, data_indices, score, metadata=None, index=None,
		is_activated=True):
		self.centroid = centroid
		self.data_indices = data_indices
		self.is_activated = is_activated
		self.index = index
		self.score = score
		self.metadata = metadata if metadata is not None else dict()

		self.distributions = dict()
		self.distributions_data_indices = dict()
		self.centroid_data_index = None
		self.label = None
		self.data_num = 0
		self.error_count = np.inf
		self.evaluated = False

	def __str__(self):
		string = ''
		if not self.is_activated:
			string += '[NOT ACTIVATED]\n'
		if self.score is not None:
			string += 'Score: {}\n'.format(self.score)
		if self.is_activated:
			string += 'Label: {}\n'.format(self.label)
		string += 'Data Number: {}\n'.format(self.data_num)
		string += 'Distributions: {}\n'.format(self.distributions)
		string += 'Distribution Data Indices: {}\n'.format(self.distributions_data_indices)
		for key, value in self.metadata.items():
			string += '{}: {}\n'.format(key, value)
		
		if self.evaluated:
			string += 'TP: {}, FP: {}, TN: {}, FN: {}\n'.format(self.tp, self.fp, self.tn, self.fn)
			if self.tp > 0 and len(self.tp_text) > 0:
				string += 'TP Answer Text:\n{}\n'.format(self.tp_text)
			if self.tn > 0 and len(self.tn_text) > 0:
				string += 'TN Answer Text:\n{}\n'.format(self.tn_text)
			if self.fp > 0 and len(self.fp_text) > 0:
				string += 'FP Answer Text:\n{}\n'.format(self.fp_text)
			if self.fn > 0 and len(self.fn_text) > 0:
				string += 'FN Answer Text:\n{}\n'.format(self.fn_text)
			string += 'Error Count: {} ({:.2f}%)\n'.format(self.error_count, \
				self.error_count * 100 / self.data_num if self.data_num > 0 else 0)
			string += 'MSE: {}\n'.format(self.mse)
		return string

	def set_label_according_to_centroid(self, dataset, labels):
		data = dataset.data[self.data_indices]
		if self.centroid is None:
			self.centroid = np.mean(data, axis=0)
		distances = utils.dist([self.centroid], data, config.DISTANCE_FUNCTION)[0]
		self.centroid_data_index = self.data_indices[np.argmin(distances)]
		self.metadata['Centroid Data Index'] = self.centroid_data_index
		self.label = labels[self.centroid_data_index]

	def label_assignment(self, dataset, labels, known_data_labels=dict()):
		if self.label is not None:
			return

		if len(known_data_labels) > 0:
			intersection = list(set(self.data_indices) & set(known_data_labels.keys()))
			if len(intersection) > 0:
				self.label = known_data_labels[intersection[0]]
			else:
				self.label = unknown_label
		elif 'Centroid Data Index' in self.metadata:
			self.centroid_data_index = self.metadata['Centroid Data Index']
			self.metadata['Centroid Data Text'] = dataset.answers[self.centroid_data_index].text
			self.label = labels[self.centroid_data_index]
		else:
			self.set_label_according_to_centroid(dataset, labels)

	def evaluate(self, dataset, labels, known_data_labels=dict()):
		self.distributions = dict()
		self.distributions_data_indices = dict()
  		
		for index in self.data_indices:
			answer_class = labels[index]
			self.distributions[answer_class] = 1 if not answer_class in self.distributions else self.distributions[answer_class] + 1
			if not answer_class in self.distributions_data_indices:
				self.distributions_data_indices[answer_class] = list()
			self.distributions_data_indices[answer_class].append(index)

		if len(self.distributions) > 0:
			self.label_assignment(dataset=dataset, labels=labels, known_data_labels=known_data_labels)

			self.data_num = sum(self.distributions.values())
			self.error_count = sum(self.distributions[key] for key in self.distributions.keys() if not key == self.label)
		else:
			self.label = None
			self.data_num = 0
			self.error_count = 0

		self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0
		self.tp_text, self.tn_text, self.fp_text, self.fn_text = set(), set(), set(), set()
		self.mse = 0
  
		if self.label is None:
			return

		# class_count = len(np.unique(labels))
		unique_labels = sorted(np.unique(labels))
		class_count = len(unique_labels)
		if class_count == 2 and self.label == LABELS_2WAY[1]:
			self.tn = self.distributions[self.label] if self.label in self.distributions else 0
			self.fn = sum(self.distributions[key] for key in self.distributions.keys() if not key == self.label)
		else:
			self.tp = self.distributions[self.label] if self.label in self.distributions else 0.
			self.fp = sum(self.distributions[key] for key in self.distributions.keys() if not key == self.label)

		label_score = unique_labels.index(self.label) / (len(unique_labels) - 1)
		for key, data_count in self.distributions.items():
			if key == self.label:
				continue
			label_score_difference = abs(label_score - unique_labels.index(key) / (len(unique_labels) - 1))
			self.mse += label_score_difference * data_count
		
		for key, value in self.distributions_data_indices.items():
			for index in value:
				answer_text = dataset.answers[index].text if index < len(dataset.answers) else dataset.reference_answers[index - len(dataset.answers)].text
				if key == self.label:					
					if class_count == 2 and self.label == LABELS_2WAY[1]:
						self.tn_text.add(answer_text)
					else:
						self.tp_text.add(answer_text)
				else:
					if class_count == 2 and self.label == LABELS_2WAY[1]:
						self.fn_text.add(answer_text)
					else:
						self.fp_text.add(answer_text)
		self.error_count = self.fp + self.fn
		self.evaluated = True

class Result(ABC):
	def __init__(self, solution, dataset, labels):
		self.solution = solution
		self.dataset = dataset
		self.labels = labels

		self.class_count = len(np.unique(self.labels))

		self.distributions = dict()
		self.reference_answer_count = 0
		
		self.clusters = []
		self.error_count = None
		self.error_rate = None
		self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
		self.precision, self.recall, self.f1_score, self.accuracy = 0., 0., 0., 0.
		self.mse = 0

		self.evaluated = False
		self.print_non_activated = True

		self.evaluate()

	def __str__(self):
		string = self.details()
		string += self.clusters_details()
		return string

	def details(self):
		string = '[Result]\n'
		string += 'Dataset Name: {}\n'.format(self.dataset.name)
		string += 'Question ID: {}\n'.format(self.dataset.question.id)
		string += 'Number of Data: {}\n'.format(self.dataset.num_data)
		string += 'Distributions: {}\n'.format(self.distributions)
		string += 'Reference Answer Number: {}\n'.format(self.reference_answer_count)
		string += 'Encoder: {}\n'.format(self.dataset.encoder.name)
		string += 'Data Dimension: {}\n'.format(self.dataset.data_dim)
		string += 'Score: {}\n'.format(self.solution.score)
		string += 'Objectives: {}\n'.format(self.solution.objectives)
		if self.evaluated:
			string += '\nTP: {}, FP: {}, TN: {}, FN: {}\n'.format(self.tp, self.fp, self.tn, self.fn)
			if self.class_count != 2:
				string += 'Precisions: {}\n'.format(self.precisions)
				string += 'Recall: {}\n'.format(self.recalls)
				string += 'F1-Scores: {}\n'.format(self.f1_scores)
			string += 'Precision: {}\n'.format(self.precision)
			string += 'Recall: {}\n'.format(self.recall)
			string += 'F1-Score: {}\n'.format(self.f1_score)
			string += 'Accuracy: {}\n\n'.format(self.accuracy)
			string += 'Error Count: {} ({:.2f}%)\n'.format(self.error_count, self.error_rate)
			string += 'MSE: {}\n'.format(self.mse)
		return string

	def clusters_details(self):
		string = 'Number of Clusters: {}\n'.format(self.solution.count_cluster)
		if not self.print_non_activated:
			clusters = [c for c in self.clusters if c.is_activated]
		else:
			clusters = self.clusters

		for c in range(len(clusters)):
			string += '\n[CLUSTER #{}]'.format(c + 1 if clusters[c].index is None else clusters[c].index)
			string += '\n{}\n'.format(clusters[c])
		return string

	def calculate_error_rate(self):
		self.error_count = sum(cluster.error_count for cluster in self.clusters if cluster.is_activated)
		classified_data_number = sum(cluster.data_num for cluster in self.clusters if cluster.is_activated)
		self.error_rate = self.error_count * 100 / classified_data_number if classified_data_number > 0 else 0

	def evaluate(self):
		self.tp, self.fp, self.tn, self.fn = 0, 0, 0, 0
		self.mse = 0

		# find distributions
		self.distributions = dict()
		for i in range(len(self.dataset.answers)):
			answer = self.dataset.answers[i]
			answer_class = self.labels[i]
			self.distributions[answer_class] = 1 if not answer_class in self.distributions \
				else self.distributions[answer_class] + 1
			if answer.is_reference:
				self.reference_answer_count += 1

		self.clusters = []

		if self.solution.encoding == 'centroid':
			self.evaluate_centroid()
		elif self.solution.encoding == 'label':
			self.evaluate_label()
		elif self.solution.encoding == 'dendrogram':
			self.evaluate_dendrogram()
		elif self.solution.encoding == 'link':
			self.evaluate_link()
		elif self.solution.encoding == 'true_grade':
			self.evaluate_true_grade()
		else:
			raise utils.InvalidNameError('encoding', self.solution.encoding)

		for cluster in [c for c in self.clusters if c.is_activated]:
			self.tp += cluster.tp
			self.fp += cluster.fp
			self.tn += cluster.tn
			self.fn += cluster.fn
			self.mse += cluster.mse
		self.mse /= self.dataset.num_data

		self.precision, self.recall, self.f1_score, self.accuracy = 0., 0., 0., 0.
		
		if len(np.unique(self.labels)) == 2:
			self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
			self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
			self.f1_score = 2 * (self.recall * self.precision) / (self.recall + self.precision) \
				if (self.recall + self.precision) > 0 else 0
		else:
			self.precisions, self.recalls, self.f1_scores = dict(), dict(), dict()
			for key, value in self.distributions.items():
				correct_count, total_count = 0., 0.
				for cluster in [c for c in self.clusters if c.label == key and c.is_activated]:
					correct_count += cluster.tp
					total_count += cluster.tp + cluster.fp
				self.precisions[key] = correct_count / total_count if total_count > 0 else 0
				self.recalls[key] = correct_count / value if value > 0 else 0
				self.f1_scores[key] = 2 * (self.precisions[key] * self.recalls[key]) / (self.precisions[key] + self.recalls[key]) if (self.precisions[key] + self.recalls[key] > 0) else 0

			self.f1_score = sum([value for value in self.f1_scores.values()]) / len(self.f1_scores)
			self.precision = sum([value for value in self.precisions.values()]) / len(self.precisions)
			self.recall = sum([value for value in self.recalls.values()]) / len(self.recalls)

		total = self.tp + self.fp + self.fn + self.tn
		self.accuracy = (self.tp + self.tn) / total if total > 0 else 0

		self.calculate_error_rate()
		self.evaluated = True

	def evaluate_link(self):
		raise NotImplementedError('evaluate_link not implemented')

	def evaluate_dendrogram(self):
		if len(self.solution.clusters) <= 0:
			self.solution.clusters = self.solution.get_clusters(self.solution.abstract_clusters[self.solution.root_index])
		self.solution.count_cluster = len(self.solution.clusters)

		if len(self.solution.labels) <= 0:
			self.solution.labels = np.zeros(shape=self.dataset.num_data)
			update_label = True
		else:
			update_label = False

		for i in range(self.solution.count_cluster):
			if update_label:
				self.solution.labels[self.solution.clusters[i][1]] = i + 1

			if len(self.solution.abstract_clusters) > 0 \
				and self.solution.clusters[i][0] in self.solution.abstract_clusters:
				is_leaf = self.solution.abstract_clusters[self.solution.clusters[i][0]].is_seed
			else:
				is_leaf = False

			self.clusters.append(Cluster(data_indices=self.solution.clusters[i][1],
				centroid=np.mean(self.dataset.data[self.solution.clusters[i][1]]),
				score=None, index=i+1, is_activated=True,
				metadata={'Leaf': is_leaf}))
			self.clusters[len(self.clusters) - 1].evaluate(self.dataset, self.labels)

	def evaluate_label(self):
		for i in range(len(self.solution.centroids)):
			centroid = self.solution.centroids[i]
			if i < self.solution.reserved_cluster_number or centroid is not None:
				self.clusters.append(Cluster(data_indices=self.solution.cluster_data_indices[i],
					centroid=centroid, score=self.solution.cluster_scores[i],
					index=i - self.solution.reserved_cluster_number + 1,
					is_activated=(i >= self.solution.reserved_cluster_number)))
				self.clusters[len(self.clusters) - 1].evaluate(self.dataset, self.labels)

	def evaluate_centroid(self):
		for i in range(self.max_cluster_num):
			centroid = self.solution.centroids[i]
			probability = self.x[int(self.dataset.data_dim * self.max_cluster_num + i)]

			self.clusters.append(Cluster(data_indices=self.solution.cluster_data_indices[i],
				centroid=centroid, score=self.solution.cluster_scores[i],
				index=i - self.solution.reserved_cluster_number + 1,
				is_activated=(i >= self.solution.reserved_cluster_number),
				metadata={'Probability': probability}))
			self.clusters[len(self.clusters) - 1].evaluate(self.dataset, self.labels)

	def evaluate_true_grade(self):
		self.solution.count_cluster = 0
		unique = np.unique(self.solution.labels)
		for i in range(len(unique)):
			label = unique[i]
			indices = np.where(self.solution.labels == label)[0]
			cluster = Cluster(data_indices=indices, centroid=None, score=None,
				index = i + 1, is_activated=True)
			cluster.label = label
			cluster.evaluate(self.dataset, self.labels)
			self.clusters.append(cluster)
			self.solution.count_cluster += 1

class GALResult(Result):
	def details(self):
		string = super().details()
		for k, v in self.solution.get_printable_params().items():
			string += '{}: {}\n'.format(k, v)
		return string

class DBSCANResult(Result):	
	def details(self):
		string = 'Eps: {}\n'.format(self.solution.eps)
		string += 'Min Samples: {}\n'.format(self.solution.min_samples)
		string += 'Metric: {}\n\n'.format(self.solution.metric)
		string += super().details()
		string += 'Unclassified Number: {}\n'.format(self.solution.unclassified_num)
		string += 'Count Unclassified: {}\n'.format(self.solution.count_unclassified)
		return string

	def evaluate_label(self):
		super().evaluate_label()
		if self.solution.count_unclassified:
			for cluster in self.clusters:
				cluster.is_activated = True
				cluster.evaluate(self.dataset, self.labels)

class BirchResult(Result):
	def details(self):
		string = 'Threshold: {}\n'.format(self.solution.threshold)
		string += 'Branching Factor: {}\n'.format(self.solution.branching_factor)
		string += super().details()
		string += 'Maximum Cluster Number: {}'.format(self.solution.max_cluster_num)
		return string

class KMeansResult(Result):
	def details(self):
		string = super().details()
		string += 'Maximum Cluster Number: {}\n'.format(self.solution.max_cluster_num)
		string += 'Random State: {}\n'.format(self.solution.random_state)
		return string

class GaussianMixtureResult(Result):
	def details(self):
		string = super().details()
		for k, v in self.solution.get_printable_params().items():
			string += '{}: {}\n'.format(k, v)
		return string

class SpectralClusteringResult(Result):
	def details(self):
		string = ''
		for k, v in self.solution.params.items():
			string += '{}: {}\n'.format(k.replace('_', ' ').capitalize(), v)
		string += '\n'
		string += super().details()
		return string

class HDBSCANResult(Result):
	def details(self):
		string = 'Metric: {}\n'.format(self.solution.clusterer.metric_name)
		string += 'Min Samples: {}\n\n'.format(self.solution.clusterer.min_samples)
		string += super().details()
		string += 'Unclassified Number: {}\n'.format(self.solution.unclassified_num)
		string += 'Count Unclassified: {}\n'.format(self.solution.count_unclassified)
		return string

	def evaluate_label(self):
		super().evaluate_label()
		if self.solution.count_unclassified:
			for cluster in self.clusters:
				cluster.is_activated = True
				cluster.evaluate(self.dataset, self.labels)

class OPTICSResult(Result):
	def details(self):
		metric = self.solution.params.get('metric', None)
		p = self.solution.params.get('p', None)
		metric_params = self.solution.params.get('metric_params', None)
		string = 'Min Samples: {}\n'.format(self.solution.params.get('min_samples', None))
		cluster_method = self.solution.params.get('cluster_method', None)

		string += 'Max EPS: {}\n'.format(self.solution.params.get('max_eps', None))
		string += 'Metric: {}\n'.format(self.solution.params.get('metric', None))
		if metric == 'minkowski':
			string += 'Minkowski parameter: {}\n'.format('manhattan' if p == 1 else ('euclidean' if p == 2 else 'minkowski'))
		string += 'p: {}\n'.format(p)
		if metric_params is not None:
			string += 'Metric Params: {}\n'.format(metric_params)
		string += 'Cluster Method: {}\n'.format(cluster_method)
		if cluster_method == 'dbscan':
			string += 'EPS: {}\n'.format(self.solution.params.get('eps', None))
		elif cluster_method == 'xi':
			string += 'XI: {}\n'.format(self.solution.params.get('xi', None))
			string += 'Predecessor Correction: {}\n'.format(self.solution.params.get('predecessor_correction'))
			string += 'Min Cluster Size: {}\n'.format(self.solution.params.get('min_cluster_size', None))
		string += 'Nearest Neighbour Algorithm: {}\n'.format(self.solution.params.get('algorithm', None))
		string += 'Leaf Size: {}\n\n'.format(self.solution.params.get('leaf_size', None))
		string += super().details()
		string += 'Unclassified Number: {}\n'.format(self.solution.unclassified_num)
		string += 'Count Unclassified: {}\n'.format(self.solution.count_unclassified)
		return string

	def evaluate_label(self):
 		super().evaluate_label()
 		if self.solution.count_unclassified:
 			for cluster in self.clusters:
 				cluster.is_activated = True
 				cluster.evaluate(self.dataset, self.labels)

class ResultSet(object):
	def __init__(self, dataset, algorithm, version=None, encoding='label', framework='sklearn',
		time_used=None):
		self.dataset = dataset
		self.algorithm_name = algorithm.name
		self.version = version
		self.encoding = encoding
		self.framework = framework
		self.time_used = time_used

		self.metadata = dict()
		self.results = []
		self.selected_results = dict()

	def __str__(self):
		string = 'Time Used: {}\n'.format(self.time_used)
		string += 'Dataset Name: {}\n'.format(self.dataset.name)
		string += 'Algorithm Name: {}\n'.format(self.algorithm_name)
		if self.version is not None:
			string += 'Version: {}\n'.format(self.version.upper())
		for key, value in self.metadata.items():
			string += '\n{}: {}'.format(key, value)
		return string

	def get_printable_params(self):
		return {}

	def create_result(self, solution, labels):
		if self.algorithm_name == 'gal':
			self.result_class = GALResult
		elif self.algorithm_name == 'dbscan':
			self.result_class = DBSCANResult
		elif self.algorithm_name == 'birch':
			self.result_class = BirchResult
		elif self.algorithm_name == 'kmeans':
			self.result_class = KMeansResult
		elif self.algorithm_name == 'gaussian_mixture':
			self.result_class = GaussianMixtureResult
		elif self.algorithm_name == 'spectral_clustering':
			self.result_class = SpectralClusteringResult
		elif self.algorithm_name == 'hdbscan':
			self.result_class = HDBSCANResult
		elif self.algorithm_name == 'optics':
			self.result_class = OPTICSResult
		else:
			raise utils.InvalidNameError('algorithm', self.algorithm_name)
		return self.result_class(solution=solution, dataset=self.dataset, labels=labels)

	def handle_result(self, solutions, labels):
		if len(solutions) <= 0:
			return

		self.results = []
		self.selected_results = dict()

		i = 0
		for solution in solutions:
			result = self.create_result(solution, labels)
			self.results.append(result)

			print('Result #{}'.format(i+1))
			print('Cluster Count: {}, Score: {}, Accuracy: {}'.format(result.solution.count_cluster, result.solution.score, result.accuracy))
			i += 1

		self.selected_results['objective'] = self.results[0]
