import copy, math, importlib
from enum import Enum

import random
import numpy as np
from sklearn import decomposition

from abc import ABC, ABCMeta, abstractmethod

from core import utils
from basis import config, constants
from basis.problems import GALSubspace

ALGORITHM_NAME_DICT = dict()
for a in constants.ALGORITHMS:
	if 'kmeans' in a:
		ALGORITHM_NAME_DICT[a] = a.replace('kmeans', 'KMeans')
	else:
		ALGORITHM_NAME_DICT[a] = a.upper() if 'dbscan' in a or 'optics' in a \
			else a.replace('_', ' ').capitalize()

def get_subspace_dimension_string(subspace_param_list):
	dimensions = dict()
	encoders = []
	for param in subspace_param_list:
		encoder = param.get('encoder', '').lower()
		encoder = encoder.replace('google_universal_sentence_encoder', 'guse')
		random_reference_positions = param.get('random_reference_positions', [(0, 1)])
		if len(random_reference_positions) > 1:
			encoder += '_' + '_'.join([str(r[1]) for r in random_reference_positions])
		if not encoder in encoders:
			encoders.append(encoder)

		if param.get('weight', 1) <= 0:
			continue
		dimension = encoder + '_' + str(param.get('random_dimension', 'original'))
		random_selection_method = param.get('random_selection_method', None)
		horizontal_random_selection_number = param.get('horizontal_random_selection_number', dimension)
		if random_selection_method == 'vertical' or horizontal_random_selection_number == dimension:
			dimension = str(dimension) + ('H' if random_selection_method == 'horizontal' else 'V')
		elif random_selection_method == 'horizontal':
			# dimension += 'H'
			dimension = '{}_{}H'.format(horizontal_random_selection_number, dimension)
		if not dimension in dimensions:
			dimensions[dimension] = 1
		else:
			dimensions[dimension] += 1

	if len(encoders) > 1:
		encoder = '_'.join(encoders)
	else:
		encoder = encoders.pop()

	string = ''
	for dimension, count in dimensions.items():
		if string != '':
			string += '_'
		string += str(dimension)
		if count > 1:
			string += 'x{}'.format(count)
	return string if len(encoders) > 1 else string.replace(encoder + '_', ''), encoder

class Algorithm(ABC):
	def __init__(self, name, data, version=None):
		self.name = name
		self.data = data
		self.version = version
		self.result = None

	def __str__(self):
		if self.version is None:
			return '[{}]'.format(self.name.upper())
		else:
			return '[{}V{}]'.format(self.name.upper(), self.version)

	def set_attributes(self, args):
		for k, v in args.items():
			if k[0] == '_' or k[len(k)-1] == '_' or k in ['name', 'data', 'result', 'args']:
				continue
			
			if hasattr(self, k):
				setattr(self, k, v)

	@abstractmethod
	def run(self):
		raise NotImplementedError('run is not implemented')

	@abstractmethod
	def get_result(self):
		raise NotImplementedError('get_result not implemented')

class PairwiseRelation(Enum):
	SAME_ANS = 0
	SAME_GRADE = 1
	DIFF_GRADE = 2

class GradedAnswer(object):
	def __init__(self, cluster_id, core_index):
		self.cluster_id = cluster_id
		self.core_index = core_index
		self.indices = [self.core_index]

	def should_merge(self, index, pairwise_relations):
		for i in self.indices:
			if (i, index) in pairwise_relations \
				and pairwise_relations[(i, index)] == PairwiseRelation.SAME_ANS:
				return True

class QueryStatus(object):
	def __init__(self, index):
		self.index = index
		self.status = None
		self.remarks = None

class GAL(Algorithm):
	def __init__(self, subspace_param_list, grading_actions, iterations, possible_grades,
		possible_grades_weights=None, default_grade=None, random_seed=None, **kwargs):
		Algorithm.__init__(self, 'gal', None, version=kwargs.get('version', None))
		if grading_actions <= 0:
			raise Exception('grading_actions must be >= 1')

		# self.subspace_param_list = subspace_param_list
		self.grading_actions = grading_actions
		self.iterations = iterations
		self.batch_size = None
		self.batch_sizes = None
		self.possible_grades = possible_grades
		self.possible_grades_weights = possible_grades_weights
		self.default_grade = default_grade
		self.random_seed = random_seed
		
		self.version = config.VERSION
		self.version_variant = None
		self.distance_function = config.DISTANCE_FUNCTION
		
		self.relevant_subspace_number = config.RELEVANT_SUBSPACE_NUMBER
		
		self.rd_cutoff = config.RD_CUTOFF
		self.rd_deriving_factor = config.RD_DERIVING_FACTOR
		
		self.exclusion_rd_factor = config.EXCLUSION_RD_FACTOR
		self.min_exclusion_rd_factor = config.MIN_EXCLUSION_RD_FACTOR
		self.exclusion_rd_deduction_factor = config.EXCLUSION_RD_DEDUCTION_FACTOR
		self.marginalness_rd_factor = config.MARGINALNESS_RD_FACTOR
		self.voting_version = config.VOTING_VERSION
		self.grade_assignment_method = config.GRADE_ASSIGNMENT_METHOD
		self.outlier_removal = config.OUTLIER_REMOVAL

		self.subspace_replacement_ratio = config.SUBSPACE_REPLACEMENT_RATIO
		self.reset_exclusion_rd_factor = config.RESET_EXCLUSION_RD_FACTOR

		self.label_searching_boundary_factor = config.LABEL_SEARCHING_BOUNDARY_FACTOR
		self.label_knn = config.LABEL_KNN
		
		self.delta_link_threshold = config.DELTA_LINK_THRESHOLD
		self.delta_link_threshold_factor = config.DELTA_LINK_THRESHOLD_FACTOR
		self.delta_link_threshold_window_size_factor = config.DELTA_LINK_THRESHOLD_WINDOW_SIZE_FACTOR
		self.delta_link_threshold_window_size = None
		
		# Internal configurations
		self.pgv_use_normalized_density = True
		self.pgv_density_adjustment = None
		self.normalize_pgv = False
		self.normalize_distances = True
		
		self.diverse_answer_queries = True
		self.diverse_answer_queries_outlier_counts = False
		self.diverse_answer_queries_all = False

		self.dynamic_grade_weights = False
		self.weighted_global_entropy = False
		self.weighted_label_assignment = False
		
		self.subspaces = None
		self.reset_subspaces = True
		
		self.set_attributes(kwargs)
		# if self.version_variant is not None:
		# 	try:
		# 		version_variant = globals()['V{}_{}'.format(self.version, self.version_variant)]()
		# 		self.set_attributes(version_variant.params)
		# 	except Exception as e:
		# 		raise utils.InvalidNameError('version_variant', 'V{}_{}'.format(self.version, self.version_variant))

		if self.possible_grades_weights is None \
			or len(self.possible_grades_weights) != len(self.possible_grades):
			self.possible_grades_weights = [1 / len(self.possible_grades)] * len(self.possible_grades)
			# self.possible_grades_weights = [1] * len(self.possible_grades)

		if self.default_grade is None or self.default_grade not in self.possible_grades:
			self.default_grade = self.possible_grades[-1]

		self.known_data_labels, self.reduced_known_data_labels = dict(), dict()
		self.pairwise_relations = dict()

		self.current_exclusion_rd_factor = self.exclusion_rd_factor
		self.remaining_grading_actions = self.grading_actions
		
		self.reduced_data_indices, self.reduced_data_index_dict = None, None
		self.original_to_reduced_index_map = None

		if self.version != 4:
			self.subspace_replacement_ratio = None
		self.subspace_replacement_number = 0 if not self.subspace_replacement_ratio \
			else (int(self.subspace_replacement_ratio) if self.subspace_replacement_ratio >= 1 \
			else max(int(self.subspace_replacement_ratio * len(self.subspaces)), 1))

		if self.relevant_subspace_number == 0 or self.version < 5:
			self.relevant_subspace_number = None

		self.relevant_subspaces = None
		self.phrase1_selection_factor = 0.33

		self.current_iteration = 0
		
		self.init_subspaces(subspace_param_list)
		self.update_batch_size()
		
		self.graded_answers = []
		self.selection_phrase = 1

		self.labels = np.zeros(shape=self.reduced_num_data, dtype=int)
		self.true_grade_labels = np.empty(shape=self.reduced_num_data, dtype=object)

		# self.confidence_version = 'ig_norden'
		self.confidence_version = None

	def __str__(self):
		string = super().__str__()
		string += '\nSubspaces:'
		string += '\n'.join([str(s) for s in self.subspaces])
		string += '\nGrading Actions: {}\n'.format(self.grading_actions)
		return string

	def get_printable_params(self):
		params = {
			'Version': self.version,
			'Version Variant': self.version_variant,
			'Subspace Dimensions': get_subspace_dimension_string(self.subspace_param_list)[0],
			'Distance Function': self.distance_function,
			'Iterations:': self.iterations,
			'Grading Actions': self.grading_actions,
			'Batch Sizes': self.batch_sizes,
			'Diverse Answer Queries': self.diverse_answer_queries,
			'Dynamic Grade Weights': self.dynamic_grade_weights,
			'Possible Grade Weights': self.possible_grades_weights,
			'PGV Use Normalized Density': self.pgv_use_normalized_density,
			'PGV Density Adjustment': self.pgv_density_adjustment,
			'Voting Version': self.voting_version,
			'Confidence Version': self.confidence_version,
			'Normalize Distances': self.normalize_distances,
			'RD Cutoff': self.rd_cutoff,
			# 'Grading Actions First': self.grading_actions_first,
			# 'Grading Actions Iterate': self.grading_actions_iterate
		}
		if self.version != 6:
			params['Marginalness RD Factor'] = self.marginalness_rd_factor

		params['Grade Assignment Method'] = self.grade_assignment_method
		
		if self.version == 6:
			params['Delta Link Threshold Factor'] = self.delta_link_threshold_factor
			params['Delta Link Threshold'] = self.delta_link_threshold
			params['Delta Link Thredhold Window Size Factor'] = self.delta_link_threshold_window_size_factor
			params['Delta Link Threshold Window Size'] = self.delta_link_threshold_window_size
		
		params['Voting Version'] = self.voting_version
		
		if self.version == 4:
			params.update({
				'Subspace Replacement Ratio': self.subspace_replacement_ratio,
				'Subspace Replacement Number': self.subspace_replacement_number
			})
		
		params['Relevant Subspace Number'] = self.relevant_subspace_number
		
		if self.version == 5.5:
			params['Phrase1 Selection Factor'] = self.phrase1_selection_factor

		if self.version > 3 and self.version < 6:
			params.update({
				'Exclusion RD Factor': self.exclusion_rd_factor,
				'Min Exclusion RD Factor': self.min_exclusion_rd_factor,
				'Exclusion RD Deduction Factor': self.exclusion_rd_deduction_factor
			})
		
		params['Random Seed'] = self.random_seed
		
		return params

	def get_subspace_printable_params(self):
		names, weights, random_selections, rd_cutoffs, delta_link_thresholds = [], [], [], [], []
		relevant_counts = []
		for subspace in self.subspaces:
			names.append(subspace.get_name())
			weights.append(subspace.weight)
			random_selections.append(str(subspace.random_selections) \
				if subspace.random_selections is not None else '')
			rd_cutoffs.append(subspace.rd_cutoff)
			delta_link_thresholds.append(subspace.delta_link_threshold \
				if subspace.grade_assignment_method == 'parent_breaks' else '')

		params = {
			'Subspaces': names,
			'Weights': weights,
			'Random Selections': random_selections,
		}
		# if self.version == 5 or self.version == 5.5:
		if self.relevant_subspaces is not None:
			relevant_counts = np.zeros(shape=len(self.subspaces))
			for relevant_subspaces in self.relevant_subspaces:
				for index in relevant_subspaces:
					relevant_counts[index] += 1
			params['Relevant Counts'] = relevant_counts.tolist()
		params['RD Cutoff'] = rd_cutoffs
		params['Delta Link Threshold'] = delta_link_thresholds
		return params

	def reset(self, subspace_param_list, grading_actions=None, iterations=None, **kwargs):
		self.set_attributes(kwargs)
		self.init_subspaces(subspace_param_list)

		self.current_iteration = 0
		if grading_actions is not None:
			self.grading_actions = grading_actions
		if iterations is not None:
			self.iterations = iterations
		self.remaining_grading_actions = self.grading_actions
		self.update_batch_size()
		if self.version == 3:
			self.subspace_replacement_ratio = None
		self.subspace_replacement_number = 0 if not self.subspace_replacement_ratio \
			else (int(self.subspace_replacement_ratio) if self.subspace_replacement_ratio >= 1 \
			else max(int(self.subspace_replacement_ratio * len(self.subspaces)), 1))

		# reset exclusion RD
		self.current_exclusion_rd_factor = self.exclusion_rd_factor
		self.exclusion_rd = self.rd_cutoff * self.exclusion_rd_factor

		# rest quest status
		for qs in self.query_status_list:
			qs.status = None

		# clear marked answers
		self.graded_answers.clear()
		self.pairwise_relations = dict()
		self.known_data_labels.clear()
		self.reduced_known_data_labels.clear()
		for subspace in self.subspaces:
			subspace.known_data_labels.clear()
			subspace.reduced_known_data_labels.clear()

		# clear labels
		self.labels = np.zeros(shape=self.reduced_num_data, dtype=int)
		self.true_grade_labels = np.empty(shape=self.reduced_num_data, dtype=object)

	# def run(self):
	# 	if self.remaining_grading_actions <= 0:
	# 		return [], [], None, None, None, None, None, None, None, None
	# 	assigned_grading_actions = min(self.grading_actions_first if len(self.graded_answers) <= 0 \
	# 		else self.grading_actions_iterate, self.remaining_grading_actions)
	# 	return self.get_valuable_answer_indices(assigned_grading_actions)

	def run(self):
		if self.current_iteration > self.iterations - 1:
			return [], [], None, None, None, None, None, None, None, None
		assigned_grading_actions = self.batch_sizes[self.current_iteration]
		return self.get_valuable_answer_indices(assigned_grading_actions)

	def get_result(self):
		unique_label_list, true_grade_label_list = [], []
		confidence_level_dict_list, marginalness_list = [], []
		for subspace in self.subspaces:
			nearest_ground_truth_distances, nearest_ground_truth_index_map = subspace.get_nearest_ground_truths()
			_ = subspace.update_labels(self.graded_answers, self.labels, nearest_ground_truth_index_map,
				nearest_ground_truth_distances)
			unique_labels, true_grade_labels, confidence_levels_dict, marginalness \
				= subspace.update_true_grade_labels(self.graded_answers, self.labels)
			for label in unique_labels:
				if not label in unique_label_list:
					unique_label_list.append(label)
			true_grade_label_list.append(true_grade_labels)
			confidence_level_dict_list.append(confidence_levels_dict)
			marginalness_list.append(marginalness / marginalness.max() if marginalness.max() != 0 \
				else marginalness)
		
		self.true_grade_labels = np.empty(shape=self.reduced_num_data, dtype=object)
		subspace_total_weights = sum([s.weight for s in self.subspaces])
		for index in range(self.reduced_num_data):
			if 'average' in self.voting_version:
				confidence_levels = np.zeros(shape=len(unique_label_list))
				for label_index in range(len(unique_label_list)):
					label = unique_label_list[label_index]
					if self.voting_version == 'average':
						# for d in confidence_level_dict_list:
						count = 0
						for s in range(len(self.subspaces)):
							# if (self.version == 5 or self.version == 5.5) \
							if self.relevant_subspaces is not None \
								and not s in self.relevant_subspaces[index]:
								continue
							d = confidence_level_dict_list[s]
							confidence_levels[label_index] += d[label][index]
							count += 1
						# confidence_levels[label_index] /= len(confidence_level_dict_list)
						confidence_levels[label_index] /= count
					elif self.voting_version == 'weighted_average' \
						or self.voting_version == 'weighted_ig_average':
						subspace_total_weights = 0
						for s in range(len(self.subspaces)):
							# if (self.version == 5 or self.version == 5.5) \
							if self.relevant_subspaces is not None \
								and not s in self.relevant_subspaces[index]:
								continue
							d = confidence_level_dict_list[s]
							# confidence_levels[label_index] += d[label][index] * self.subspaces[s].weight
							if not self.weighted_label_assignment:
								confidence_levels[label_index] += d[label][index] * self.subspaces[s].weight
							else:
								confidence_levels[label_index] += d[label][index] * self.subspaces[s].weight * self.possible_grades_weights[label_index]
							subspace_total_weights += self.subspaces[s].weight
						confidence_levels[label_index] /= subspace_total_weights

				self.true_grade_labels[index] = unique_label_list[np.argmax(confidence_levels)]
			elif self.voting_version == 'lowest':
				# m = [marginalness_list[s][index] for s in range(len(self.subspaces))]
				# if self.version == 5 or self.version == 5.5:
				if self.relevant_subspaces is not None:
					m = [marginalness_list[s][index] for s in self.relevant_subspaces[index]]
				else:
					m = [marginalness_list[s][index] for s in range(len(self.subspaces))]
				s = np.argmin(m)
				self.true_grade_labels[index] = self.subspaces[s].true_grade_labels[index]
			# elif self.voting_version == 'majority':
			elif 'majority' in self.voting_version:
				counts = [0] * len(unique_label_list)
				for s in range(len(self.subspaces)):
					# if (self.version == 5 or self.version == 5.5) \
					if self.relevant_subspaces is not None \
						and not s in self.relevant_subspaces[index]:
						continue
					counts[unique_label_list.index(true_grade_label_list[s][index])] += self.subspaces[s].weight

				if self.voting_version == 'majority_ordinal':
					num_possible_grades = len(self.possible_grades)
					score = 0
					for l in range(len(unique_label_list)):

						score += (num_possible_grades - self.possible_grades.index(unique_label_list[l]) - 1) * counts[l]
					score /= sum(counts)
					self.true_grade_labels[index] = self.possible_grades[num_possible_grades - round(score) - 1]
				else:
					# self.true_grade_labels[index] = unique_label_list[np.argmax(counts)]
					sorted_labels, sorted_counts = zip(*sorted(zip(unique_label_list, counts),
						key=lambda i: i[1], reverse=True))
					choices = [sorted_labels[c] for c in range(len(sorted_counts)) if sorted_counts[c] == sorted_counts[0]]
					self.true_grade_labels[index] = random.choices(population=choices)[0] if len(choices) > 1 else choices[0]

		result = np.empty(shape=self.num_data, dtype=object)
		for index in range(self.reduced_num_data):
			result[self.get_original_data_indices([index])] = self.true_grade_labels[index]
		return result

	def mark_answers(self, answer_dict):
		should_update_subspaces = self.version == 4 and len(self.known_data_labels) > 0
		self.known_data_labels.update(answer_dict)
		if len(self.known_data_labels) <= 0:
			raise Exception('At least 1 data must be known')

		for key, value in self.known_data_labels.items():
			index = self.original_to_reduced_index_map[key]
			self.reduced_known_data_labels[index] = value

		reduced_answer_dict = dict()
		indices = []
		for key, value in answer_dict.items():
			reduced_index = self.original_to_reduced_index_map[key]
			indices.append(reduced_index)
			reduced_answer_dict[reduced_index] = value

		self.update_graded_answers(indices=indices)
		self.update_subspaces(should_update=should_update_subspaces)

	def add_data(self, subspace_param_list, additional_grading_actions, additional_batch_sizes=[]):
		self.subspace_param_list = subspace_param_list
		self.grading_actions += additional_grading_actions
		self.remaining_grading_actions += additional_grading_actions
		self.batch_sizes += additional_batch_sizes if additional_batch_sizes is not None \
			else [additional_grading_actions]
		self.update_batch_size()
		
		recompute = True
		# density_list = []
		for param in subspace_param_list:
			subspace = next((s for s in self.subspaces \
				if s.encoder == param['encoder'] and len(param['new_data'][0]) == s.data_dim), None)
			subspace.add_data(param['new_data'], encapsulate=recompute, recalculate=True)
			
			if recompute:
				self.reduced_data_indices = subspace.reduced_data_indices
				self.reduced_data_index_dict = subspace.reduced_data_index_dict
				self.original_to_reduced_index_map = subspace.original_to_reduced_index_map
				self.num_data = self.subspaces[0].num_data
				self.reduced_num_data = len(self.reduced_data_indices)
				recompute = False
			else:
				subspace.set_reduced_data(self.reduced_data_indices, self.reduced_data_index_dict)
			# density_list.append(subspace.densities / subspace.densities.max())

		self.update_representations()

		self.labels = np.zeros(shape=self.reduced_num_data, dtype=int)
		self.true_grade_labels = np.empty(shape=self.reduced_num_data, dtype=object)

	def init_subspaces(self, subspace_param_list):
		if self.random_seed is not None:
			random.seed(self.random_seed)
		
		self.subspace_param_list = subspace_param_list

		if self.reset_subspaces:
			self.subspaces = []
			for params in self.subspace_param_list:
				self.subspaces.append(self.create_subspace(params))
		else:
			self.update_reduced_data_indices(self.subspaces[0])
			for s in range(len(subspace_param_list)):
				# TO-DO: remove hot fix
				# self.subspaces[s].voting_version = None
				# self.subspaces[s].random_selection_method = None
				# self.subspaces[s].horizontal_random_selection_number = None
				# self.subspaces[s].label_searching_boundary_factor = None
				# self.subspaces[s].label_knn = None
				# self.subspaces[s].delta_link_threshold_factor = None
				# self.subspaces[s].delta_link_threshold_window_size_factor = None
				# self.subspaces[s].delta_link_threshold_window_size = None
				# self.subspaces[s].delta_link_threshold_curvature = None
				# self.subspaces[s].delta_link_threshold = None
				# self.subspaces[s].possible_grades = None
				# self.subspaces[s].possible_grades_weights = None
				# self.subspaces[s].version_variant = None
				# self.subspaces[s].rd_deriving_factor = None
				# self.subspaces[s].dynamic_grade_weights = None
				# self.subspaces[s].pgv_use_normalized_density = None
				# self.subspaces[s].pgv_density_adjustment = None
				# self.subspaces[s].normalize_distances = None
				self.subspaces[s].set_attributes(subspace_param_list[s])
				self.subspaces[s].set_attributes(self.__dict__)
				self.subspaces[s].update_representations()

		self.update_representations()
		self.reset_subspaces = False

	def update_subspaces(self, should_update):
		predictive_value_list = []
		for s in range(len(self.subspaces)):
			subspace = self.subspaces[s]
			subspace.known_data_labels = self.known_data_labels
			subspace.reduced_known_data_labels = self.reduced_known_data_labels
			if subspace.random_selections is None or not should_update:
				continue
			predictive_value, predict_correct_count \
				= subspace.calculate_predictive_value(self.graded_answers, self.labels)
			print('{} predictive_value: {} ({})'.format(s, predictive_value, predict_correct_count))
			predictive_value_list.append((s, predictive_value))
		if should_update and self.subspace_replacement_number > 0:
			sorted_predictive_value_list = sorted(predictive_value_list,
				key=lambda p: p[1])[:self.subspace_replacement_number]
			print('removal list: ', sorted_predictive_value_list)
			for p in sorted_predictive_value_list:
				params = self.subspace_param_list[p[0]]
				subspace = self.create_subspace(params)
				subspace.known_data_labels = self.known_data_labels
				subspace.reduced_known_data_labels = self.reduced_known_data_labels	
				self.subspaces[p[0]] = subspace

			if self.reset_exclusion_rd_factor:
				self.current_exclusion_rd_factor = self.exclusion_rd_factor
			self.update_representations()

	def select_random_indices_for_subspace(self, source_dimension, dimension):
		selected = []
		population = list(range(source_dimension))
		for i in range(dimension):
			index = random.choices(population=population)[0]
			selected.append(index)
			population.remove(index)
		return selected

	def select_random_dimensions(self, random_dimension, horizontal_random_selection_number,
		reference_subspace, random_selection_method):
		while True:
			if random_selection_method == 'vertical':
				selected = self.select_random_indices_for_subspace(reference_subspace.data_dim,
					random_dimension)
			else:
				selected = self.select_random_indices_for_subspace(reference_subspace.num_data,
					horizontal_random_selection_number)
			duplicated = False
			for subspace in self.subspaces:
				if subspace.random_selections is not None \
				and subspace.random_selection_method == random_selection_method \
				and len(set(subspace.random_selections) - set(selected)) <= 0:
					duplicated = True
					break
			if not duplicated:
				random_selections = sorted(selected)
				break

		if random_selection_method == 'vertical':
			data = np.copy(reference_subspace.data)
			data = np.transpose(np.transpose(data)[random_selections])
		else:
			data = np.copy(reference_subspace.data)
			data = data[random_selections]

		return data, random_selections

	# def create_subspace(self, params):
	# 	if 'data' in params:
	# 		data = params.pop('data')
		
	# 	random_selections = None
	# 	if 'random_dimension' in params:
	# 		random_dimension = params.get('random_dimension', None)
	# 		random_reference_position = params.get('random_reference_position', 0)

	# 		reference_subspace = self.subspaces[random_reference_position]

	# 		random_selection_method = params.get('random_selection_method', None)
	# 		if random_selection_method is None:
	# 			random_selection_method = 'vertical'
	# 			params['random_selection_method'] = 'vertical'
	# 		horizontal_random_selection_number = params.get('horizontal_random_selection_number',
	# 			random_dimension)

	# 		while True:
	# 			if random_selection_method == 'vertical':
	# 				selected = self.select_random_indices_for_subspace(reference_subspace.data_dim,
	# 					random_dimension)
	# 			else:
	# 				selected = self.select_random_indices_for_subspace(reference_subspace.num_data,
	# 					horizontal_random_selection_number)
	# 			duplicated = False
	# 			for subspace in self.subspaces:
	# 				if subspace.random_selections is not None \
	# 				and subspace.random_selection_method == random_selection_method \
	# 				and len(set(subspace.random_selections) - set(selected)) <= 0:
	# 					duplicated = True
	# 					break
	# 			if not duplicated:
	# 				random_selections = sorted(selected)
	# 				params['random_selections'] = random_selections
	# 				break

	# 		if random_selection_method == 'vertical':
	# 			data = np.copy(reference_subspace.data)
	# 			data = np.transpose(np.transpose(data)[random_selections])
	# 		else:
	# 			# select random answers (horizontally), fit to pca and transform
	# 			data = np.copy(reference_subspace.data)
	# 			selected_data = data[random_selections]
	# 			pca = decomposition.PCA(n_components=random_dimension, svd_solver='full')
	# 			pca.fit(selected_data)
	# 			data = pca.transform(data)
	# 			# svd = decomposition.TruncatedSVD(n_components=random_dimension)
	# 			# svd.fit(selected_data)
	# 			# data = svd.transform(data)
		
	# 	if data is None:
	# 		raise Exception('Either data or (random_dimension and random_reference_position) must be provided in subspace params')
	# 	encoder = params.pop('encoder') if 'encoder' in params else None
	# 	model = params.pop('model') if 'model' in params else None

	# 	algorithm_params = copy.copy(self.__dict__)
	# 	algorithm_params.pop('data')
	# 	params.update(algorithm_params)
	# 	subspace = GALSubspace(data, encoder, model, **params)
	# 	params.update({'data': data, 'encoder': encoder, 'model': model})
		
	# 	if self.reduced_data_indices is None:
	# 		self.update_reduced_data_indices(subspace)
	# 	return subspace

	def create_subspace(self, params):
		if 'data' in params:
			data = params.pop('data')
		
		if 'random_dimension' in params:
			random_dimension = params.get('random_dimension', None)
			random_reference_positions = params.get('random_reference_positions', [(0, 1)])

			random_selection_method = params.get('random_selection_method', None)
			if random_selection_method is None:
				random_selection_method = 'vertical'
				params['random_selection_method'] = 'vertical'
			horizontal_random_selection_number = params.get('horizontal_random_selection_number',
				random_dimension)

			data, random_selections = None, None
			for random_reference_position, proportion in random_reference_positions:
				dimension = int(random_dimension * proportion)
				reference_subspace = self.subspaces[random_reference_position]

				d, r = self.select_random_dimensions(dimension,
					horizontal_random_selection_number, reference_subspace, random_selection_method)
				data = np.transpose(d) if data is None else np.append(data, np.transpose(d), 0)
				random_selections = r if random_selections is None else random_selections + r
			data = np.transpose(data)
			params['random_selections'] = random_selections

			if random_selection_method == 'horizontally':
				# select random answers (horizontally), fit to pca and transform
				# data = np.copy(reference_subspace.data)
				# selected_data = data[random_selections]
				pca = decomposition.PCA(n_components=random_dimension, svd_solver='full')
				data = pca.transform(self.subspaces[random_reference_positions[0]])
				# svd = decomposition.TruncatedSVD(n_components=random_dimension)
				# svd.fit(selected_data)
				# data = svd.transform(data)
		
		if data is None:
			raise Exception('Either data or (random_dimension and random_reference_position) must be provided in subspace params')
		encoder = params.pop('encoder') if 'encoder' in params else None
		model = params.pop('model') if 'model' in params else None

		algorithm_params = copy.copy(self.__dict__)
		algorithm_params.pop('data')
		params.update(algorithm_params)
		subspace = GALSubspace(data, encoder, model, **params)
		params.update({'data': data, 'encoder': encoder, 'model': model})
		
		if self.reduced_data_indices is None:
			self.update_reduced_data_indices(subspace)
		return subspace

	def update_reduced_data_indices(self, subspace):
		self.reduced_data_indices = subspace.reduced_data_indices
		self.reduced_data_index_dict = subspace.reduced_data_index_dict
		self.original_to_reduced_index_map = subspace.original_to_reduced_index_map

	def update_batch_size(self):
		if self.batch_sizes is not None:
			total = sum(self.batch_sizes)
			if total < self.grading_actions:
				self.batch_sizes.append(self.grading_actions - total)
			elif total > self.grading_actions:
				t, batch_sizes = 0, []
				for b in self.batch_sizes:
					if t + b >= self.grading_actions:
						batch_sizes.append(self.grading_actions - t)
					else:
						batch_sizes.append(b)
						t += b
				self.batch_sizes = batch_sizes
			self.iterations = len(self.batch_sizes)
		elif self.version != 5.5:
			if self.iterations is not None:
				grading_actions_iterate = int(self.grading_actions / self.iterations)
				grading_actions_first = (self.grading_actions % self.iterations) + grading_actions_iterate
				self.batch_sizes = [grading_actions_first] + [grading_actions_iterate] * (self.iterations - 1)
			else:
				self.batch_sizes = [self.batch_size] * int(self.grading_actions / self.batch_size)
				remaining = self.grading_actions % self.batch_size
				if remaining > 0:
					self.batch_sizes.append(remaining)
				self.iterations = len(self.batch_sizes)
		else:
			phrase1_selection_grading_actions = int(self.grading_actions * phrase1_selection_factor) + 1
			phrase2_selection_grading_actions = self.grading_actions - phrase1_selection_grading_actions
			self.batch_sizes = [phrase1_selection_grading_actions] + [phrase2_selection_grading_actions] * (self.iterations - 1)

	def update_pairwise_relations(self, i, j, relation):
		self.pairwise_relations[(i, j)] = relation
		self.pairwise_relations[(j, i)] = relation

	def update_relevant_subspaces(self):
		active_subspace_indices = [s for s in range(len(self.subspaces)) if self.subspaces[s].weight > 0]
		self.relevant_subspaces = np.zeros(shape=(self.reduced_num_data,
			min(self.relevant_subspace_number, len(active_subspace_indices))), dtype=int)
		
		subspace_density_list, subspace_spaciousness_list = [], []
		for s in self.subspaces:
			subspace_density_list.append(s.normalized_densities * s.weight)
			subspace_spaciousness_list.append(s.normalized_spaciousness * s.weight)
		subspace_density_list = np.array(subspace_density_list)
		subspace_spaciousness_list = np.array(subspace_spaciousness_list)
		
		for i in range(self.reduced_num_data):
			data_densities = subspace_density_list[active_subspace_indices, i]
			data_spaciousness =  subspace_spaciousness_list[active_subspace_indices, i]

			sorted_list = sorted(list(range(len(active_subspace_indices))),
				key=lambda s: (-data_densities[s], data_spaciousness[s]))[:self.relevant_subspace_number]
			self.relevant_subspaces[i] = [active_subspace_indices[s] for s in sorted_list]

	def update_representations(self):
		self.num_data = self.subspaces[0].num_data
		self.reduced_num_data = len(self.reduced_data_indices)
		self.query_status_list = [QueryStatus(i) for i in range(self.reduced_num_data)]
		self.reduced_data_distances, self.densities, self.spaciousness = None, None, None
		self.exclusion_rd, self.min_exclusion_rd = None, None

		if self.relevant_subspace_number is not None:
			self.update_relevant_subspaces()
		self.calculate_densities()
		self.calculate_spaciousness()
		self.update_query_status_list()
		# if self.version == 5 or self.version == 5.5:
		if self.relevant_subspaces is not None:
			self.reduced_data_distances = self.calculate_relevant_weighted_average_subspace_values(
				[s.reduced_data_distances for s in self.subspaces])
		elif self.version == 3 or self.version == 4:
			# self.reduced_data_distances = self.calculate_weighted_average_subspace_values(
			# 	[s.reduced_data_distances * s.weight for s in self.subspaces])
			self.reduced_data_distances = self.calculate_weighted_average_subspace_values(
				[s.reduced_data_distances for s in self.subspaces])
		else:
			self.reduced_data_distances = np.mean([s.reduced_data_distances for s in self.subspaces], axis=0)
		if self.rd_cutoff is None:
			# self.rd_cutoff = np.mean([s.rd_cutoff for s in self.subspaces])
			self.rd_cutoff = np.mean([s.rd_cutoff * s.weight for s in self.subspaces if s.weight > 0])
		self.delta_link_threshold = np.mean([s.delta_link_threshold * s.weight for s in self.subspaces \
			if s.weight > 0 and s.delta_link_threshold is not None])
		self.exclusion_rd = self.rd_cutoff * self.exclusion_rd_factor
		self.min_exclusion_rd = self.rd_cutoff * self.min_exclusion_rd_factor

	def update_query_status_list(self):
		self.query_status_list = sorted(self.query_status_list,
			key=lambda q: (-self.densities[q.index], self.spaciousness[q.index]))

	def is_excluded(self, index, selected_indices):
		if len(selected_indices) <= 0:
			return False
		min_distance = np.min(self.reduced_data_distances[index][selected_indices])
		if min_distance <= self.exclusion_rd:
			return True
		return False

	def deduct_exclusion_rd(self):
		if self.min_exclusion_rd_factor > 0 \
			and self.current_exclusion_rd_factor - self.exclusion_rd_deduction_factor \
			< self.min_exclusion_rd_factor:
			return False
		elif self.min_exclusion_rd_factor <= 0 and self.current_exclusion_rd_factor <= 0:
			return False
		self.current_exclusion_rd_factor -= self.exclusion_rd_deduction_factor
		self.exclusion_rd = self.rd_cutoff * self.current_exclusion_rd_factor

		for qs in self.query_status_list:
			if qs.status == 'skipped':
				qs.status = None
		return True

	def record_selection_params(self, dictionary, index, method):
		dictionary[index] = dict()
		dictionary[index]['method'] = method
		dictionary[index]['exclusion_rd'] = self.exclusion_rd
		dictionary[index]['exclusion_rd_factor'] = self.current_exclusion_rd_factor

	def select_answers_v2(self, assigned_grading_actions, informations):
		indices, skipped_qs_indices = [], []
		selection_param_dict = dict()
		selected_indices = list(self.reduced_known_data_labels.keys())

		for qs in self.query_status_list:
			if qs.status == 'selected':
				continue
			elif qs.status == 'skipped':
				skipped_qs_indices.append(qs.index)
				continue
			
			if self.version == 1 or (self.version == 2 and self.densities[qs.index] > 0):
				is_close = False				
				all_selected_indices = selected_indices + indices
				if len(all_selected_indices) > 0:
					min_distance = np.min(self.reduced_data_distances[qs.index][all_selected_indices])
					if min_distance <= self.exclusion_rd:
						is_close = True
				
				if not is_close:
					qs.status = 'consider'
					indices.append(qs.index)
					self.record_selection_params(selection_param_dict, qs.index, 'Density')
					if len(indices) >= assigned_grading_actions:
						break
				else:
					qs.status = 'skipped'
					skipped_qs_indices.append(qs.index)
			elif self.version == 2:
				skipped_qs_indices.append(qs.index)
		
		if len(indices) < assigned_grading_actions:
			sorted_skipped_qs_indices = sorted(skipped_qs_indices, key=lambda i: informations[i],
				reverse=True)
			# indices += sorted_skipped_qs_indices[:(assigned_grading_actions - len(indices))]
			remaining_grading_actions = assigned_grading_actions - len(indices)
			# indices += sorted_skipped_qs_indices[:remaining_grading_actions]
			# selected_methods += ['Information'] * remaining_grading_actions
			for index in sorted_skipped_qs_indices[:remaining_grading_actions]:
				indices.append(index)
				self.record_selection_params(selection_param_dict, index, 'Information')
		
		return indices, selection_param_dict
	
	def select_answers_v3(self, assigned_grading_actions, informations):
		indices, selection_param_dict = [], dict()
		selected_indices = list(self.reduced_known_data_labels.keys())

		full = False
		while True:
			# select byb density
			case2_qs_list = []
			for qs in self.query_status_list:
				if qs.status == 'selected' or qs.status == 'skipped' or self.densities[qs.index] <= 0:
					if (qs.status is None or qs.status == 'skipped') and informations[qs.index] > 0:
						case2_qs_list.append(qs)
					continue
				excluded = self.is_excluded(qs.index, selected_indices + indices)
				if not excluded:
					qs.status = 'consider'
					indices.append(qs.index)
					self.record_selection_params(selection_param_dict, qs.index, 'Density')
					full = len(indices) >= assigned_grading_actions
					if full:
						break
				else:
					qs.status = 'skipped'

			if full:
				break

			# select by information
			if len(case2_qs_list) > 0:
				sorted_case2_qs_list = sorted(case2_qs_list, key=lambda qs: informations[qs.index],
					reverse=True)
				for qs in sorted_case2_qs_list:
					excluded = self.is_excluded(qs.index, selected_indices + indices)
					if not excluded:
						qs.status = 'consider'
						indices.append(qs.index)
						self.record_selection_params(selection_param_dict, qs.index, 'Information')
						full = len(indices) >= assigned_grading_actions
						if full:
							break
					else:
						qs.status = 'skipped'

			if full or not self.deduct_exclusion_rd():
				break
		return indices, selection_param_dict

	def proceed_to_next_selection_pharse(self):
		self.selection_phrase += 1
		self.current_exclusion_rd_factor = self.exclusion_rd_factor
		self.exclusion_rd = self.rd_cutoff * self.exclusion_rd_factor

	def select_answers_v5_5(self, assigned_grading_actions, informations):
		indices, selection_param_dict = [], dict()
		selected_indices = list(self.reduced_known_data_labels.keys())

		full, phrase1_full = False, False
		if self.selection_phrase == 1:
			while True:
				# select byb density
				# case2_qs_list = []
				for qs in self.query_status_list:
					if qs.status == 'selected' or qs.status == 'skipped' or self.densities[qs.index] <= 0:
						# if (qs.status is None or qs.status == 'skipped') and informations[qs.index] > 0:
							# case2_qs_list.append(qs)
						continue
					excluded = self.is_excluded(qs.index, selected_indices + indices)
					if not excluded:
						qs.status = 'consider'
						indices.append(qs.index)
						self.record_selection_params(selection_param_dict, qs.index, 'Density')
						full = len(indices) >= assigned_grading_actions
						phrase1_full = (len(selected_indices) + len(indices)) >= self.phrase1_selection_grading_actions
						if phrase1_full:
							self.proceed_to_next_selection_pharse()
							break
						elif full:
							break
					else:
						qs.status = 'skipped'

				if full or phrase1_full:
					break
				elif not self.deduct_exclusion_rd():
					self.proceed_to_next_selection_pharse()
					break

		if not full and self.selection_phrase == 2:
			while True:
				# select by information
				case2_qs_list = [qs for qs in self.query_status_list \
					if (qs.status == 'selected' or qs.status == 'skipped' \
					or self.densities[qs.index] <= 0)\
					and ((qs.status is None or qs.status == 'skipped') and informations[qs.index] > 0)]
				
				if len(case2_qs_list) > 0:
					sorted_case2_qs_list = sorted(case2_qs_list, key=lambda qs: informations[qs.index],
						reverse=True)
					for qs in sorted_case2_qs_list:
						excluded = self.is_excluded(qs.index, selected_indices + indices)
						if not excluded:
							qs.status = 'consider'
							indices.append(qs.index)
							self.record_selection_params(selection_param_dict, qs.index, 'Information')
							full = len(indices) >= assigned_grading_actions
							if full:
								break
						else:
							qs.status = 'skipped'

				if full or not self.deduce_exclusion_rd():
					break
		return indices, selection_param_dict

	def select_answers_v6(self, assigned_grading_actions, pgv_values):
		indices = []
		sorted_indices = sorted(list(range(self.reduced_num_data)),key=lambda i: pgv_values[i], reverse=True)
		for index in sorted_indices:
			if index in self.reduced_known_data_labels:
				continue
			indices.append(index)
			if len(indices) >= assigned_grading_actions:
				break
		return indices, None

	def select_answers_v6_diversity(self, assigned_grading_actions, pgv_values, outlier_indices_list=None):
		indices = []
		# sorted_indices = sorted(list(range(self.reduced_num_data)),key=lambda i: pgv_values[i], reverse=True)
		# if self.version == 6 and self.version_variant == '0825B2' and len(self.reduced_known_data_labels) > 0:
		if self.diverse_answer_queries_outlier_counts and len(self.reduced_known_data_labels) > 0:
			outlier_counts = np.zeros(shape=self.reduced_num_data)
			for s in range(len(self.subspaces)):
				if self.subspaces[s].weight <= 0:
					continue
				outlier_indices = outlier_indices_list[s]
				for outlier_index in outlier_indices:
					outlier_counts[outlier_index] += self.subspaces[s].weight
			sorted_indices = sorted(list(range(self.reduced_num_data)),
				key=lambda i: (-outlier_counts[i], -pgv_values[i]))
		else:
			sorted_indices = sorted(list(range(self.reduced_num_data)), key=lambda i: pgv_values[i], reverse=True)
		sorted_indices = [i for i in sorted_indices if i not in self.reduced_known_data_labels]
		
		for index in copy.copy(sorted_indices):
			comparing_indices = indices if not self.diverse_answer_queries_all \
				else indices + list(self.reduced_known_data_labels.keys())

			# if len(indices) <= 0:
			if len(comparing_indices) <= 0:
				indices.append(index)
				sorted_indices.remove(index)
			else:
				# min_distance = np.min(self.reduced_data_distances[index][indices])
				min_distance = np.min(self.reduced_data_distances[index][comparing_indices])
				if min_distance >= self.rd_cutoff:
					indices.append(index)
					sorted_indices.remove(index)
			if len(indices) >= assigned_grading_actions:
				break

		if len(indices) < assigned_grading_actions:
			indices += sorted_indices[:assigned_grading_actions - len(indices)]
		return indices, None

	# def select_answers_v6(self, assigned_grading_actions, informations):
	# 	indices= []
	# 	selected_indices = list(self.reduced_known_data_labels.keys())

	# 	full = False
	# 	while True:
	# 		for qs in self.query_status_list:
	# 			if qs.status == 'selected' or qs.status == 'skipped':
	# 				continue
	# 			excluded = self.is_excluded(qs.index, selected_indices + indices)
	# 			if not excluded:
	# 				qs.status = 'consider'
	# 				indices.append(qs.index)
	# 				full = len(indices) >= assigned_grading_actions
	# 				if full:
	# 					break
	# 			else:
	# 				qs.status = 'skipped'

	# 		if full or not self.deduct_exclusion_rd():
	# 			break
		
	# 	return indices, None

	def select_answers(self, assigned_grading_actions, informations, outlier_indices_list):
		if self.version <= 2:
			return self.select_answers_v2(assigned_grading_actions, informations)
		elif self.version == 5.5:
			return self.select_answers_v5_5(assigned_grading_actions, informations)
		elif self.version == 6:
			# if self.version_variant == '0812A' or self.version_variant == '0812E' \
			# 	or self.version_variant == '0825A' or self.version_variant == '0825B':
			# if self.version_variant in ['0812A', '0812E', '0825A', '0825B', '0825B2', '0825C',
			# 	'0825D'] or '0903' in self.version_variant:
			if self.diverse_answer_queries:
				return self.select_answers_v6_diversity(assigned_grading_actions, informations,
					outlier_indices_list)
			else:
				return self.select_answers_v6(assigned_grading_actions, informations)
		else:
			return self.select_answers_v3(assigned_grading_actions, informations)			

	def get_valuable_answer_indices(self, assigned_grading_actions):
		label_list, outlier_indices_list, marginalness_list = [], [], []
		local_nearest_ground_truth_distance_list, local_nearest_ground_truth_index_map_list = [], []
		pgv_value_list = []

		if self.dynamic_grade_weights:
			distributions = dict()
			for k, v in self.reduced_known_data_labels.items():
				if not v in distributions:
					distributions[v] = 1
				else:
					distributions[v] += 1
			for k, v in distributions.items():
				self.possible_grades_weights[self.possible_grades.index(k)] = v / len(self.known_data_labels)

		for subspace in self.subspaces:
			subspace.possible_grades_weights = self.possible_grades_weights
			if self.version == 6:
				labels, outlier_indices, nearest_ground_truth_distances, nearest_ground_truth_index_map, \
					marginalness, pgv_values = subspace.calculate_values(self.graded_answers, self.labels)
				marginalness_list.append(marginalness)
				pgv_value_list.append(pgv_values)
			else:
				labels, outlier_indices, nearest_ground_truth_distances, nearest_ground_truth_index_map, \
					marginalness = subspace.calculate_values(self.graded_answers, self.labels)
				marginalness_list.append(marginalness / marginalness.max() \
					if marginalness.max() != 0 else marginalness)
			label_list.append(labels)
			outlier_indices_list.append(outlier_indices)
			local_nearest_ground_truth_distance_list.append(nearest_ground_truth_distances)
			local_nearest_ground_truth_index_map_list.append(nearest_ground_truth_index_map)

		# if self.version == 5 or self.version == 5.5:
		if self.relevant_subspaces is not None:
			nearest_ground_truth_distances = self.calculate_relevant_weighted_average_subspace_values(
				local_nearest_ground_truth_distance_list)
		# elif self.version == 3 or self.version == 4:
		elif self.version >= 3:
			nearest_ground_truth_distances = self.calculate_weighted_average_subspace_values(
				local_nearest_ground_truth_distance_list)
		else:
			nearest_ground_truth_distances = np.mean(local_nearest_ground_truth_distance_list, axis=0)
		
		if len(self.reduced_known_data_labels) > 0 or self.version == 6:
			adjusted_marginalness_list = pgv_value_list if self.version == 6 else marginalness_list
			if self.relevant_subspaces is not None:
				# local_marginalness = self.calculate_relevant_weighted_average_subspace_values(marginalness_list)
				local_marginalness = self.calculate_relevant_weighted_average_subspace_values(adjusted_marginalness_list)
			# elif self.version == 3 or self.version == 4:
			elif self.version >= 3:
				# local_marginalness = self.calculate_weighted_average_subspace_values(marginalness_list)
				local_marginalness = self.calculate_weighted_average_subspace_values(adjusted_marginalness_list)
			else:
				# local_marginalness = np.mean(marginalness_list, axis=0)
				local_marginalness = np.mean(adjusted_marginalness_list, axis=0)

			if len(self.reduced_known_data_labels) > 0:
				if self.relevant_subspaces is not None:
					label_list = np.array(label_list)
					relevant_label_list = np.zeros(shape=self.relevant_subspaces.shape, dtype=label_list.dtype)
					for i in range(self.reduced_num_data):
						relevant_label_list[i] = label_list[self.relevant_subspaces[i], i]
					relevant_label_list = np.transpose(relevant_label_list)
					global_marginalness = self.calculate_global_marginalness(relevant_label_list)
				else:
					global_marginalness = self.calculate_global_marginalness(label_list)
				global_marginalness = global_marginalness / global_marginalness.max() \
					if global_marginalness.max() != 0 else global_marginalness
			else:
				global_marginalness = np.zeros(shape=self.reduced_num_data)
			
			# informations = (local_marginalness + global_marginalness) / 2
			if self.version == 6:
				informations = local_marginalness + global_marginalness
			else:
				informations = (local_marginalness + global_marginalness) / 2
		else:
			global_marginalness = np.zeros(shape=self.reduced_num_data)
			informations = np.zeros(shape=self.reduced_num_data)

		indices, selection_param_dict = self.select_answers(assigned_grading_actions, informations,
			outlier_indices_list)
		self.remaining_grading_actions -= len(indices)
		self.current_iteration += 1
		return self.get_original_data_indices(indices, first_only=True), indices, \
			selection_param_dict, informations, nearest_ground_truth_distances, \
			local_nearest_ground_truth_distance_list, local_nearest_ground_truth_index_map_list, \
			marginalness_list, global_marginalness, outlier_indices_list

	def update_graded_answers(self, indices):
		for index in indices:
			query_status = next((qs for qs in self.query_status_list if qs.index == index), None)
			if query_status is not None:
				query_status.status = 'selected'

			graded_answer = next((g for g in self.graded_answers \
				if g.should_merge(index, self.pairwise_relations)), None)
			if graded_answer is None:
				graded_answer = GradedAnswer(cluster_id=len(self.graded_answers) + 1, core_index=index)
				self.graded_answers.append(graded_answer)
			else:
				graded_answer.indices.append(index)
			self.labels[index] = graded_answer.cluster_id

			# if len(self.graded_answers) >= self.grading_actions:
			# 	break

	def get_original_data_indices(self, reduced_data_indices, first_only=False):
		data_indices = []
		for index in reduced_data_indices:
			indices = np.where(self.original_to_reduced_index_map == index)[0].tolist()
			if first_only:
				data_indices.append(indices[0])
			else:
				data_indices += indices
		return data_indices

	def calculate_weighted_average_subspace_values(self, value_list):
		total_weights, values = 0, None
		for s in range(len(self.subspaces)):
			weight = self.subspaces[s].weight
			v = value_list[s] * weight
			values = v if s == 0 else np.add(values, v)
			total_weights += weight
		values /= total_weights
		return values

	def calculate_relevant_weighted_average_subspace_values(self, value_list):
		total_weights = 0
		values = np.zeros(shape=value_list[0].shape)
		for i in range(len(values)):
			if values.ndim == 1:
				total_weights = 0
				for subspace_index in self.relevant_subspaces[i]:
					weight = self.subspaces[subspace_index].weight
					values[i] += value_list[subspace_index][i] * weight
					total_weights += weight
				values[i] /= total_weights
			else:
				for j in range(i+1, len(values[i])):
					total, total_weights = 0, 0
					for subspace_index in self.relevant_subspaces[i]:
						weight = self.subspaces[subspace_index].weight
						values[i][j] += value_list[subspace_index][i][j] * weight
						total_weights += weight
					values[i][j] /= total_weights
					values[j][i] = values[i][j]
		return values
	
	def calculate_densities(self):
		density_list = [s.densities / s.densities.max() for s in self.subspaces]
		# if self.version == 5 or self.version == 5.5:
		if self.relevant_subspaces is not None:
			self.densities = self.calculate_relevant_weighted_average_subspace_values(density_list)
		elif self.version > 3:
			self.densities = self.calculate_weighted_average_subspace_values(density_list)
		else:
			self.densities = np.mean(density_list, axis=0)
		self.densities = np.log(self.densities) / np.log(math.exp(1))
		self.densities -= self.densities.min()
		if self.densities.max() > 0:
			self.densities /= self.densities.max()
		else:
			raise Exception('{} Densities.max <= 0'.format(str(self)))

	def calculate_spaciousness(self):
		# if self.version == 5 or self.version == 5.5:
		if self.relevant_subspaces is not None:
			# self.spaciousness = self.calculate_relevant_weighted_average_subspace_values(
			# 	[s.spaciousness for s in self.subspaces])
			self.spaciousness = self.calculate_relevant_weighted_average_subspace_values(
				[s.normalized_spaciousness for s in self.subspaces])
		# elif self.version == 3 or self.version == 4:
		elif self.version > 3:
			# self.spaciousness = self.calculate_weighted_average_subspace_values(
			# 	[s.spaciousness for s in self.subspaces])
			self.spaciousness = self.calculate_weighted_average_subspace_values(
				[s.normalized_spaciousness for s in self.subspaces])
		else:
			# self.spaciousness = np.mean([s.spaciousness for s in self.subspaces], axis=0)
			self.spaciousness = np.mean([s.normalized_spaciousness for s in self.subspaces], axis=0)

	def calculate_global_marginalness(self, label_list):
		keys = []
		for labels in label_list:
			keys += list(np.unique(labels))
		keys = list(set(keys))

		count_labels = np.zeros(shape=(self.reduced_num_data, len(keys)), dtype=int)
		for labels in label_list:
			for i in range(self.reduced_num_data):
				count_labels[i][keys.index(labels[i])] += 1
		
		u_g = np.zeros(shape=self.reduced_num_data)
		probabilities = count_labels / len(self.subspaces)
		for i in range(self.reduced_num_data):
			# u_g[i] = -1 * sum([p * math.log2(p) for p in probabilities[i] if p != 0])
			if not self.weighted_global_entropy:
				u_g[i] = -1 * sum([p * math.log2(p) for p in probabilities[i] if p != 0])
			else:
				for k in range(len(keys)):
					p = probabilities[i][k]
					if p == 0:
						continue
					u_g[i] += self.possible_grades_weights[k] * p * math.log2(p)
				u_g[i] *= -1
		return u_g
