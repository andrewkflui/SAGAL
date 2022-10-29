import math, copy
from enum import Enum

import numpy as np
from sklearn import decomposition
from sklearn.mixture import GaussianMixture

from core import utils
from basis import config

class GALSubspace():
	def __init__(self, data, encoder, model, **kwargs):
		self.id = None
		self.data = data
		self.encoder = encoder
		self.model = model
		self.compress_method = None
		self.weight = 1.

		self.possible_grades = None
		self.possible_grades_weights = None
		self.default_grade = None
		self.random_selections = None
		self.random_selection_method = None
		self.horizontal_random_selection_number = None

		self.distance_function = config.DISTANCE_FUNCTION
		self.epsilon = config.EPSILON

		self.rd_cutoff = None
		self.rd_deriving_factor = 0.5
		self.rd_cutoff_window_size = config.RD_CUTOFF_WINDOW_SIZE
		self.default_rd_percentile_cutoff = config.DEFAULT_RD_PERCENTILE_CUTOFF
		self.marginalness_neighbourhood_size = config.MARGINALNESS_NEIGHBOURHOOD_SIZE
		self.marginalness_rd_factor = config.MARGINALNESS_RD_FACTOR
		self.spaciousness_neighbour_number = config.SPACIOUSNESS_NEIGHBOUR_NUMBER
		self.grade_assignment_method = config.GRADE_ASSIGNMENT_METHOD
		self.voting_version = None

		self.label_searching_boundary_factor = config.LABEL_SEARCHING_BOUNDARY_FACTOR
		self.label_knn = config.LABEL_KNN
		self.delta_link_threshold_factor = None
		
		self.ranked_spaciousness = None
		self.delta_link_threshold_window_size_factor = None
		self.delta_link_threshold_window_size = None
		self.delta_link_threshold_start_position = None
		self.delta_link_threshold_curvature = None
		self.delta_link_threshold = None

		self.outlier_removal = False
		self.outlier_density_condition = config.OUTLIER_DENSITY_CONDITION
		self.outlier_spaciousness_condition = config.OUTLIER_SPACIOUSNESS_CONDITION
		self.outlier_nearest_true_grade_condition = config.OUTLIER_NEAREST_TRUE_GRADE_CONDITION

		self.version = 1
		self.version_variant = None

		self.normalize_distances = True
		self.dynamic_grade_weights = True
		self.pgv_use_normalized_density = True
		self.pgv_density_adjustment = None

		self.known_data_number = 0
		self.known_data_labels = dict()

		self.reduced_data_indices, self.reduced_data_index_dict = None, None

		for kw in kwargs:
			if hasattr(self, kw):
				setattr(self, kw, kwargs[kw])

		self.num_data = len(self.data)
		self.data_dim = len(self.data[0])

		self.data_distances = utils.get_data_distances(self.data, self.distance_function)
		self.reduced_data_distances = None
		
		self.densities, self.normalized_densities = None, None
		self.spaciousness, self.normalized_spaciousness = None, None
		self.max_delta, self.max_density = 0., 0.
		self.max_density_indices = []

		self.reduced_data, self.reduced_all_zero_vector_indices = [], []
		self.original_to_reduced_index_map = np.array([-1 for _ in range(self.num_data)])

		self.delta_index_map, self.delta_distances = None, None
		self.children_list = None
		self.descendant_list, self.normalized_descendant_counts = None, None
		self.predicted_total, self.predict_correct_count, self.predictive_value = 0, 0, 0.

		if self.reduced_data_indices is None:
			self.reduced_data_indices = []
			self.reduced_data_index_dict = dict()
			self.encapsulate_data(start_index=0, recalculate=True)
		else:
			self.set_reduced_data(self.reduced_data_indices, self.reduced_data_index_dict)

		self.reduced_known_data_labels = dict()
		if len(self.known_data_labels) > 0:
			self.set_known_data_labels(self.known_data_labels)

		self.labels, self.true_grade_labels = None, None
		self.confidence_level_dict = None, None

		nearest_true_grade_condition = self.outlier_nearest_true_grade_condition.replace(
			'RD', 'self.rd_cutoff')
		self.outlier_condition = 'self.normalized_densities[{index}] {density}'
		self.outlier_condition += ' and self.normalized_spaciousness[{index}] {spaciousness}'
		self.outlier_condition += ' and nearest_ground_truth_distances[{index}] {true_grade}'
		self.outlier_condition = self.outlier_condition.format(index='{index}', density=self.outlier_density_condition,
			spaciousness=self.outlier_spaciousness_condition, true_grade=nearest_true_grade_condition)

		self.confidence_version = None

	def __str__(self):
		string = 'Encoder: {}'.format(self.encoder)
		if self.model is not None:
			string += ', Model, {}'.format(self.model)
		string += ', Dimension: {}'.format(self.data_dim)
		if self.random_selections is not None:
			string += 'V' if self.random_selection_method is None \
				or self.random_selection_method == 'vertical' else 'H'
		if self.compress_method is not None:
			string += ' (' + self.compress_method.upper() + ')'
		if self.id is not None:
			string += ' ({})'.format(self.id)
		return string

	def set_attributes(self, args):
		for k, v in args.items():			
			if hasattr(self, k):
				setattr(self, k, v)

	def get_name(self):
		if self.encoder == 'google_universal_sentence_encoder':
			# name = 'GUSE {}'.format(self.data_dim)
			name = 'GUSE'
		else:
			name = self.encoder.upper()
		if self.random_selections is not None:
			# name += 'V' if self.random_selection_method is None \
			# 	or self.random_selection_method == 'vertical' else 'H'
			if self.random_selection_method == 'vertical':
				name += ' {}V'.format(self.data_dim)
			else:
				name += ' {}_{}H'.format(self.horizontal_random_selection_number \
					if self.horizontal_random_selection_number is not None else self.data_dim,
					self.data_dim)
		if self.id is not None:
			name += ' ({})'.format(self.id)
		return name

	def update_representations(self):
		if self.normalize_distances and self.reduced_data_distances.max() != 0:
			# self.reduced_data_distances /= self.max_delta
			self.reduced_data_distances /= self.reduced_data_distances.max()
		self.calculate_spaciousness()
		self.rd_cutoff, self.max_delta = self.find_rd_cutoff()
		self.calculate_densities()
		# self.calculate_spaciousness()

	def assign_cluster_id_to_unassigned_children(self, cluster_id, index, global_labels):
		indices = [c for c in self.children_list[index] if global_labels[c] <= 0]
		self.labels[indices] = cluster_id
		return indices

	def assign_roots_from_nearest_true_grade(self, labels):
		for index in self.max_density_indices:
			if self.labels[index] != 0:
				continue
			
			# assign grade to answer of density = max from nearest true grade
			for n in self.nearest_neighbours[index][1:]:
				if n in self.reduced_known_data_labels:
					self.labels[index] = self.labels[n]
					self.labels[self.children_list[index]] = self.labels[n]
					self.source_map[index] = (n, 'nearest_true_grade', True)
					for child in self.children_list[index]:
						self.source_map[child] = (n, 'inherit', True)
					break
		return self.labels
	
	def spot_outliers(self, unassigned_indices, nearest_ground_truth_distances):
		if self.version < 3 or not self.outlier_removal:
			return []
		
		outlier_indices = []
		for index in unassigned_indices:
			condition = self.outlier_condition.format(index=index)
			eval_result = eval(condition)
			if eval(condition):
				outlier_indices.append(index)
		return outlier_indices

	def wkernal(self, index, neighbour):
		return self.densities[neighbour]

	def update_labels(self, graded_answers, global_labels, nearest_ground_truth_index_map=None,
		nearest_ground_truth_distances=None):
		self.labels = np.copy(global_labels)
		if not 'oc' in self.grade_assignment_method:
			self.labels = np.copy(global_labels)
		else:
			self.labels = np.zeros(shape=self.reduced_num_data, dtype=object)
			self.labels[:] = -1
			assigned_indices = np.where(global_labels != 0)[0]
			for index in assigned_indices:
				self.labels[index] = self.reduced_known_data_labels[index]
		self.source_map = np.empty(shape=self.reduced_num_data, dtype='object')
		
		outlier_indices = []
		clustering_result = None
		
		if len(graded_answers) <= 0:
			# return self.labels, outlier_indices, clustering_result
			return self.labels, outlier_indices, clustering_result

		for i in range(self.reduced_num_data):
			if self.labels[i] != 0:
				self.source_map[i] = (i, 'self', True)
		
		# assign answers of max density if not yet graded
		self.labels = self.assign_roots_from_nearest_true_grade(self.labels)

		index_dict = dict()
		for ga in graded_answers:
			for i in ga.indices:
				index_dict[i] = dict()
				# index_dict[i]['cluster_id'] = ga.cluster_id
				index_dict[i]['cluster_id'] = ga.cluster_id if not 'oc' in self.grade_assignment_method \
					else self.get_graded_answer_true_grade(ga)
				index_dict[i]['type'] = 'core' if i == ga.core_index else 'normal'

		sorted_indices = sorted(list(index_dict.keys()), key=lambda k: self.densities[k], reverse=True)
		for index in sorted_indices:
			cluster_id = index_dict[index]['cluster_id']
			if nearest_ground_truth_index_map is None:
				# ancestor = self.find_unassigned_ancestor(index, labels=self.labels) or index
				ancestor = index
				self.labels[ancestor] = cluster_id
				# children_indices = self.assign_cluster_id_to_unassigned_children(cluster_id, ancestor, global_labels)
				self.source_map[ancestor] = (index,
					'self' if ancestor == index else 'inherit', ancestor in self.reduced_known_data_labels)
				# self.source_map[children_indices] = (index, 'inherit', True)
			else:
				self.labels[index] = cluster_id
				self.source_map[index] = (index, 'self', True)

		unassigned_indices = np.where(self.labels == 0)[0].tolist()
		unassigned_indices = sorted(unassigned_indices, key=lambda k: self.densities[k], reverse=True)

		if self.version > 3 and self.outlier_removal:
			outlier_indices = self.spot_outliers(unassigned_indices, nearest_ground_truth_distances)
			self.labels[outlier_indices] = -1
			for index in outlier_indices:
				self.labels[index] = -1
				unassigned_indices.remove(index)

		if self.grade_assignment_method == 'gaussian_mixture':
			reduced_known_data_indices = list(self.reduced_known_data_labels.keys())
			means = self.reduced_data[reduced_known_data_indices]
			gaussian_mixture = GaussianMixture(n_components=len(self.reduced_known_data_labels),
				covariance_type='full', max_iter=1, means_init=means, random_state=0)
			result = gaussian_mixture.fit_predict(self.reduced_data)
			predicted_means = gaussian_mixture.means_
			
			assigned_count = 0
			for index in reduced_known_data_indices:
				member_indices = np.where(result == result[index])[0]
				for member_index in member_indices:
					if self.labels[member_index] == 0:
						self.labels[member_index] = self.labels[index]
						self.source_map[member_index] = (index, 'clustering', False)
						assigned_count += 1
		elif self.grade_assignment_method == 'oc' or self.grade_assignment_method == 'moc' \
			or self.grade_assignment_method == 'soc':
			labels = np.copy(self.labels)
			labels[labels == -1] = ''
			unique_labels = np.unique(labels).tolist()
			if '' in unique_labels:
				unique_labels.remove('')
			unique_labels_count = len(unique_labels)
			
			graded_answer_indices = [i for i in range(self.reduced_num_data) if self.labels[i] != -1]
			sorted_graded_answer_indices = sorted(graded_answer_indices,
				key=lambda g: (-self.normalized_densities[g], self.normalized_spaciousness[g]))

			search_boundary = self.rd_cutoff * self.label_searching_boundary_factor
			unassigned_indices = np.where(self.labels == -1)[0].tolist()
			unassigned_indices = sorted(unassigned_indices,
				key=lambda u: (-self.normalized_densities[u], self.normalized_spaciousness[u]))

			if self.grade_assignment_method == 'oc':
				# assign grades to unassigned if distance  <= label_search_boundary
				for index in copy.copy(unassigned_indices):
					for ga_index in sorted_graded_answer_indices:
						if self.reduced_data_distances[index][ga_index] > search_boundary:
							break
						self.labels[index] = self.labels[ga_index]
						self.source_map[index] = (ga_index, 'oc', ga_index in self.reduced_known_data_labels)
						unassigned_indices.remove(index)
						break
			elif self.grade_assignment_method == 'moc':
				# assign grades to unassigned by nn ratio
				for index in copy.copy(unassigned_indices):
					j_indices = [None] * unique_labels_count
					k_indices = [None] * unique_labels_count
					dj_distances = np.zeros(shape=unique_labels_count)
					dk_distances = np.zeros(shape=unique_labels_count)
					nn_ratios = np.zeros(shape=unique_labels_count)

					dj_distances[:] = np.nan
					dk_distances[:] = np.nan
					nn_ratios[:] = np.nan
					
					for j in self.nearest_neighbours[index][1:]:
						if self.reduced_data_distances[index][j] > search_boundary:
							break

						if self.labels[j] == -1:
							continue
						
						# find dj for earch label
						position = unique_labels.index(self.labels[j])
						if not np.isnan(dj_distances[position]):
							continue
						dj_distances[position] = self.reduced_data_distances[index][j]
						j_indices[position] = j

						# find dk where label of dk = label of dj
						for k in self.nearest_neighbours[j][1:]:
							# if self.reduced_data_distances[j][k] > search_boundary:
							if self.reduced_data_distances[index][k] > search_boundary:
								break
							if self.labels[k] != self.labels[j]:
								continue 

							dk_distances[position] = self.reduced_data_distances[j][k]
							ratio = dj_distances[position] / dk_distances[position]
							if ratio < 1:
								nn_ratios[position] = ratio
								k_indices[position] = k
							break
						
						if None not in j_indices:
							break

					j_exists_positions = [j for j in range(unique_labels_count) if j_indices[j] is not None]
					k_exists_positions = [k for k in range(unique_labels_count) if k_indices[k] is not None]
					
					if len(j_exists_positions) <= 0:
						# leave as outlier if no j found
						continue

					if len(k_exists_positions) <= 0:
						# no k exists, assign to label of shorter dj
						position = np.argmin(dj_distances[j_exists_positions])
						source = np.array(j_indices)[j_exists_positions][position]
						self.labels[index] = self.labels[source]
						self.source_map[index] = (source, 'dj', source in self.reduced_known_data_labels)
						unassigned_indices.remove(index)
					else:
						position = np.nanargmin(nn_ratios)
						source = k_indices[position]
						self.labels[index] = self.labels[source]
						self.source_map[index] = (source, 'dk', source in self.reduced_known_data_labels)
						unassigned_indices.remove(index)
			elif self.grade_assignment_method == 'soc':
				for index in copy.copy(unassigned_indices):
					max_value_label, source, neighbour_label_dict = None, None, dict()
					for n in self.nearest_neighbours[index][1:self.label_knn + 1]:
						label_n = self.labels[n]
						if not label_n in neighbour_label_dict:
							neighbour_label_dict[label_n] = 0
						if label_n == -1:
							neighbour_label_dict[label_n] += ((self.wkernal(index, n) / self.reduced_data_distances[index][n]) * 0.1)
						else:
							neighbour_label_dict[label_n] += (self.wkernal(index, n) / self.reduced_data_distances[index][n])
						if max_value_label is None or neighbour_label_dict[max_value_label] < neighbour_label_dict[label_n]:
							max_value_label = label_n
							source = n
					self.labels[index] = max_value_label
					self.source_map[index] = (source, 'knn', False)
					if max_value_label != -1:
						unassigned_indices.remove(index)
			
			for index in unassigned_indices:
				self.source_map[index] = (index, 'outlier', False)
		else:
			if self.grade_assignment_method == 'parent_breaks':
				unassigned_indices = sorted(unassigned_indices,
					key=lambda u: (-self.normalized_densities[u], self.normalized_spaciousness[u]))

			for index in unassigned_indices:
				if self.labels[index] != 0:
					continue

				if self.grade_assignment_method == 'parent':
					parent = self.delta_index_map[index]
					# self.labels[index] = self.get_parent_label(index)
					ancestor, self.labels[index] = self.get_parent_label(index)
					self.source_map[index] = (ancestor, 'inherit', ancestor in self.reduced_known_data_labels)
					if self.labels[index] == 0 and self.densities[index] == self.densities[parent]:
						self.labels[index] = self.labels[nearest_ground_truth_index_map[index]]
						self.source_map[index] = (nearest_ground_truth_index_map[index],
							'nearest_true_grade', True)
				elif self.grade_assignment_method == 'parent_breaks':
					parent = self.delta_index_map[index]
					# self.labels[index] = self.get_parent_label(index)
					ancestor, label = self.get_parent_label(index)
					if label > 0:
						ancestor = self.source_map[ancestor][0]
						source_type = 'inherit'
					elif label == 0 and self.densities[index] == self.densities[parent]:
						ancestor = nearest_ground_truth_index_map[index]
						label = self.labels[ancestor]
						source_type = 'nearest_true_grade'
					
					# if label == 0 or self.reduced_data_distances[index][ancestor] \
					# 	> self.rd_cutoff * self.delta_link_threshold_factor:
					if label == 0 or self.reduced_data_distances[index][ancestor] > self.delta_link_threshold:
						ancestor = index
						label = -1
						source_type = 'outlier'
						outlier_indices.append(index)
					
					self.labels[index] = label
					# self.source_map[index] = (ancestor, source_type, ancestor in self.reduced_known_data_labels)
					self.source_map[index] = (ancestor, source_type,
						parent == ancestor and ancestor in self.reduced_known_data_labels)
				elif self.grade_assignment_method == 'nearest_true_grade':
					# assign from nearest true grade
					self.labels[index] = self.labels[nearest_ground_truth_index_map[index]]
					self.source_map[index] = (nearest_ground_truth_index_map[index],
						'nearest_true_grade1', True)
		return self.labels, outlier_indices, clustering_result

	def get_graded_answer_true_grade(self, graded_answer):
		for index in graded_answer.indices:
			if index in self.reduced_known_data_labels:
				return self.reduced_known_data_labels[index]
		return None	

	def update_true_grade_labels(self, graded_answers, global_labels):
		# if self.grade_assignment_method != 'oc' and self.grade_assignment_method != 'moc':
		if not 'oc' in self.grade_assignment_method:
			self.true_grade_labels = np.empty(shape=self.reduced_num_data, dtype=object)
			for ga in graded_answers:
				true_grade = self.get_graded_answer_true_grade(ga)
				indices = np.where(self.labels == ga.cluster_id)[0].tolist()
				self.true_grade_labels[indices] = true_grade
		else:
			self.true_grade_labels = np.copy(self.labels)

		self.true_grade_labels[self.labels == -1] = self.default_grade

		distributions = dict()
		for t in self.true_grade_labels:
			if not t in distributions:
				distributions[t] = 0
			distributions[t] += 1

		unique_labels = np.unique(self.true_grade_labels).tolist()
		self.calculate_confidence_levels(keys=unique_labels)
		return unique_labels, self.true_grade_labels, self.confidence_level_dict, \
			self.calculate_marginalness(labels=self.true_grade_labels)

	def calculate_predictive_value(self, graded_answers, global_labels):
		previous_true_grade_labels = np.copy(self.true_grade_labels)
		nearest_ground_truth_distances, nearest_ground_truth_index_map = self.get_nearest_ground_truths()
		labels, outlier_indices, clustering_result = self.update_labels(graded_answers, global_labels,
			nearest_ground_truth_index_map, nearest_ground_truth_distances)
		self.update_true_grade_labels(graded_answers, global_labels)
		correct_count = len(np.where(previous_true_grade_labels == self.true_grade_labels)[0])
		self.predictive_value = float(correct_count / self.reduced_num_data)
		return self.predictive_value, correct_count

	def get_original_data_indices(self, reduced_data_indices):
		data_indices = []
		for index in reduced_data_indices:
			data_indices += np.where(self.original_to_reduced_index_map == index)[0].tolist()
		return data_indices

	def get_data_representation_number(self, data_index, version):
		if version == 1:
			return 1
		else:
			dw = len(self.reduced_data_index_dict[data_index])
			if version == 2:
				return dw
			elif version == 3:
				return math.log(dw, 2) + 1
			elif version == 4:
				return math.log(dw) + 1
			else:
				raise Exception('Invalid DW Version')

	def get_next_highest_index(self, i):
		next_highest_index = None
		if self.densities[i] == self.max_density:
			next_highest_index = i
		else:
			for j in self.nearest_neighbours[i][1:]:
				if self.densities[i] <= self.densities[j]:
					next_highest_index = j
					break
		return next_highest_index

	def get_max_distance(self):
		if self.distance_function == 'angular':
			return math.pi / 2
		else:
			return  self.max_delta

	def get_parent_label_with_searched_list(self, index, searched=None, start=None):
		parent = self.delta_index_map[index]
		searched = [] if searched is None else searched
		start = index if start is None else start
		searched.append(index)
		# if index == self.delta_index_map[parent]:
		if index == self.delta_index_map[parent] and self.labels[parent] <= 0:
			searched.append(parent)
			return 0, searched
		if self.labels[parent] > 0:
			searched.append(parent)
			return self.labels[parent], searched
		return self.get_parent_label_with_searched_list(parent, searched, start)

	def get_parent_label(self, index):
		parent = self.delta_index_map[index]
		# if index == self.delta_index_map[parent]:
		if index == self.delta_index_map[parent] and self.labels[parent] <= 0:
			# return 0
			return -1, 0
		if self.labels[parent] > 0:
			# return self.labels[parent]
			return parent, self.labels[parent]
		return self.get_parent_label(parent)

	def get_parent_true_grade_label(self, index):
		parent = self.delta_index_map[index]
		if index == self.delta_index_map[parent]:
			# return None
			return ''
		if self.true_grade_labels[parent] is not None and self.true_grade_labels[parent] != '':
			return self.true_grade_labels[parent]
		return self.get_parent_true_grade_label(parent)

	def find_unassigned_ancestor(self, index, labels, searched=None):
		parent = self.delta_index_map[index]
		searched = [] if searched is None else searched
		if parent in searched:
			return index
		if labels[parent] != 0:
			return index
		searched.append(index)
		return self.find_unassigned_ancestor(parent, labels, searched)

	# def get_descendants(self, index, descendants=None):
	# 	descendants = [] if descendants is None else descendants
	# 	if len(self.children_list[index]) <= 0:
	# 		return []
	# 	for children in self.children_list[index]:
	# 		if children == index:
	# 			continue
	# 		descendants += self.get_descendants(children)
	# 	return descendants

	def add_data(self, new_data, encapsulate=True, recalculate=True):
		original_num_data = self.num_data
		self.data = np.append(self.data, new_data, axis=0)
		self.num_data = len(self.data)

		# extend arrays
		new_data_len = len(new_data)
		self.original_to_reduced_index_map = np.append(self.original_to_reduced_index_map,
			np.array([-1 for _ in range(new_data_len)]))
		self.delta_index_map = np.append(self.delta_index_map, np.zeros(shape=new_data_len))
		self.delta_distances = np.append(self.delta_distances, np.zeros(shape=new_data_len))
		self.densities = np.append(self.densities, np.zeros(shape=new_data_len))
		self.normalized_densities = np.append(self.normalized_densities, shape=new_data_len)
		self.data_distances = utils.get_data_distances(self.data, self.distance_function)
		
		if encapsulate:
			self.encapsulate_data(start_index=original_num_data)
		else:
			if not recalculate:
				for i in range(original_num_data, self.num_data):
					self.reduced_data_indices.append(i)
					self.original_to_reduced_index_map[i] = i
			self.update_reduced_data(recalculate=recalculate)

	def update_reduced_data(self, recalculate=True):
		self.reduced_num_data = len(self.reduced_data_indices)
		self.reduced_data_index_dict = dict()
		self.reduced_data_distances = np.zeros(shape=(self.reduced_num_data, self.reduced_num_data))

		# Find nearest neighbours
		self.reduced_data = []
		self.nearest_neighbours = []
		self.nearest_neighbours_mean_distances = []

		for i in range(self.reduced_num_data):
			index_i = self.reduced_data_indices[i]
			self.reduced_data.append(self.data[index_i])

			indices = np.where(self.original_to_reduced_index_map == i)[0]
			self.reduced_data_index_dict[i] = indices

		self.reduced_data = np.array(self.reduced_data)
		self.reduced_data_distances = utils.get_data_distances(self.reduced_data, self.distance_function)
		self.reduced_data_distances[np.isnan(self.reduced_data_distances)] = self.get_max_distance()

		if self.encoder == 'tfidf':
			self.reduced_all_zero_vector_indices = np.where(np.all(self.reduced_data == 0, axis=1))[0].tolist()
		else:
			self.reduced_all_zero_vector_indices = []

		self.max_delta = self.reduced_data_distances.max()
		self.reduced_data_distances[np.isnan(self.reduced_data_distances)] = self.max_delta
		
		if self.normalize_distances:
			self.reduced_data_distances /= self.max_delta

		# Build NN Table, make self always the nearest (first)
		for i in range(self.reduced_num_data):
			nn = [(j, self.reduced_data_distances[i][j]) for j in range(self.reduced_num_data) if j != i]
			self.nearest_neighbours.append([i] + [v[0] for v in sorted(nn, key=lambda v: v[1])])

		self.calculate_spaciousness()
		self.rd_cutoff, _ = self.find_rd_cutoff()

		if recalculate:
			self.calculate_densities()
			# self.calculate_spaciousness()

	def encapsulate_data(self, start_index=0, recalculate=True):
		for i in range(start_index, self.num_data):
			if i not in self.known_data_labels:
				for j in range(i if start_index == 0 else 0, self.num_data):
					if self.data_distances[i][j] <= self.epsilon:
						if self.original_to_reduced_index_map[i] < 0 and self.original_to_reduced_index_map[j] < 0:
							index = len(self.reduced_data_indices)
							self.original_to_reduced_index_map[i] = index
							self.original_to_reduced_index_map[j] = index
							self.reduced_data_indices.append(i)
						elif self.original_to_reduced_index_map[i] < 0 and self.original_to_reduced_index_map[j] >= 0:
							self.original_to_reduced_index_map[i] = self.original_to_reduced_index_map[j]
						elif self.original_to_reduced_index_map[i] >= 0 and self.original_to_reduced_index_map[j] < 0:
							self.original_to_reduced_index_map[j] = self.original_to_reduced_index_map[i]

			if self.original_to_reduced_index_map[i] < 0:
				index = len(self.reduced_data_indices)
				self.original_to_reduced_index_map[i] = index
				self.reduced_data_indices.append(i)
		
		self.update_reduced_data(recalculate=recalculate)

	def set_reduced_data(self, reduced_data_indices, reduced_data_index_dict):
		self.reduced_data_indices = copy.copy(reduced_data_indices)
		self.reduced_data_index_dict = copy.copy(reduced_data_index_dict)
		for k, v in reduced_data_index_dict.items():
			self.original_to_reduced_index_map[v] = k
		self.update_reduced_data(recalculate=True)

	def find_rd_cutoff(self):
		distances = []
		for i in range(self.reduced_num_data):
			for j in range(i+1, self.reduced_num_data):
				distances.append(self.reduced_data_distances[i][j])
		distances = np.sort(distances)

		# if self.rd_cutoff is not None:
		# 	return self.rd_cutoff, np.nanmax(distances)
		
		# if self.rd_deriving_factor is None:
		if self.rd_cutoff is None and self.rd_deriving_factor is None:
			position = None
			gradients = np.gradient(distances)
			count = 0
			for g in range(1, len(gradients)):
				if gradients[g] == 0.:
					continue
				
				if gradients[g] <= gradients[g-1]:
					if position is None:
						position = g
					count += 1
					if count >= self.rd_cutoff_window_size:
						break
				elif gradients[g] > gradients[g-1]:
					position = None
					count = 0
			self.rd_percentile_cutoff = position / len(gradients) \
				if position is not None else self.default_rd_percentile_cutoff
			position = int((len(distances) - 1) * self.rd_percentile_cutoff)
			self.rd_cutoff_position = position
			return distances[position], np.nanmax(distances)
		else:
			spaciousness = np.array(sorted(self.spaciousness))
			ranks = np.array(list(range(self.reduced_num_data)))

			if self.rd_cutoff is not None:
				for s in range(spaciousness.size):
					if spaciousness[s] >= self.rd_cutoff:
						self.rd_cutoff_position = s
						break
			else:
				self.rd_cutoff_position = int(ranks[-1] / 2)

			# find label_searching_rd
			if self.grade_assignment_method == 'parent_breaks' and self.delta_link_threshold is None:
				if self.delta_link_threshold_factor is None:
					window_size = int(self.reduced_num_data * self.delta_link_threshold_window_size_factor)
					window_size_median = int(window_size / 2)
					start_position, max_curvature = None, None
					for i in range(self.rd_cutoff_position, len(spaciousness)):
						if i + window_size >= len(spaciousness):
							break
						differences = np.diff(spaciousness[[i, i + window_size_median, i + window_size]])
						curvature = differences[1] - differences[0]

						if start_position is None or max_curvature < curvature:
							start_position = i
							max_curvature = curvature
							self.delta_link_threshold_curvature = curvature

					if start_position is None:
						start_position = self.rd_cutoff_position
						window_size = len(spaciousness) - self.rd_cutoff_position
					self.delta_link_threshold_window_size = window_size
					
					self.delta_link_threshold_start_position = start_position
					self.delta_link_threshold = spaciousness[start_position + int(window_size / 2)]
				else:
					if self.rd_cutoff is None:
						self.delta_link_threshold = spaciousness[self.rd_cutoff_position] * self.delta_link_threshold_factor
					else:
						self.delta_link_threshold = self.rd_cutoff * self.delta_link_threshold_factor
					self.delta_link_threshold_window_size = self.delta_link_threshold_start_position = None

			self.ranked_spaciousness = spaciousness
			self.spaciousness_ranks = ranks
			return spaciousness[self.rd_cutoff_position], np.nanmax(distances)
	
	def calculate_densities(self):
		self.densities = np.zeros(shape=self.reduced_num_data)

		for i in range(self.reduced_num_data):
			for n in self.nearest_neighbours[i]:
				distance = self.reduced_data_distances[i][n]
				if distance > self.rd_cutoff:
					break
				self.densities[i] += math.exp(-1 * (math.pow(distance, 2) / math.pow(self.rd_cutoff, 2))) * self.get_data_representation_number(n, 2)
		
		self.max_density = self.densities.max()

		# normalize densities
		self.normalized_densities = self.densities / self.max_density if self.max_density > 1 else self.densities
		self.normalized_densities = np.log(self.normalized_densities) / np.log(math.exp(1))
		self.normalized_densities -= self.normalized_densities.min()
		if self.normalized_densities.max() > 0:
			self.normalized_densities /= self.normalized_densities.max()
		self.max_density_indices = []
		self.delta_index_map = np.zeros(shape=self.reduced_num_data, dtype=int)
		self.delta_distances = np.zeros(shape=self.reduced_num_data)
		self.children_list = [[] for _ in range(self.reduced_num_data)]

		for i in range(self.reduced_num_data):
			self.delta_index_map[i] = self.get_next_highest_index(i)
			self.delta_distances[i] = self.reduced_data_distances[i][self.delta_index_map[i]]
			self.delta_distances[i] = self.reduced_data_distances[i][self.delta_index_map[i]] \
				if self.delta_index_map[i] != i else 1.

			self.children_list[self.delta_index_map[i]].append(i)
			if self.densities[i] == self.max_density:
				self.max_density_indices.append(i)

		self.descendant_list = [[] for _ in range(self.reduced_num_data)]
		self.normalized_descendant_counts = np.zeros(shape=self.reduced_num_data)
		for i in range(self.reduced_num_data):
			self.descendant_list[i] = self.get_descendants(i)
			self.normalized_descendant_counts[i] = math.log(len(self.descendant_list[i]))
		if self.normalized_descendant_counts.max() > 0:
			self.normalized_descendant_counts /= self.normalized_descendant_counts.max()

		if self.delta_distances.max() > 0:
			self.delta_distances /= self.max_delta

	def get_descendants(self, index, descendants=None):
		descendants = [index] if descendants is None else descendants
		# descendants.append(index)
		if len(self.children_list[index]) > 0:
			for children in self.children_list[index]:
				if children in descendants:
					continue
				# descendants.append(children)
				self.get_descendants(children, descendants=descendants)
			descendants += self.children_list[index]
		return descendants

	def calculate_spaciousness(self):
		self.spaciousness = np.zeros(shape=self.reduced_num_data)
		for i in range(self.reduced_num_data):
			self.spaciousness[i] = np.mean([self.reduced_data_distances[i][n] \
				for n in self.nearest_neighbours[i][1:self.spaciousness_neighbour_number+1]])
			# self.spaciousness[i] = np.mean([self.reduced_data_distances[i][n] \
			# 	for n in self.nearest_neighbours[i] if self.reduced_data_distances[i][n] <= self.rd_cutoff])
		if self.spaciousness.max() > 0:
			# self.spaciousness /= self.spaciousness.max()
			self.normalized_spaciousness = self.spaciousness / self.spaciousness.max()
		else:
			self.normalized_spaciousness = self.spaciousness

	def get_nearest_ground_truths(self):
		nearest_ground_truth_distances = np.ones(shape=self.reduced_num_data)
		nearest_ground_truth_index_map = np.zeros(shape=self.reduced_num_data, dtype=int)
		nearest_ground_truth_index_map[:] = -1
		
		known_keys = list(self.reduced_known_data_labels.keys())
		if len(known_keys) <= 0:
			return nearest_ground_truth_distances, nearest_ground_truth_index_map

		for i in range(self.reduced_num_data):
			nearest_index = known_keys[np.argmin(self.reduced_data_distances[i][known_keys])]
			nearest_ground_truth_index_map[i] = nearest_index
			nearest_distance = self.reduced_data_distances[i][nearest_index]
			nearest_ground_truth_distances[i] = nearest_distance

		max_distance = nearest_ground_truth_distances.max()
		if nearest_ground_truth_distances.max() > 0:
			nearest_ground_truth_distances /= nearest_ground_truth_distances.max()

		for index in self.max_density_indices:
			self.delta_index_map[index] = nearest_ground_truth_distances[index]
		return nearest_ground_truth_distances, nearest_ground_truth_index_map

	def calculate_uadp(self, labels, keys=None):
		if keys is None:
			keys = np.unique(labels).tolist()
		num_keys = len(keys)
		u_adp = np.zeros(shape=self.reduced_num_data)

		if num_keys < 1:
			return u_adp

		for index in range(self.reduced_num_data):
			count, count_labels = 0, np.zeros(shape=num_keys, dtype=int)
			count_rd_neighbour = 0
			for n in self.nearest_neighbours[index]:
				if self.reduced_data_distances[index][n] > self.rd_cutoff * self.marginalness_rd_factor:
					if self.marginalness_neighbourhood_size is None:
						break
					elif count_rd_neighbour <= 1 and count >= self.marginalness_neighbourhood_size:
						break
				else:
					count_rd_neighbour += 1
				count_labels[keys.index(labels[n])] += 1
				count += 1

			probabilities = count_labels / count
			u_adp[index] = -1 * sum([p * math.log2(p) for p in probabilities if p != 0])
		return u_adp

	def calculate_marginalness(self, labels=None):
		if len(self.reduced_known_data_labels) <= 0:
			return np.zeros(shape=self.reduced_num_data)
		else:
			return self.calculate_uadp(self.labels if labels is None else labels, keys=None)

	def calculate_confidence_levels(self, keys):
		self.confidence_level_dict = dict()

		for key in keys:
			self.confidence_level_dict[key] = np.zeros(shape=self.reduced_num_data)
		
		# u_adp = self.calculate_uadp(self.true_grade_labels, keys)
		# u_adp[list(self.reduced_known_data_labels.keys())] = 0
		# for k in range(len(keys)):
		# 	matrix = np.zeros(shape=self.reduced_num_data, dtype=int)
		# 	matrix[np.where(self.true_grade_labels == keys[k])] = 1
		# 	self.confidence_level_dict[keys[k]] = np.multiply(1 - u_adp, matrix)

		if self.voting_version == 'weighted_ig_average':
			information_gains, _ = self.calculate_information_gains(self.true_grade_labels)
			if self.confidence_version == 'ig_norden':
				confidence_levels = math.log(len(self.possible_grades), 2) - (information_gains * self.normalized_densities)
			else:
				confidence_levels = (math.log(len(self.possible_grades), 2) - information_gains)
		else:
			confidence_levels = self.calculate_uadp(self.true_grade_labels, keys)
			confidence_levels[list(self.reduced_known_data_labels.keys())] = 0
			confidence_levels = 1 - confidence_levels

		for k in range(len(keys)):
			matrix = np.zeros(shape=self.reduced_num_data, dtype=int)
			matrix[np.where(self.true_grade_labels == keys[k])] = 1
			self.confidence_level_dict[keys[k]] = np.multiply(confidence_levels, matrix)

	def calculate_entropy(self, probabilities):
		# class_count = len(self.possible_grades)
		# return -1 * sum([p * math.log(p, class_count) for p in probabilities if p > 0])
		return -1 * sum([p * math.log(p, 2) for p in probabilities if p > 0])

	def calculate_information_gains(self, labels):
		class_count = len(self.possible_grades)
		pre_grading_entropy = np.ones(shape=self.reduced_num_data)
		post_grading_entropy = np.zeros(shape=self.reduced_num_data)

		# if self.version == 6 and self.version_variation in ['0812C', '0812D', '0812E', '0825A', '0825B',
		# 	'0825B2', '0825C', '0825D'] or '0903' in self.version_variation:
		# if self.dynamic_grade_weights:
		# 	distributions = dict()
		# 	for k, v in self.reduced_known_data_labels.items():
		# 		if not v in distributions:
		# 			distributions[v] = 1
		# 		else:
		# 			distributions[v] += 1
		# 	for k, v in distributions.items():
		# 		self.possible_grades_weights[self.possible_grades.index(k)] = v / len(self.known_data_labels)
				
		if labels is not None:
			outlier_indices = np.where(self.labels == -1)[0]

			# calculate entropy for each data
			for i in range(self.reduced_num_data):
				if self.version == 6 and self.version_variant == '0825B' and i in outlier_indices:
					pre_grading_entropy[i] = 1
					continue

				# information gain should be 0 for answers already being graded
				if i in self.reduced_known_data_labels:
					post_grading_entropy[i] = 1
					continue
				
				true_grade_counts = np.zeros(shape=class_count)
				non_true_grade_counts = np.zeros(shape=class_count)
				neighbour_count = 0
				for n in self.nearest_neighbours[i]:
					if self.reduced_data_distances[i][n] > self.rd_cutoff:
						break
					label = self.true_grade_labels[n]
					# 'Unknown' considered as default grade
					if not label in self.possible_grades:
						label = self.default_grade
					position = self.possible_grades.index(label)
					dw = self.get_data_representation_number(n, 2)
					if n in self.reduced_known_data_labels:
						true_grade_counts[position] += dw
					else:
						non_true_grade_counts[position] += dw
					neighbour_count += 1

				counts = true_grade_counts + non_true_grade_counts
				total_neighbour_count = sum(counts)

				# information gain should be 0 for answers with no neighbour
				if neighbour_count <= 1:
					post_grading_entropy[i] = 1
					continue
				
				# calculate pre-grading entropy
				probabilities = counts / total_neighbour_count
				pre_grading_entropy[i] = self.calculate_entropy(probabilities)

				# first exclude current assigned label from non true grade counts 
				assigned_label = self.true_grade_labels[i] \
					if self.true_grade_labels[i] in self.possible_grades else self.default_grade
				post_grading_non_true_grade_counts = np.copy(non_true_grade_counts)
				post_grading_non_true_grade_counts[self.possible_grades.index(assigned_label)] \
					-= self.get_data_representation_number(i, 2)

				# calculate influence strengths
				influence_strengths = np.zeros(shape=class_count)
				for l in range(class_count):
					post_grading_true_grade_counts = np.copy(true_grade_counts)
					post_grading_true_grade_counts[l] += 1

					post_grading_probabilities = np.zeros(shape=class_count)
					# post_grading_true_grade_probabilities = post_grading_true_grade_counts \
					# 	/ sum(post_grading_true_grade_counts)
					post_grading_true_grade_count = sum(post_grading_true_grade_counts)
					post_grading_true_grade_probabilities = post_grading_true_grade_counts \
						/ (post_grading_true_grade_count if post_grading_true_grade_count > 0 else 1)
					# post_grading_non_true_grade_probabilities = post_grading_non_true_grade_counts \
					# 	/ sum(post_grading_non_true_grade_counts)
					post_grading_non_true_grade_count = sum(post_grading_non_true_grade_counts)
					post_grading_non_true_grade_probabilities = post_grading_non_true_grade_counts \
						/(post_grading_non_true_grade_count if post_grading_non_true_grade_count > 0 else 1)
					
					probability = sum([post_grading_non_true_grade_probabilities[p] \
						* post_grading_true_grade_probabilities[l] for p in range(class_count) if p != l])
					for p in range(class_count):
						post_grading_probabilities[p] = post_grading_non_true_grade_probabilities[p]
						if p == l:
							post_grading_probabilities[p] += probability
						else:
							post_grading_probabilities[p] -= probability
					influence_strengths[l] = self.calculate_entropy(post_grading_probabilities)
					post_grading_entropy[i] += influence_strengths[l] * self.possible_grades_weights[l]

		information_gains = pre_grading_entropy - post_grading_entropy

		if self.pgv_density_adjustment is not None:
			if self.pgv_density_adjustment == 'k':
				adjustments = self.normalized_descendant_counts
			elif 'k_den' in self.pgv_density_adjustment:
				adjustments = self.normalized_descendant_counts
				if self.pgv_density_adjustment == 'k_den_0':
					adjustments[self.normalized_densities != 0] = 0
				if self.pgv_use_normalized_density:
					adjustments = adjustments + self.normalized_densities
				else:
					adjustments = adjustments + self.densities
			return information_gains, information_gains * adjustments
		else:
			if self.pgv_use_normalized_density:
				return information_gains, information_gains * self.normalized_densities
			else:
				return information_gains, information_gains * self.densities

	def calculate_values(self, graded_answers, global_labels):
		nearest_ground_truth_distances, nearest_ground_truth_index_map = self.get_nearest_ground_truths()
		labels, outlier_indices, clustering_result = self.update_labels(graded_answers, global_labels,
			nearest_ground_truth_index_map, nearest_ground_truth_distances)

		if len(self.known_data_labels) > 0:
			_ = self.update_true_grade_labels(graded_answers, global_labels)
		labels = self.true_grade_labels

		if self.version == 6:
			information_gains , pgv_values = self.calculate_information_gains(labels)
			return labels, outlier_indices, nearest_ground_truth_distances, nearest_ground_truth_index_map, \
				information_gains, pgv_values
		else:
			return labels, outlier_indices, nearest_ground_truth_distances, nearest_ground_truth_index_map, \
				self.calculate_marginalness(labels=labels)
