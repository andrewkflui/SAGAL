import os, sys
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..{}..{}..{}'.format(os.sep, os.sep, os.sep)))

import random
import matplotlib.pyplot as plt

from basis import config
from core import utils
from experiments.density_peak.gal.initialization import *

def save_subspaces_batch(name, question_id, subspace_param_list, grading_actions, iterations, possible_grades,
	args, random_seeds):
	for random_seed in random_seeds:
		iterations = int(grading_actions / batch_size) if batch_size is not None else iterations
		algorithm = save_subspaces(name, question_id, subspace_param_list, grading_actions, iterations,
			possible_grades, random_seed, args)
		if algorithm is not None:
			print('saved, question_id: {}, version: {}, {}, reduced_num_data: {}, random_seed: {}'.format(
				question_id, algorithm.version, get_subspace_dimension_string(subspace_param_list),
				algorithm.reduced_num_data, random_seed))
		else:
			raise Exception('Subspaces Not Saved!', name, quesiton_id, subspace_param_list, random_seed, args)

def plot_inter_data_distance_histogram(algorithm, title='Inter-distances', data_indices=None, plot_every=True):
	subspace_dimension_string = get_subspace_dimension_string(algorithm.subspace_param_list)
	data_indices = data_indices or list(range(algorithm.reduced_num_data))
	x_labels, distance_list = [], []
	for subspace in algorithm.subspaces:
		subspace_name = subspace_dimension_string
		if subspace.id is not None:
			subspace_name += '({})'.format(subspace.id)
		distances = subspace.reduced_data_distances[data_indices]
		distances = distances[:, data_indices]
		distances = sorted(distances.flat)
		distance_list.append(distances)
		x_labels.append(subspace_name)
		if plot_every:
			utils.plot_histogram('{} {}'.format(subspace_name, title),
				[distances], bins=5, labels=['Count', subspace_name], log_scale=False,
				x_ticks=[i * 0.2 for i in range(6)], y_ticks=[i * 10000 for i in range(15)])
	utils.plot_histogram('Subspace {}'.format(title), distance_list, bins=5,
		labels=['Count'] + x_labels, log_scale=False)
	plt.show()

def plot_delta_link_distance_histogram(question_id, algorithm, title='Delta-link Distances',
	bins=[0.4 * i for i in range(4)], x_ticks=None, plot_every=False):
	subspace_dimension_string = get_subspace_dimension_string(algorithm.subspace_param_list)
	description = 'Original' if not algorithm.subspaces[0].normalize_distances else 'Normalized'
	x_labels, distance_list = [], []
	if x_ticks is None:
		x_ticks = [i * 0.2 for i in range(6)] if type(bins) != list else bins
	y_ticks = list(range(0, 700, 100))
	for subspace in algorithm.subspaces:
		subspace_name = subspace_dimension_string
		if subspace.id  is not None:
			subspace_name += '({})'.format(subspace.id)
		
		distances = []
		for i in range(subspace.reduced_num_data):
			parent = subspace.delta_index_map[i]
			if i == parent:
				continue
			distance = subspace.reduced_data_distances[i][parent]
			distances.append(distance)
		distance_list.append(distances)
		x_labels.append(subspace_name)
		if plot_every:
			utils.plot_histogram('{} {} {} ({})'.format(question_id, subspace_name, title, description),
				[distances], bins=bins, labels=['Count', subspace_name], log_scale=False, x_ticks=x_ticks)
	utils.plot_histogram('{} Subspace {} ({})'.format(question_id, title, description), distance_list, bins=bins,
		labels=['Count'] + x_labels, log_scale=False, x_ticks=x_ticks)
	plt.show()

def plot_delta_link_line_chart(question_id, algorithm, subspace, labels=None, title='Ranked Delta-link Distances'):
	subspace_dimension_string = get_subspace_dimension_string(algorithm.subspace_param_list)
	x_ticks = [i * 0.2 for i in range(6)]
	distances = []
	for i in range(subspace.reduced_num_data):
		parent = subspace.delta_index_map[i]
		distance = subspace.reduced_data_distances[i][parent]
		# distances.append(distance)
		distances += [distance] * subspace.get_data_representation_number(i, 2)
	distances = sorted(distances)
	ranks = [0]
	for d in range(1, len(distances)): 
		if distances[d] == distances[d-1]:
			ranks.append(ranks[d-1])
		else:
			ranks.append(ranks[d-1] + 1)
	
	data_list = []
	unique_labels = np.unique(np.array(labels))
	for unique_label in unique_labels:
		data_list.append([(ranks[i], distances[i]) for i in range(algorithm.num_data) \
			if labels[i] == unique_label])
	data_list = np.array(data_list)
	# utils.plot_multiple_lines('{} {} {} ({})'.format(question_id, title, subspace_dimension_string,
	# 	'Original' if not subspace.normalize_distances else 'Normalized'),
	# 	ranks, [distances], x_label='Rank', legend=['rank', 'delta-link distance'])
	utils.plot_scatter(data_list, x_label='Rank', y_label='Distance', legends=unique_labels,
		title='{} {} {} ({})'.format(question_id, title, subspace_dimension_string,
			'Original' if not subspace.normalize_distances else 'Normalized'), size=30)
	plt.show()

def plot_spaciousness_line_chart(question_id, algorithm, subspace, labels=None, title='Ranked Spaciousness'):
	subspace_dimension_string = get_subspace_dimension_string(algorithm.subspace_param_list)
	spaciousness_list = []
	for i in range(subspace.reduced_num_data):
		# spaciousness_list.append(subspace.spaciousness[i])
		spaciousness_list += [subspace.spaciousness[i]] * subspace.get_data_representation_number(i, 2)
	spaciousness_list = sorted(spaciousness_list)
	ranks = [0]
	for s in range(1, len(spaciousness_list)):
		if spaciousness_list[s] == spaciousness_list[s-1]:
			ranks.append(ranks[s-1])
		else:
			ranks.append(ranks[s-1] + 1)

	data_list = []
	unique_labels = np.unique(np.array(labels))
	for unique_label in unique_labels:
		data_list.append([(ranks[i], spaciousness_list[i]) for i in range(algorithm.num_data) \
			if labels[i] == unique_label])
	data_list = np.array(data_list)
	# utils.plot_multiple_lines('{} {} {} ({})'.format(question_id, title, subspace_dimension_string,
	# 	'Original Distance' if not subspace.normalize_distances else 'Normalized Distance'),
	# 	ranks, data_list, x_label='Rank',
	# 	legend=['rank', 'spaciousness'])
	utils.plot_scatter(data_list, x_label='Rank', y_label='Spaciousness', legends=unique_labels,
		title='{} {} {} ({})'.format(question_id, title, subspace_dimension_string,
		'Original Distance' if not subspace.normalize_distances else 'Normalized Distance'), size=30)

def find_common_questions_in_seb():
	from evaluation.results import load_encoded_dataset
	seb2, _ = load_encoded_dataset('SEB2', encoder, model, mapping)
	seb3, _ = load_encoded_dataset('SEB3', encoder, model, mapping)
	seb5, _ = load_encoded_dataset('SEB5', encoder, model, mapping)

	seb2_question_ids = set([q.id for q in seb2.questions])
	seb3_question_ids = set([q.id for q in seb3.questions])
	seb5_question_ids = set([q.id for q in seb5.questions])
	common_questions = seb2_question_ids & seb3_question_ids & seb5_question_ids
	print(common_questions)
	print(len(common_questions))

	lines = []
	# for question_id in question_ids:
	for question_id in common_questions:
		if question_id not in common_questions:
			print('{} NOT IN COMMON'.format(question_id))
		else:
			valid = True
			question_lines = []
			question_lines.append(question_id)
			for w in ['2', '3', '5']:
				dataset, labels, text_list, _, _ = load_dataset('SEB{}'.format(w), question_id, encoder, model, mapping=mapping)
				# if len(labels) < 50 or len(np.unique(labels)) < int(w):
				if len(labels) < 50 or (int(w) < 5 and len(np.unique(labels)) < int(w)) \
					or (int(w) == 5 and len(np.unique(labels)) < 4):
					valid = False
					break
				distributions = dict()
				for label in labels:
					if not label in distributions:
						distributions[label] = 1
					else:
						distributions[label] += 1
				question_lines.append('SEB{}, Data Number: {}, Distributions: {}'.format(w, len(labels), distributions))

			if valid:
				lines += question_lines
				lines.append('Question: {}'.format(dataset.question.text))
				lines.append('Reference Answers')
				for reference_answer in dataset.reference_answers:
					lines.append(reference_answer.text)
				lines.append('')
	utils.write('d:\\data\\Desktop', 'SEB.txt', lines)
	map(print, lines)

if __name__ == '__main__':
	name = 'USCIS'
	question_id = '3'
	
	name = 'SEB2'
	# SEB2 long answers
	# question_ids = ['HB_43', 'EV_25', 'EV_12b', 'EV_35a', 'HB_35', 'EV_22a', 'WA_29', 'WA_31', 'WA_51', 'PS_24a', 'WA_32b']
	# selected
	# question_ids = ['HB_43', 'EV_25', 'EV_12b', 'EV_35a', 'WA_51', 'PS_24a']
	# SEB2 short answers
	# question_ids = ['HB_59a', 'WA_12c', 'HB_53a2', 'HB_24b1', 'BURNED_BULB_SERIES_Q2', 'DAMAGED_BULB_SWITCH_Q']
	# selected
	# question_ids = ['HB_53a2', 'DAMAGED_BULB_SWITCH_Q']
	# question_ids = ['HB_43', 'EV_25', 'EV_12b', 'EV_35a', 'WA_51', 'PS_24a', 'HB_53a2', 'DAMAGED_BULB_SWITCH_Q']

	# name = 'SEB3'
	# SEB3 long questions
	# question_ids = ['DESCRIBE_GAP_LOCATE_PROCEDURE_Q', 'EV_22a', 'EV_25', 'EV_35a',
	# 	'HYBRID_BURNED_OUT_EXPLAIN_Q2']
	# SEB5 long questions
	# question_ids = ['EV_12b', 'HB_35', 'WA_52b']
	# SEB3 short questions
	# question_ids = ['DAMAGED_BULB_SWITCH_Q']
	# SEB5 short questions
	# question_ids = ['CLOSED_PATH_EXPLAIN', 'HB_24b1', 'HYBRID_BURNED_OUT_EXPLAIN_Q2',
	# 	'VOLTAGE_ELECTRICAL_STATE_DISCUSS_Q']
	# selected
	# question_ids = ['EV_12b', 'EV_22a', 'EV_25', 'EV_35a', 'HB_35', 'WA_52b', 'HB_24b1',
	# 	'VOLTAGE_ELECTRICAL_STATE_DISCUSS_Q']
	# question_ids = ['DAMAGED_BULB_SWITCH_Q', 'DESCRIBE_GAP_LOCATE_PROCEDURE_Q', 'EV_12b', 'EV_25',
	# 	'HB_24b1', 'HB_35', 'HYBRID_BURNED_OUT_EXPLAIN_Q2', 'WA_52b']
	question_id = 'DAMAGED_BULB_SWITCH_Q'

	encoder = 'google_universal_sentence_encoder'
	model = None
	mapping = None
	# subspace_param_list = [
	# 	{'encoder': 'google_universal_sentence_encoder'},
	# 	{'encoder': 'google_universal_sentence_encoder', 'compress_factor': 128},
	# 	{'encoder': 'google_universal_sentence_encoder', 'compress_factor': 32},
	# 	{'encoder': 'tfidf', 'model': '1_3_min_df_2', 'weight': 0}
	# ]
	
	# Single Encoder
	encoder = 'glove'
	subspace_param_list = [{'encoder': encoder, 'weight': 0}]
	for i in range(16):
		subspace_param_list.append({'encoder': encoder, 'random_dimension': 64,
			'random_reference_position': 0, 'id': i+1})

	# Mixed subspace of difference encoders, 64V x 16
	# encoders = [('google_universal_sentence_encoder', 12), ('glove', 4)]
	# subspace_param_list = [{'encoder': encoder, 'weight': 0} for encoder in encoders]
	# for position in range(len(encoders)):
	# 	encoder,  count = encoders[position]
	# 	for i in range(0, count):
	# 		subspace_param_list.append({'encoder': encoder, 'random_dimension': 64,
	# 			'random_reference_positions': [(position, 1)], 'random_selection_method': 'vertical',
	# 			'id': len(subspace_param_list)-len(encoders)+1})

	# Mixed embedding methods within subspaces
	# encoders = ['google_universal_sentence_encoder', 'glove']
	# subspace_param_list = [{'encoder': encoder, 'weight': 0} for encoder in encoders]
	# for i in range(16):
	# 	encoder = '_'.join(encoders)
	# 	encoder = encoder.replace('google_universal_sentence_encoder', 'GUSE')
	# 	subspace_param_list.append({'encoder': encoder, 'random_dimension': 128,
	# 		'random_reference_positions': [(0, 0.5), (1, 0.5)], 'random_selection_method': 'vertical',
	# 		'id': i+1})
	
	grading_actions = 30
	iterations = None
	batch_size_one_length = 0
	batch_size = 5
	if name == 'SEB3' or mapping == '3way':
		possible_grades = [str(i) for i in range(2, -1, -1)]
		default_grade = '0'
	else:
		possible_grades = ['Correct', 'Wrong']
		default_grade = 'Wrong'

	random_seeds = [0, 49, 97, 53, 5, 33, 65, 62, 51, 100]
	# version 3
	# args = {'batch_size': batch_size, 'version': 3, 'distance_function': 'angular',
	# 	'exclusion_rd_deduction_factor': 0.25, 'grade_assignment_method': 'nearest_true_grade',
	# 	'voting_version': 'weighted_average'}

	# version 4
	# args = {'batch_size': batch_size, 'version': 4, 'distance_function': 'manhattan',
	# 	'exclusion_rd_deduction_factor': 0.25,
	# 	'grade_assignment_method': 'parent', 'voting_version': 'weighted_average',
	# 	'subspace_replacement_ratio': 0.1}

	# version 5
	# args = {'batch_size': batch_size, 'version': 5, 'distance_function': 'angular',
	# 	'relevant_subspace_number': 5, 'exclusion_rd_deduction_factor': 0.25,
	# 	'grade_assignment_method': 'nearest_true_grade', 'voting_version': 'weighted_average'}
	# args = {'batch_size': batch_size, 'version': 5, 'distance_function': 'angular',
	# 	'relevant_subspace_number': 0, 'exclusion_rd_deduction_factor': 0.25,
	# 	'grade_assignment_method': 'soc', 'label_searching_boundary_factor': 2,
	# 	'label_knn': 5, 'voting_version': 'weighted_average'}
	# args = {'batch_size': batch_size, 'version': 5, 'distance_function': 'angular',
	# 	'relevant_subspace_number': 0, 'rd_cutoff': 0.4, 'exclusion_rd_factor': 2.,
	# 	'exclusion_rd_deduction_factor': 0.2, 'marginalness_rd_factor': 1.,
	# 	'grade_assignment_method': 'parent_breaks', 'delta_link_threshold_factor': 1.5,
	# 	'voting_version': 'weighted_average'}

	# version 6
	batch_sizes = calculate_batch_sizes(grading_actions, batch_size_one_length=batch_size_one_length,
		batch_size=batch_size)
	# args = {'batch_sizes': batch_sizes, 'version': 6, 'version_variant': '0812E',
	# 	'distance_function': 'angular', 'relevant_subspace_number': 0, 'rd_cutoff': 0.4,
	# 	'grade_assignment_method': 'parent_breaks', 'delta_link_threshold_factor': 1.3,
	# 	'pgv_use_normalized_density': True, 'normalize_distances': True,
	# 	'voting_version': 'weighted_average'}
	# args = {'batch_sizes': batch_sizes, 'version': 6, 'version_variant': '0825D',
	# 	'distance_function': 'angular', 'relevant_subspace_number': 0, 'rd_cutoff': None,
	# 	'rd_deriving_factor': 0.5, 'grade_assignment_method': 'parent_breaks',
	# 	'delta_link_threshold_factor': None, 'delta_link_threshold_window_size_factor': 1 / 10,
	# 	'pgv_use_normalized_density': True, 'normalize_distances': True,
	# 	'voting_version': 'weighted_average'}
	args = {'batch_sizes': batch_sizes, 'version': 6, 'version_variant': '0825D1_eqw',
		'distance_function': 'angular', 'relevant_subspace_number': 0, 'rd_cutoff': None,
		'rd_deriving_factor': 0.5, 'grade_assignment_method': 'parent_breaks',
		'delta_link_threshold_factor': None, 'delta_link_threshold_window_size_factor': 1 / 10,
		'pgv_use_normalized_density': True, 'normalize_distances': True,
		'voting_version': 'weighted_average'}

	# save subspaces  for speeding up
	# save_subspaces_batch(name, question_id, subspace_param_list, grading_actions, iterations,
	# 	possible_grades, args, random_seeds)
	# sys.exit()

	algorithm, labels, text_list, reduced_labels, reduced_text_list, result, gp_dict, time_used \
		= evaluate(name, question_id, mapping, subspace_param_list, grading_actions, iterations,
			possible_grades, random_seed=config.RANDOM_SEED, load_subspaces=True, **args)
	# folder_name, _, _ = print_selection_values(name, question_id, algorithm, gp_dict, labels, text_list,
	# 	explicit_list=None, save_mode=2, folder_name=None)
	# _, _ = print_wrongly_graded_answers(name, question_id, [algorithm], labels, text_list, [gp_dict],
	# 	save_mode=1, folder_name=folder_name)
	sys.exit()

	# print delta link threshold
	# question_params = [('USCIS', str(i)) for i in range(1, 9)]
	# question_params += [('SEB2', question_id) for question_id in question_ids]
	# question_params = [('SEB3', question_id) for question_id in question_ids]
	# question_params = [('USCIS', '3')]
	# print_delta_link_thresholds(question_params, subspace_param_list, random_seeds=random_seeds,
	# 	load_subspaces=True, **args)
	# sys.exit()

	# plot histograms
	# for name, question_id in question_params:
	# 	algorithm = create_algorithm(name, question_id, subspace_param_list, grading_actions, iterations,
	# 		possible_grades, random_seed=config.RANDOM_SEED, load_subspaces=True, **args)
	# 	print('Dataset: {}, Question: {}'.format(name, question_id))
	# 	print('RD:', algorithm.rd_cutoff)
	# 	print('Delta Link Threshold:', algorithm.delta_link_threshold)
	# 	dataset, labels, text_list, _, _ = load_dataset(name, question_id, encoder, model, mapping=mapping)
	# 	reduced_labels = np.array(labels)[algorithm.reduced_data_indices]
	# 	# data_indices = [i for i in range(algorithm.reduced_num_data) if reduced_labels[i] == 'Correct']
	# 	# plot_inter_data_distance_histogram(algorithm, 
	# 	# 	title='Q{} Correct Answer Inter-distances'.format(question_id), data_indices=data_indices,
	# 	# 	plot_every=False)
	# 	# plot_delta_link_distance_histogram(question_id, algorithm, title='Delta-link Distances',
	# 	# 	bins=[0.05 * i for i in range(20)], x_ticks=[i * 0.1 for i in range(13)], plot_every=False)
	# 	# plot_delta_link_line_chart(question_id, algorithm, algorithm.subspaces[1], labels=labels)
	# 	plot_spaciousness_line_chart(question_id, algorithm, algorithm.subspaces[1], labels=labels)
	# sys.exit()

	# dataset, labels, text_list, _, _ = load_dataset(name, question_id, encoder, None, mapping=mapping)
	# algorithms = []
	# for random_seed in random_seeds:
	# 	iterations = int(grading_actions / batch_size) if batch_size is not None else iterations
	# 	algorithm = create_algorithm(name, question_id, subspace_param_list, grading_actions, iterations,
	# 		possible_grades, random_seed=random_seed, load_subspaces=True, **args)
	# 	algorithms.append(algorithm)
	# print_relevant_subspace_values(name, question_id, algorithms, labels, text_list, save_mode=1,
	# 	folder_name=None)

	# USCIS Q3 answers of high density (>= 0.9)
	explicit_list = [
		'declared the united states (or rather, the body of the u.s. at the time) independent from great britain',
		'freed america from the british',
		'allow the united states to become a country',
		'declare and liberate to form the u.s.a. from great britain',
		'declare the us\' independence from the foreign oppression of great britain.',
		'declared our independence from the english crown, asserted our sovereighnty',
		'shows that america is independent from britain',
		'declared the (previously) 13 colonies of the (new) united states of america to be a separate entity from great britain, which had owned the colonies up until that point.',
		'gave everyone rights.',
		'separation of us from gb',
		'declared indpendence from england',
		'sort of nothing, but it announced revolt from the british crown by the american colonies.',
		'very little',
		'don\'t know.',
	]
	# USCIS Q3 answers of low density (<=0.3)
	explicit_list += [
		'declared independence from england',
		'declared the colony\'s independence from britain.',
		'declared us independence from great britain',
		'declared independence from the british',
		'declare independence from great britain',
		'informed the british government that its territories in north america wanted to be independent from their rule.',
		'secession from britain',
		'united the 13 states against tyranny',
		'setup guidelines for a new country independent of british rule',
		'started the revolutionary war'
	]
	explicit_list = None
	
	# count = 0
	# sorted_density_indices = sorted(list(range(algorithm.reduced_num_data)),
	# 	key=lambda i: algorithm.densities[i], reverse=True)

	# pool1, pool2 = [], []
	# for i in range(algorithm.reduced_num_data):
	# 	if algorithm.densities[i] >= 0.9:
	# 		pool1.append(i)
	# 	elif algorithm.densities[i] <= 0.3:
	# 		pool2.append(i)

	# while count < 10:
	# 	pool = pool1 if count < 5 else pool2
	# 	index = random.choices(population=pool)[0]
	# 	if not reduced_text_list[index] in explicit_list:
	# 		explicit_list.append(reduced_text_list[index])
	# 		count += 1	

	# print_selection_values(name, question_id, algorithm, gp_dict, labels, text_list,
	# 	explicit_list=explicit_list, save_mode=2)
