import os, sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..{}..{}..{}'.format(os.sep, os.sep, os.sep)))

import numpy as np
import matplotlib.pyplot as plt

from core import utils
from evaluation.results import load_encoded_dataset, load_dataset
from experiments.density_peak.gal.initialization import *

if __name__ == '__main__':
	name = 'SEB3'
	# question_ids = ['HB_43', 'EV_25', 'EV_12b', 'EV_35a', 'WA_51', 'PS_24a', 'HB_53a2',
	# 	'DAMAGED_BULB_SWITCH_Q']
	question_ids = ['DAMAGED_BULB_SWITCH_Q', 'DESCRIBE_GAP_LOCATE_PROCEDURE_Q', 'EV_12b', 'EV_25',
		'HB_24b1', 'HB_35', 'HYBRID_BURNED_OUT_EXPLAIN_Q2', 'WA_52b']

	encoder = 'google_universal_sentence_encoder'
	model = None
	mapping = None

	subspace_param_list = [
		{'encoder': 'google_universal_sentence_encoder', 'weight': 0}
	]
	for i in range(16):
		subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
			'random_dimension': 64, 'random_reference_position': 0, 'id': i+1})
	grading_actions = 30
	iterations = None
	batch_size_one_length = 0
	batch_size = 5
	possible_grades = ['Correct', 'Wrong']

	# version 6
	batch_sizes = calculate_batch_sizes(grading_actions, batch_size_one_length=batch_size_one_length,
		batch_size=batch_size)
	args = {'batch_sizes': batch_sizes, 'version': 6, 'version_variation': None,
		'distance_function': 'angular', 'relevant_subspace_number': 0, 'rd_cutoff': 0.4,
		'grade_assignment_method': 'parent_breaks', 'label_searching_factor': 1.3,
		'pgv_use_normalized_density': True, 'normalize_distances': True,
		'voting_version': 'weighted_average'}

	encoded_dataset, _ = load_encoded_dataset(name, encoder, model, mapping)
	answer_lengths = []
	for question in encoded_dataset.questions:
		length = []
		for answer in question.answers:
			if not answer.is_reference:
				continue
			length.append(len(answer.text))
		answer_lengths.append(sum(length) / len(length))
	sorted_questions = sorted(list(range(len(encoded_dataset.questions))), key=lambda i: answer_lengths[i],
		reverse=False)
	
	# question_ids = []
	# for q in sorted_questions:
	# 	question = encoded_dataset.questions[q]
	# 	question_id = question.id
	for question_id in question_ids:
		question = next((q for q in encoded_dataset.questions if q.id == question_id), None)
		algorithm = create_algorithm(name, question_id, subspace_param_list, grading_actions, iterations,
			possible_grades, random_seed=config.RANDOM_SEED, load_subspaces=True, **args)
		print('Question ID:', question_id)
		print('Data Number:', algorithm.num_data)
		# print('Encap Data Number', algorithm.reduced_num_data)
		# print('Densities, min: {}, max: {}'.format(algorithm.densities.min(), algorithm.densities.max()))

		# if algorithm.reduced_num_data < 50:
		# 	continue
		continue
		
		# print('{}, average answer length: {}'.format(question, answer_lengths[q]))
		dataset, labels, text_list, reference_answer_labels, reference_answer_text_list \
			= load_dataset(name, question_id, encoder, model, mapping)
		distributions = dict()
		for label in labels:
			distributions[label] = 1 if not label in distributions else distributions[label] + 1
		print('Question', question.text)
		print('Distributions: {}'.format(distributions))
		print('Reference Answers')
		for reference_answer in reference_answer_text_list:
			print(reference_answer)
		print('')

		reduced_labels = np.array(labels)[algorithm.reduced_data_indices]
		reduced_text_list = np.array(text_list)[algorithm.reduced_data_indices]
		for i in range(algorithm.num_data):
			reduced_index = algorithm.original_to_reduced_index_map[i]
			if labels[i] != reduced_labels[reduced_index]:
				print('i: {}, label: {}, text: {}'.format(i, labels[i], text_list[i]))
				print('reduced: {}, label: {}, text: {}'.format(reduced_index, reduced_labels[reduced_index],
					reduced_text_list[reduced_index]))

		# data = utils.get_tsne_data(dataset.data, mode='2d', perplexity=5.)
		# utils.plot(data, np.array(labels), 'DataSet {}'.format(question_id), core_data_indices=[])

		question_ids.append(question_id)
		if len(question_ids) > 10:
			break
	print('q_list', q_list)
	print('Question IDs', question_ids)
	print('Total Questions', len(sorted_questions))
	# plt.show()
	# sys.exit()
