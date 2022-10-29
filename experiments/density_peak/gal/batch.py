import os, sys
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..{}..{}..{}'.format(os.sep, os.sep, os.sep)))

import random

from basis import config
from experiments.density_peak.gal.initialization import *

if __name__ == '__main__':
	# name = 'USCIS'
	# question_ids = ['3'] + [str(i) for i in range(1, 9) if i != 3]
	# question_ids = [str(i) for i in range(1, 9) if i != 3]
	# question_ids = ['3']

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
	# question_ids = ['EV_12b', 'EV_22a', 'EV_25', 'EV_35a', 'HB_24b1', 'HB_35', 'WA_52b', 
	# 	'VOLTAGE_ELECTRICAL_STATE_DISCUSS_Q']
	question_ids = ['DAMAGED_BULB_SWITCH_Q', 'DESCRIBE_GAP_LOCATE_PROCEDURE_Q', 'EV_12b', 'EV_25',
		'HB_24b1', 'HB_35', 'HYBRID_BURNED_OUT_EXPLAIN_Q2', 'WA_52b']
	# N >= 100
	# question_ids =  ['DESCRIBE_GAP_LOCATE_PROCEDURE_Q', 'EV_12b', 'EV_25', 'HB_24b1', 'HB_35', 'WA_52b']

	mapping = None
	# subspace_param_list = [
	# 	{'encoder': 'google_universal_sentence_encoder'},
	# 	{'encoder': 'google_universal_sentence_encoder', 'compress_factor': 128},
	# 	{'encoder': 'google_universal_sentence_encoder', 'compress_factor': 32},
	# 	{'encoder': 'tfidf', 'model': '1_3_min_df_2', 'weight': 0},	# 77, 117 all zero
	# ]
	
	# subspace_param_list = [
	# 	{'encoder': 'google_universal_sentence_encoder', 'weight': 1}
	# ]
	# for i in range(8):
	# 	subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
	# 		'random_dimension': 128, 'random_reference_position': 0, 'id': i+1})

	subspace_param_list_list = []
	base_subspace1 = {'encoder': 'google_universal_sentence_encoder', 'weight': 1}
	base_subspace0 = {'encoder': 'google_universal_sentence_encoder', 'weight': 0}
	# subspace_param_list = [base_subspace1]
	# for i in range(8):
	# 	subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
	# 		'random_dimension': 128, 'random_reference_position': 0, 'id': i+1})
	# subspace_param_list_list.append(subspace_param_list)
	# subspace_param_list = [base_subspace1]
	# for i in range(4):
	# 	subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
	# 		'random_dimension': 256, 'random_reference_position': 0, 'id': i+1})
	# subspace_param_list_list.append(subspace_param_list)
	# subspace_param_list = [base_subspace1]
	# for i in range(8):
	# 	subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
	# 		'random_dimension': 256, 'random_reference_position': 0, 'id': i+1})
	# subspace_param_list_list.append(subspace_param_list)
	# subspace_param_list = [base_subspace1]
	# for i in range(32):
	# 	subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
	# 		'random_dimension': 32, 'random_reference_position': 0, 'id': i+1})
	# subspace_param_list_list.append(subspace_param_list)
	# subspace_param_list = [base_subspace1]
	# for i in range(48):
	# 	subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
	# 		'random_dimension': 32, 'random_reference_position': 0, 'id': i+1})
	# subspace_param_list_list.append(subspace_param_list)
	# subspace_param_list = [base_subspace0]
	# for i in range(32):
	# 	subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
	# 		'random_dimension': 32, 'random_reference_position': 0, 'id': i+1})
	# subspace_param_list_list.append(subspace_param_list)
	# subspace_param_list = [base_subspace0]
	# for i in range(48):
	# 	subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
	# 		'random_dimension': 32, 'random_reference_position': 0, 'id': i+1})
	# subspace_param_list_list.append(subspace_param_list)
		
	# Single Encoder
	encoder = 'glove'
	subspace_param_list = [{'encoder': encoder, 'weight': 0}]
	for i in range(16):
		subspace_param_list.append({'encoder': encoder, 'random_dimension': 64,
			'random_reference_position': 0, 'id': i+1})
	subspace_param_list_list.append(subspace_param_list)

	encoder = 'bert'
	subspace_param_list = [{'encoder': encoder, 'weight': 0}]
	for i in range(16):
		subspace_param_list.append({'encoder': encoder, 'random_dimension': 64,
			'random_reference_position': 0, 'id': i+1})
	subspace_param_list_list.append(subspace_param_list)

	# Mixed subspace of difference encoders, 64V x 16
	encoders = [('google_universal_sentence_encoder', 12), ('glove', 4)]
	subspace_param_list = [{'encoder': encoder, 'weight': 0} for encoder, _ in encoders]
	for position in range(len(encoders)):
		encoder,  count = encoders[position]
		for i in range(0, count):
			subspace_param_list.append({'encoder': encoder, 'random_dimension': 64,
				'random_reference_positions': [(position, 1)], 'random_selection_method': 'vertical',
				'id': len(subspace_param_list)-len(encoders)+1})
	subspace_param_list_list.append(subspace_param_list)

	encoders = [('google_universal_sentence_encoder', 12), ('bert', 4)]
	subspace_param_list = [{'encoder': encoder, 'weight': 0} for encoder, _ in encoders]
	for position in range(len(encoders)):
		encoder,  count = encoders[position]
		for i in range(0, count):
			subspace_param_list.append({'encoder': encoder, 'random_dimension': 64,
				'random_reference_positions': [(position, 1)], 'random_selection_method': 'vertical',
				'id': len(subspace_param_list)-len(encoders)+1})
	subspace_param_list_list.append(subspace_param_list)

	# Mixed embedding methods within subspaces
	encoders = ['google_universal_sentence_encoder', 'glove']
	subspace_param_list = [{'encoder': encoder, 'weight': 0} for encoder in encoders]
	for i in range(16):
		encoder = '_'.join(encoders)
		encoder = encoder.replace('google_universal_sentence_encoder', 'GUSE')
		subspace_param_list.append({'encoder': encoder, 'random_dimension': 128,
			'random_reference_positions': [(0, 0.5), (1, 0.5)], 'random_selection_method': 'vertical',
			'id': i+1})
	subspace_param_list_list.append(subspace_param_list)

	encoders = ['google_universal_sentence_encoder', 'bert']
	subspace_param_list = [{'encoder': encoder, 'weight': 0} for encoder in encoders]
	for i in range(16):
		encoder = '_'.join(encoders)
		encoder = encoder.replace('google_universal_sentence_encoder', 'GUSE')
		subspace_param_list.append({'encoder': encoder, 'random_dimension': 128,
			'random_reference_positions': [(0, 0.5), (1, 0.5)], 'random_selection_method': 'vertical',
			'id': i+1})
	subspace_param_list_list.append(subspace_param_list)

	iterations = None
	batch_size_one_length = 0
	batch_size = 5
	# possible_grades = ['Correct', 'Wrong']
	if name == 'SEB3' or mapping == '3way':
		possible_grades = [str(i) for i in range(2, -1, -1)]
		default_grade = '0'
	else:
		possible_grades = ['Correct', 'Wrong']
		default_grade = 'Wrong'
	
	grading_actions_list = list(range(20, 110, 10)) + [120, 150]
	
	# version 3
	# args = {'version': [3], 'distance_function': ['angular'],
	# 	'exclusion_rd_deduction_factor': [0.25], 'reset_exclusion_rd_factor': [True],
	# 	'grade_assignment_method': ['nearest_true_grade'], 'voting_version': ['weighted_average']}

	# version 4
	# args = {'version': [4], 'exclusion_rd_deduction_factor': [0.25], 'reset_exclusion_rd_factor': [True],
	# 	'grade_assignment_method': ['parent'], 'voting_version': ['weighted_average'],
	# 	'subspace_replacement_ratio': [0.5]}

	# version 5
	# args = {'version': [5], 'distance_function': ['angular'],
	# 	'grade_assignment_method': ['nearest_true_grade'], 'voting_version': ['weighted_average'],
	# 	'relevant_subspace_number': [5]}
	# args = {'version': [5], 'distance_function': ['angular'],
	# 	'grade_assignment_method': ['parent'], 'voting_version': ['weighted_average'],
	# 	'exclusion_rd_deduction_factor': [0.25], 'relevant_subspace_number': [5]}
	# args = {'version': [5], 'distance_function': ['angular'],
	# 	'grade_assignment_method': ['moc', 'oc'], 'voting_version': ['weighted_average'],
	# 	'exclusion_rd_deduction_factor': [0.25], 'relevant_subspace_number': [None]}
	# args = {'batch_size': [batch_size], 'version': [5], 'distance_function': ['angular'],
	# 	'relevant_subspace_number': [0], 'exclusion_rd_deduction_factor': [0.25],
	# 	'grade_assignment_method': ['parent_breaks', 'parent'], 'delta_link_threshold_factor': [5],
	# 	'voting_version': ['weighted_average']}
	
	# version 6
	# args = {'version': [6], 'distance_function': ['angular'], 'relevant_subspace_number': [0],
	# 	'rd_cutoff': [0.4], 'batch_size_one_length': [batch_size_one_length], 'batch_size': [batch_size],
	# 	'grade_assignment_method': ['parent_breaks'], 'delta_link_threshold_factor': [1.3],
	# 	'voting_version': ['weighted_average']}
	# args = {'version': [6], 'version_variant': [None],
	# 	'distance_function': ['angular'], 'relevant_subspace_number': [0], 'rd_cutoff': [0.4],
	# 	'batch_size_one_length': [batch_size_one_length], 'batch_size': [batch_size],
	# 	'grade_assignment_method': ['parent_breaks'], 'delta_link_threshold_factor': [1.3],
	# 	'voting_version': ['weighted_average']}
	# args = {'version': [6], 'version_variant': [None. '0812A', '0812B', '0812C', '0812D'],
	# 	'distance_function': ['angular'], 'relevant_subspace_number': [0], 'rd_cutoff': [0.4],
	# 	'batch_size_one_length': [batch_size_one_length], 'batch_size': [batch_size],
	# 	'grade_assignment_method': ['parent_breaks'], 'delta_link_threshold_factor': [1.3],
	# 	'voting_version': ['weighted_average']}
	# args = {'version': [6], 'version_variant': ['0825B'], 'distance_function': ['angular'],
	# 	'relevant_subspace_number': [0], 'rd_cutoff': [None],
	# 	'batch_size_one_length': [batch_size_one_length], 'batch_size': [batch_size],
	# 	'grade_assignment_method': ['parent_breaks'], 'delta_link_threshold_factor': [1],
	# 	'pgv_use_normalized_density': [True], 'normalize_distances': [True],
	# 	'voting_version': ['weighted_average']}
	# args = {'version': [6], 'version_variant': ['0825C'], 'distance_function': ['angular'],
	# 	'relevant_subspace_number': [0], 'rd_cutoff': [None],
	# 	'batch_size_one_length': [batch_size_one_length], 'batch_size': [batch_size],
	# 	'grade_assignment_method': ['parent_breaks'], 'delta_link_threshold_factor': [1.3],
	# 	'pgv_use_normalized_density': [True], 'normalize_distances': [True],
	# 	'voting_version': ['weighted_average']}
	args = {'version': [6], 'version_variant': ['0825D1_eqw'], 'distance_function': ['angular'],
		'relevant_subspace_number': [0], 'rd_cutoff': [None], 'rd_deriving_factor': [0.5],
		'batch_size_one_length': [batch_size_one_length], 'batch_size': [batch_size],
		'grade_assignment_method': ['parent_breaks'], 'delta_link_threshold_factor': [None],
		'delta_link_threshold_window_size_factor': [1 / 10],
		'pgv_use_normalized_density': [True], 'normalize_distances': [True],
		'voting_version': ['weighted_average']}
	
	explicit_param_list = generate_param_conbinations(args)
	# explicit_param_list = None
	
	# result_dict, labels, text_list, reduced_labels, reduced_text_list \
	# 	= evaluate_batch(name, question_id, mapping, subspace_param_list,
	# 		grading_actions_list=grading_actions_list, batch_size=batch_size, possible_grades=possible_grades,
	# 		explicit_param_list=explicit_param_list, random_seed=config.RANDOM_SEED)
	# _, _ = print_accuracy_batch(name, question_id, result_dict, labels, text_list, reduced_labels,
	# 	reduced_text_list, save_mode=1, folder_name=None)

	question_params = [(name, question_id) for question_id in question_ids]
	# question_params = [('USCIS', question_id) for question_id in ['3', '6', '8']]
	for name, question_id in question_params:
		folder_name = None
		for random_seed in [0, 49, 97, 53, 5, 33, 65, 62, 51, 100]:
			config.RANDOM_SEED = random_seed
			for s in range(len(subspace_param_list_list)):
				subspace_param_list = subspace_param_list_list[s]
				result_dict, labels, text_list, reduced_labels, reduced_text_list \
					= evaluate_batch(name, question_id, mapping, subspace_param_list,
						grading_actions_list=grading_actions_list, batch_size_one_length=batch_size_one_length,
						batch_size=batch_size, possible_grades=possible_grades,
						explicit_param_list=explicit_param_list, random_seed=config.RANDOM_SEED,
						load_subspaces=True)
				_, timestamp = print_accuracy_batch(name, question_id, result_dict, labels, text_list, reduced_labels,
					reduced_text_list, explicit_param_list=explicit_param_list, save_mode=1,
					folder_name=folder_name)
				if folder_name is None:
					folder_name = timestamp
