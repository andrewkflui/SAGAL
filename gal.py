import os, sys
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..{}..{}..{}'.format(os.sep, os.sep, os.sep)))

import random

from basis import config
from core import utils
from experiments.density_peak.gal.initialization import *

def save_subspaces_batch(name, question_id, subspace_param_list, grading_actions, iterations, class_count,
	args, random_seeds):
	for random_seed in random_seeds:
		iterations = int(grading_actions / batch_size) if batch_size is not None else iterations
		algorithm = save_subspaces(name, question_id, subspace_param_list, grading_actions, iterations,
			class_count, random_seed, args)
		if algorithm is not None:
			print('saved, question_id: {}, version: {}, {}, reduced_num_data: {}, random_seed: {}'.format(
				question_id, algorithm.version, get_subspace_dimension_string(subspace_param_list),
				algorithm.reduced_num_data, random_seed))
		else:
			raise Exception('Subspaces Not Saved!', name, quesiton_id, subspace_param_list, random_seed, args)

def create_subspace_param_list(encoder, original_subspace_weight, num_subspace, random_dimension):
	subspace_param_list = [{'encoder': encoder, 'weight': original_subspace_weight}]
	for i in range(num_subspace):
		subspace_param_list += [{'encoder': encoder, 'random_dimension': random_dimension,
			'random_reference_position': 0, 'id': i+1}]
	dimension_string, subspaceencoder = get_subspace_dimension_string(subspace_param_list)
	return {'{} {}'.format(subspaceencoder.upper(), dimension_string): subspace_param_list}

def create_mixed_subspace_param_list(encoders, random_dimension):
	subspace_param_list = [{'encoder': encoder[0], 'weight': 0} for encoder in encoders]
	for position in range(len(encoders)):
		encoder,  count = encoders[position]
		for i in range(0, count):
			subspace_param_list.append({'encoder': encoder, 'random_dimension': random_dimension,
				'random_reference_positions': [(position, 1)], 'random_selection_method': 'vertical',
				'id': len(subspace_param_list)-len(encoders)+1})
	dimension_string, subspaceencoder = get_subspace_dimension_string(subspace_param_list)
	return {'{} {}'.format(subspaceencoder.upper(), dimension_string): subspace_param_list}

def create_mixed_inside_subspace_param_list(encoders, random_dimension):
	subspace_param_list = [{'encoder': encoder, 'weight': 0} for encoder in encoders]
	for i in range(16):
		encoder = '_'.join(encoders)
		encoder = encoder.replace('google_universal_sentence_encoder', 'GUSE')
		subspace_param_list.append({'encoder': encoder, 'random_dimension': random_dimension,
			'random_reference_positions': [(0, 0.5), (1, 0.5)], 'random_selection_method': 'vertical',
			'id': i+1})
	dimension_string, subspaceencoder = get_subspace_dimension_string(subspace_param_list)
	return {'{} {}'.format(subspaceencoder.upper(), dimension_string): subspace_param_list}

POSSIBLE_GRADES_OPTIONS = {1: ['Correct', 'Wrong'], 2: ['2', '1', '0']}
ENCODERS = {'GUSE': 'google_universal_sentence_encoder', 'Skip Thoughts': 'skip_thoughts',
	'Bert': 'bert', 'GloVe': 'glove', 'TFIDF': 'tfifd'}
SUBSPACE_OPTION_LIST = {}
SUBSPACE_OPTION_LIST.update(create_subspace_param_list('google_universal_sentence_encoder', 0, 16, 64))
SUBSPACE_OPTION_LIST.update(create_subspace_param_list('glove', 0, 16, 64))
SUBSPACE_OPTION_LIST.update(create_mixed_subspace_param_list(
	[('google_universal_sentence_encoder', 12), ('glove', 4)], 64))
SUBSPACE_OPTION_LIST.update(create_mixed_inside_subspace_param_list(
	['google_universal_sentence_encoder', 'glove'], 128))

def gal(name, question_id, mapping, subspace_param_list_key, grading_actions, load_subspaces=False,
	save_mode=0, **args):
	try:
		subspace_param_list = SUBSPACE_OPTION_LIST[subspace_param_list_key]
	except Exception as e:
		raise utils.InvalidNameError('subspace param list key', subspace_param_list_key)
	
	if name == 'SEB3' or mapping == '3way':
		possible_grades = [str(i) for i in range(2, -1, -1)]
		# default_grade = '0'
	else:
		possible_grades = ['Correct', 'Wrong']
		# default_grade = 'Wrong'
	
	iterations = args.pop('iterations') if 'iterations' in args else None
	algorithm, labels, text_list, reduced_labels, reduced_text_list, result, gp_dict, time_used \
		= evaluate(name, question_id, mapping, subspace_param_list, grading_actions, iterations, possible_grades,
		random_seed=config.RANDOM_SEED, load_subspaces=load_subspaces, **args)
	folder_name, _, _ = print_selection_values(name, question_id, algorithm, gp_dict, labels, text_list,
		explicit_list=None, save_mode=2 if save_mode > 0 else 0, folder_name=None)
	_, _ = print_wrongly_graded_answers(name, question_id, [algorithm], labels, text_list, [gp_dict], save_mode=1 if save_mode > 0 else 0,
		folder_name=folder_name)
	return algorithm, result, folder_name

if __name__ == '__main__':
	name = 'USCIS'
	question_id = '8'
	encoder = 'google_universal_sentence_encoder'
	mapping = None
	subspace_param_list = [
		{'encoder': 'google_universal_sentence_encoder', 'weight': 0}
	]
	for i in range(16):
		subspace_param_list.append({'encoder': 'google_universal_sentence_encoder',
			'random_dimension': 64, 'random_reference_position': 0, 'id': i+1})
	
	grading_actions = 150
	iterations = None
	batch_size = 10
	# class_count = 2
	if name == 'SEB3' or mapping == '3way':
		possible_grades = [str(i) for i in range(2, -1, -1)]
		default_grade = '0'
	else:
		possible_grades = ['Correct', 'Wrong']
		default_grade = 'Wrong'

	random_seeds = [0, 49, 97, 53, 5, 33, 65, 62, 51, 100]
	args = {'batch_size': batch_size, 'version': 5, 'distance_function': 'angular',
		'relevant_subspace_number': 0, 'exclusion_rd_deduction_factor': 0.25,
		'grade_assignment_method': 'moc', 'label_search_boundary_factor': 2,
		'voting_version': 'weighted_average'}
	
	# run and evaluate
	algorithm, labels, text_list, reduced_labels, reduced_text_list, result, gp_dict, time_used \
		= evaluate(name, question_id, mapping, subspace_param_list, grading_actions, iterations, possible_grades,
		random_seed=config.RANDOM_SEED, load_subspaces=True, **args)
	folder_name, _, _ = print_selection_values(name, question_id, algorithm, gp_dict, labels, text_list,
		explicit_list=None, save_mode=2, folder_name=None)
	_, _ = print_wrongly_graded_answers(name, question_id, [algorithm], labels, text_list, [gp_dict], save_mode=1,
		folder_name=folder_name)
	
	# save subspaces  for speeding up
	# save_subspaces_batch(name, question_id, subspace_param_list, grading_actions, iterations, class_count,
	# 	args, random_seeds)
