import os, sys, math, pickle, argparse, statistics, copy, datetime
from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt

from core import utils
from basis import constants, config
from evaluation.algorithms import setup
from evaluation.results import load_dataset, handle_result, plot_graphs

def load_data(name, question_id, encoder, model=None, compress_factor=None, compress_method='pca'):
	source_path = encoder
	model = model if model is not None else constants.ENCODERS[encoder]['default_model']
	source_path = Path('data/datasets/encoded/{}/{}/{}/{}'.format(encoder, model, constants.DATASETS[name]['processed_folder'], question_id))
	source_path = Path(config.ROOT_PATH) / source_path
	file_name = 'data.npy'
	reference_answer_file_name = 'reference_answer_data.npy'

	if not os.path.exists(source_path / file_name):
		raise Exception('No data found for dataset {}, question {}, encoder: {}, model: {}'.format(name, question_id, encoder, model))

	data = np.load(source_path / file_name, allow_pickle=True)
	reference_data = np.load(source_path / reference_answer_file_name, allow_pickle=True)

	compressed_data = None
	compressed_reference_data = None

	if compress_factor is not None:
		factor = int(len(data[0]) * compress_factor) if compress_factor < 1. else int(compress_factor)
		compress_path = source_path / 'compressed/{}_{}'.format(compress_method, factor) / file_name
		compress_reference_answer_path = source_path / 'compressed/{}_{}'.format(compress_method, factor) / reference_answer_file_name

		if config.RANDOM_SEED is not None and os.path.exists(compress_path):
			# data = utils.load(pca_path)
			compressed_data = np.load(compress_path)
			if os.path.exists(compress_reference_answer_path):
				# reference_data = utils.load(pca_reference_answer_path)
				compressed_reference_data = np.load(compress_reference_answer_path, allow_pickle=True)
			else:
				# reference_data = utils.load(source_path / reference_answer_file_name)
				compressed_reference_data = np.load(source_path / reference_answer_file_name,
					allow_pickle=True)
				# compressed_reference_data = utils.compress_data(reference_data, factor, source_path,
				# 	is_reference_answers=True, save_to_file=config.RANDOM_SEED is not None,
				# 	method=compress_method)
				compressed_data, compressed_reference_data = utils.compress_data(
					data, reference_data, factor, source_path,
					save_to_file=config.RANDOM_SEED is not None, method=compress_method)
			# return data, reference_data
		else:
			# data = utils.load(source_path / file_name)
			# compressed_data = np.load(source_path / file_name)
			compressed_data = np.copy(data)
			# reference_data = utils.load(source_path, reference_answer_file_name)
			# compressed_reference_data = np.load(source_path / reference_answer_file_name)
			compressed_reference_data = np.copy(reference_data)
			# compressed_data = utils.compress_data(compressed_data, factor, source_path,
			# 	is_reference_answers=False, save_to_file=config.RANDOM_SEED is not None,
			# 	method=compress_method)
			# compressed_reference_data = utils.compress_data(compressed_reference_data, factor,
			# 	source_path, is_reference_answers=True, save_to_file=config.RANDOM_SEED is not None,
			# 	method=compress_method)
			compressed_data, compressed_reference_data = utils.compress_data(
				data, reference_data, factor, source_path, save_to_file=config.RANDOM_SEED is not None,
				method=compress_method)
	
	# return np.load(source_path / file_name, allow_pickle=True), \
	# 	np.load(source_path / reference_answer_file_name, allow_pickle=True), \
	# 	compressed_data, compressed_reference_data
	return data, reference_data, compressed_data, compressed_reference_data

def run(name, question_id, encoder, model, args):
	save_mode = args.get('save_mode', 0)
	result_set = None

	if args.get('algorithm') == 'gal':
		print('Clustering Start')
		start_time = datetime.datetime.now()

		# from evaluation.algorithms import GALSolution
		# from experiments.density_peak.gal.initialization import evaluate
		# algorithm, labels, text_list, reduced_labels, reduced_text_list, result, gp_dict \
		# 	= evaluate(**args)
		# algorithm.data = algorithm.subspaces[0].data

		# end_time = datetime.datetime.now()
		# time_used = end_time - start_time

		# result = [GALSolution(algorithm.get_result())]

		from gal import gal
		algorithm, result, folder_name = gal(**args)
		if algorithm.subspaces[0].data is None:
			algorithm.data, _, _, _ = load_data(name, question_id, encoder, model,
				args.get('compress_factor', None), args.get('compress_method', 'pca'))
		else:
			algorithm.data = algorithm.subspaces[0].data

		end_time = datetime.datetime.now()
		time_used = end_time - start_time

		from evaluation.algorithms import GALSolution
		result = [GALSolution(algorithm.get_result(), algorithm.get_printable_params())]
	else:
		data, reference_answer_data, compressed_data, compressed_reference_answer_data \
			= load_data(name, question_id, encoder, model, args.get('compress_factor', None),
				args.get('compress_method', 'pca'))
		_, labels, text_list, reference_answer_labels, reference_answer_text_list \
			= load_dataset(name, question_id, encoder, model, mapping=args.get('mapping', None))

		data = compressed_data if compressed_data is not None else data

		# handle case that number of data < specified cluster number
		if 'cluster_num' in args:
			args['cluster_num'] = min(len(data), args['cluster_num'])

		algorithm = setup(data, args=args)
		# print(algorithm)
		
		print('Clustering Start')
		start_time = datetime.datetime.now()
		print('algorithm.name: {}'.format(algorithm.name))
		algorithm.run()
		end_time = datetime.datetime.now()
		time_used = end_time - start_time
		print('Clustering Finished, {}\n'.format(time_used))

		result = algorithm.get_result()
		folder_name = None
	
	result_set, _ = handle_result(name, question_id, encoder, model, args.get('compress_factor', None),
		compress_method=args.get('compress_method', None), mapping=args.get('mapping', None),
		algorithm=algorithm, result=result, time_used=time_used, save_mode=save_mode,
		plot=args.get('plot', None), folder_name=folder_name)

	return result_set

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Required
	parser.add_argument('--name', help='dataset name, options={}'.format([key for key in constants.DATASETS.keys()] + ['dummy']), required=True)
	parser.add_argument('--question_id', help='ID of question for clustering', type=str, default=None, required=False)
	parser.add_argument('--encoder', help='encoder name, options={}'.format([name for name in constants.ENCODERS]), required=False)
	parser.add_argument('--without_stop_words', help='stop words removed (for word embeddings only)', action='store_true', required=False)
	parser.add_argument('--model', help='model name (for some encoders only)', required=False)
	parser.add_argument('--mapping', help='mapping applied to original labels', required=False)
	parser.add_argument('--algorithm', help='algorithm, options={}'.format([x for x in constants.ALGORITHMS]), default=constants.ALGORITHMS[0])
	parser.add_argument('--compress_factor', help='compress factor', type=float, required=False)
	parser.add_argument('--compress_method', help='compress method', type=str, default='pca')
	parser.add_argument('--cluster_num', help='number of cluster', type=int, required=False)

	# GAL arguments
	parser.add_argument('--version', help='gal version, default is 5 (latest)', type=int, default=5, required=False)
	parser.add_argument('--subspace_param_list_key', help='subspace param list key, default is \'GUSE 64Vx16\'', default='GUSE 64Vx16', required=False)
	parser.add_argument('--grading_actions', help='grading actions', type=int, required=False)
	parser.add_argument('--iterations', help='number of iterations, not used if batch_size is given', type=int, required=False)
	parser.add_argument('--batch_size', help='batch size for each iteration (default 5)', type=int, default=5, required=False)
	# parser.add_argument('--class_count', help='number of classes (default is 2)', default=2, required=False)
	parser.add_argument('--exclusion_rd_deduction_factor', help='value to be used in each exclusion RD deduction', default=config.EXCLUSION_RD_DEDUCTION_FACTOR, required=False)
	parser.add_argument('--grade_assignment_method', help='parent / nearest_true_grade / oc / moc', default=config.GRADE_ASSIGNMENT_METHOD, required=False)
	parser.add_argument('--label_search_boundary_factor', help='delta-link break for oc / moc', default=config.LABEL_SEARCHING_BOUNDARY_FACTOR, required=False)
	parser.add_argument('--relevant_subspace_number', help='affects the number of relevant subspaces, no effect if 0 or None', default=config.RELEVANT_SUBSPACE_NUMBER, required=False)

	# DBSCAN arguments
	parser.add_argument('--eps', help='eps (for DBSCAN)', type=float, default=0.1, required=False)
	parser.add_argument('--min_samples', help='min samples (for DBSCAN / HDBSCAN / OPTICS)', type=float, default=10, required=False)

	# Birch
	parser.add_argument('--threshold', help='threshold', type=float, default=0.5, required=False)

	# For algorithms that ignore noises
	parser.add_argument('--count_unclassified', help='include unclassified cluster', action='store_true', required=False)

	parser.add_argument('--save_mode', help='save mode, 0: print only, 1: write text file, 2: write text file and save solution set', type=int, default=0)
	parser.add_argument('--plot', help='plot data with mode 2d or 3d', type=str, default=None, required=False)
	# parser.add_argument('--show_stats', help='show information of encoded data', action='store_true', required=False)

	args = parser.parse_args()
	name = args.name
	question_id = args.question_id
	encoder = args.encoder if args.algorithm != constants.ALGORITHMS[0] else 'google_universal_sentence_encoder'
	without_stop_words = args.without_stop_words
	model = args.model

	if name != 'dummy':
		utils.validate_option(constants.DATASETS.keys(), name, 'dataset')
		utils.validate_option(constants.ENCODERS, encoder, 'encoder')

	# show_stats = args.show_stats
	algorithm_name = args.algorithm
	utils.validate_option(constants.ALGORITHMS, algorithm_name, 'algorithm')

	# DBSCAN
	eps = args.eps
	min_samples = args.min_samples
 
	save_mode = args.save_mode
	plot = args.plot

	args.mapping = '3way' if name == 'SEB5' else None

	if config.RANDOM_SEED is not None:
		random.seed(config.RANDOM_SEED)

	# if show_stats:
	# 	show_dataset_stats(encoded_dataset, mapping, scale)
	# 	sys.exit(0)

	if model is None:
		model = constants.ENCODERS[encoder]['default_model']

	#if without_stop_words:
	#	encoder += '_without_stop_words'

	if plot is not None:
		utils.validate_option(['2d', '3d'], plot, 'plot')

	# normal usage, run 1 time
	result_set = run(name, question_id, encoder, model, vars(args))
	print('Dataset:', name)
	print('Question ID:', question_id)
	print('Encoder:', encoder)
	print('Model:', model)
	print('Algorithm:', args.algorithm)
	print('Cluster Number:', args.cluster_num)
	print('Accuracy:', result_set.results[0].accuracy)
	print('MSE:', result_set.results[0].mse)
