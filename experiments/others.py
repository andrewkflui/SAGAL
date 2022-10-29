import os, sys
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../'.format(os.sep)))


import datetime, random, xlsxwriter

from basis import config
from core import utils
from evaluation.results import load_dataset
from experiments.density_peak.gal.initialization import generate_param_conbinations
from clustering import run

def evaluate_batch(name, question_id, encoder, model, mapping, cluster_num_list, explicit_param_list,
	save_mode=1, **kwargs):
	args = kwargs
	result_dict = dict()
	if explicit_param_list is None or len(explicit_param_list) <= 0:
		explicit_param_list = [{'Sheet': 1}]
	
	for params in explicit_param_list:
		key = ''.join([str(p)[:(5 if not type(p) == bool else 1)] for p in params.values()])
		result_dict[key] = dict()
		for cluster_num in cluster_num_list:
			args['cluster_num'] = cluster_num
			args.update(params)
			result_set = run(name, question_id, encoder, model, args)			
			result_dict[key][cluster_num] = dict()
			result_dict[key][cluster_num]['result_set'] = result_set

	dataset, labels, text_list, reference_answer_labels, reference_answer_text_list \
		= load_dataset(name, question_id, encoder, model, mapping=mapping)
	return result_dict, labels, text_list, reference_answer_labels, reference_answer_text_list

def print_result_details(name, question_id, encoder, model, mapping, cluster_num_list, save_mode=1, **kwargs):
	result_sets = []
	args = kwargs
	for cluster_num in cluster_num_list:
		args['cluster_num'] = cluster_num
		result_set = run(name, question_id, encoder, model, args)
		result_sets.append(result_set)
	
	# dataset, labels, text_list, reference_answer_labels, reference_answer_text_list \
	# 	= load_dataset(name, question_id, encoder, model, mapping=args.get('mapping', None))
	dataset, labels, text_list, reference_answer_labels, reference_answer_text_list \
		= load_dataset(name, question_id, encoder, model, mapping=mapping)
	
	headers = ['Question ID', 'Grading Actions', 'Cluster #', 'Cluster Size', 'Centroid Index',
		'Ground Truth', 'Text']
	values = []

	folder_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	# path = '{}/data/results/{}/{}/{}/{}/{}'.format(config.ROOT_PATH, name, question_id, args.get('encoder'),
	# 	args.get('algorithm'), folder_name)
	path = '{}/data/results/{}/{}/{}/{}/{}'.format(config.ROOT_PATH, name, question_id, encoder,
		args.get('algorithm'), folder_name)
	path = utils.create_directories(path)

	file_name = '{}_centroids_{}'.format(args.get('algorithm'), '_'.join([str(c) for c in cluster_num_list]))
	workbook = xlsxwriter.Workbook('{}/{}.xlsx'.format(path, file_name))

	header_cell_format = workbook.add_format()
	header_cell_format.set_bold()
	header_cell_format.set_bg_color('#DDDDDD')

	cell_format = workbook.add_format()

	# overview
	sheet = workbook.add_worksheet('Overview')
	values = [['Dataset', name], ['Question', question_id], ['Encoder', encoder],
		['Algorithm', kwargs.get('algorithm')], ['Compress Method', args.get('compress_method', None)],
		['Compress Factor', args.get('compress_factor', None)], ['']]
	# values += [['Random State', result_sets[0].results[0].solution.random_state]]
	for pkey, pvalue in result_sets[0].results[0].solution.get_printable_params().items():
		values += [[pkey, pvalue]]
	grading_actions_values, accuracy_values = ['Grading Actions'], ['Accuracy']
	for i in range(len(result_sets)):
		grading_actions_values.append(cluster_num_list[i])
		accuracy_values.append(result_sets[i].results[0].accuracy)
	values += [grading_actions_values, accuracy_values]
	for i in range(len(values)):
		for j in range(len(values[i])):
			sheet.write(i, j, values[i][j], cell_format)

	if save_mode == 2:
		for result_set in result_sets:
			result = result_set.results[0]
			sheet = workbook.add_worksheet(str(len(result.clusters)))

			for j in range(len(headers)):
				sheet.write(0, j, headers[j], header_cell_format)

			i = 1
			for cluster in result.clusters:
				centroid_index = cluster.centroid_data_index
				values = [question_id, len(result.clusters), cluster.index, len(cluster.data_indices),
					centroid_index, labels[centroid_index], text_list[centroid_index]]
				for j in range(len(values)):
					sheet.write(i, j, values[j], cell_format)
				i += 1

	workbook.close()

def print_accuracy_batch(name, question_id, result_dict, labels, text_list, save_mode=0, folder_name=None):
	values = [['Dataset', name], ['Question', question_id],
		['Compress Method', args.get('compress_method', None)],
		['Compress Factor', args.get('compress_factor', None)], ['']]

	for _, v in result_dict.items():
		result = list(v.values())[0]['result_set'].results[0]
		# values += [['Random State', result.solution.random_state]]
		for pkey, pvalue in result.solution.get_printable_params().items():
			values += [[pkey, pvalue]]
		
		grading_actions = list(v.keys())
		cluster_num_row, accuracy_row = ['Cluster Num'], ['Accuracy']
		for cluster_num, v2 in v.items():
			cluster_num_row.append(cluster_num)
			accuracy_row.append(v2['result_set'].results[0].accuracy)
		values += [cluster_num_row, accuracy_row, ['']]

	if save_mode < 1:
		return values, None

	result_set = list(list(result_dict.values())[0].values())[0]['result_set']
	algorithm_name = result_set.algorithm_name
	random_state = getattr(result_set.results[0].solution, 'random_state', None)
	timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	path = '{}/data/results/{}/{}/{}/{}/{}'.format(config.ROOT_PATH, name, question_id, encoder,
		algorithm_name, folder_name if folder_name is not None else timestamp)
	path = utils.create_directories(path)
	
	file_name = 'accuracy_{}_{}_{}'.format(algorithm_name, question_id, timestamp)
	if random_state is not None:
		file_name += '_seed{}'.format(random_state)

	workbook = xlsxwriter.Workbook('{}/{}.xlsx'.format(path, file_name))
	sheet = workbook.add_worksheet(question_id[:31])
	sheet.set_column(0, 0, 30)
	for i in range(len(values)):
		for j in range(len(values[i])):
			sheet.write(i, j, values[i][j])

	workbook.close()
	return values, timestamp

if __name__ == '__main__':
	name = 'USCIS'
	question_id = '3'
	question_ids = list(str(i) for i in range(1, 9))

	# name = 'SEB2'
	# question_ids = ['HB_43', 'EV_25', 'EV_12b', 'EV_35a', 'WA_51', 'PS_24a']
	# question_ids = ['HB_53a2', 'DAMAGED_BULB_SWITCH_Q']
	# short answers
	# question_ids = ['HB_59a', 'WA_12c', 'HB_53a2', 'HB_24b1', 'BURNED_BULB_SERIES_Q2', 'DAMAGED_BULB_SWITCH_Q']
	# question_ids = ['HB_43', 'EV_25', 'EV_12b', 'EV_35a', 'WA_51', 'PS_24a', 'HB_53a2', 'DAMAGED_BULB_SWITCH_Q']
	# question_ids = ['DAMAGED_BULB_SWITCH_Q']

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
	# N >= 100
	# question_ids =  ['DESCRIBE_GAP_LOCATE_PROCEDURE_Q', 'EV_12b', 'EV_25', 'HB_24b1', 'HB_35', 'WA_52b']

	encoder = 'google_universal_sentence_encoder'
	model = None
	mapping = None
	compress_factor = None
	compress_method = None

	algorithm = 'birch'
	threshold = 0.5
	# threshold = config.EPSILON
	
	cluster_num_list = list(range(20, 110, 10)) + [120, 150]
	# normal usage, run 1 time
	# args = {'name': name, 'question_id': question_id, 'encoder': encoder, 'model': model,
	# 	'mapping': mapping, 'algorithm': algorithm, 'init_params': 'random', 'threshold': threshold,
	# 	'compress_factor': compress_factor, 'compress_method': compress_method}
	# print_result_details(cluster_num_list=cluster_num_list, save_mode=0, **args)

	for question_id in question_ids:
		args = {'name': name, 'question_id': question_id, 'encoder': encoder, 'model': model,
			'mapping': mapping, 'algorithm': algorithm, 'init_params': 'random', 'threshold': threshold,
			'compress_factor': compress_factor, 'compress_method': compress_method}
		
		folder_name = None
		for random_state in [0, 49, 97, 53, 5, 33,65, 62, 51, 100] if algorithm != 'birch' else [0]:
			args['random_state'] = random_state
			# explicit_param_list = generate_param_conbinations({
			# 	'covariance_type': ['full', 'tied', 'diag', 'spherical']
			# })
			explicit_param_list = None
			result_dict, labels, text_list, reference_answer_labels, reference_answer_text_list \
				= evaluate_batch(cluster_num_list=cluster_num_list, explicit_param_list=explicit_param_list, **args)
			_, timestamp = print_accuracy_batch(name, question_id, result_dict, labels, text_list,
				save_mode=1, folder_name=folder_name)
			folder_name = timestamp if folder_name is None else folder_name

