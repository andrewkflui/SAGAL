import os, sys
from pathlib import Path
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..{}..{}..{}'.format(os.sep, os.sep, os.sep)))

import datetime, random, copy, math, xlsxwriter

import numpy as np

from basis import config, version_variants

from core import utils
from core.algorithms import GAL, get_subspace_dimension_string

from evaluation.results import load_dataset
from clustering import load_data, run

# def get_subspace_dimension_string(subspace_param_list):
# 	dimensions = dict()
# 	encoders = []
# 	for param in subspace_param_list:
# 		encoder = param.get('encoder', '').lower()
# 		encoder = encoder.replace('google_universal_sentence_encoder', 'guse')
# 		random_reference_positions = param.get('random_reference_positions', [(0, 1)])
# 		if len(random_reference_positions) > 1:
# 			encoder += '_' + '_'.join([str(r[1]) for r in random_reference_positions])
# 		if not encoder in encoders:
# 			encoders.append(encoder)

# 		if param.get('weight', 1) <= 0:
# 			continue
# 		dimension = encoder + '_' + str(param.get('random_dimension', 'original'))
# 		random_selection_method = param.get('random_selection_method', None)
# 		horizontal_random_selection_number = param.get('horizontal_random_selection_number', dimension)
# 		if random_selection_method == 'vertical' or horizontal_random_selection_number == dimension:
# 			dimension = str(dimension) + ('H' if random_selection_method == 'horizontal' else 'V')
# 		elif random_selection_method == 'horizontal':
# 			# dimension += 'H'
# 			dimension = '{}_{}H'.format(horizontal_random_selection_number, dimension)
# 		if not dimension in dimensions:
# 			dimensions[dimension] = 1
# 		else:
# 			dimensions[dimension] += 1

# 	if len(encoders) > 1:
# 		encoder = '_'.join(encoders)
# 	else:
# 		encoder = encoders.pop()

# 	string = ''
# 	for dimension, count in dimensions.items():
# 		if string != '':
# 			string += '_'
# 		string += str(dimension)
# 		if count > 1:
# 			string += 'x{}'.format(count)
# 	return string if len(encoders) > 1 else string.replace(encoder + '_', ''), encoder

def calculate_batch_sizes(grading_actions, batch_size_one_length=5, batch_size=5):
	batch_sizes = [1] * batch_size_one_length
	remaining_grading_actions = grading_actions - batch_size_one_length
	while remaining_grading_actions - batch_size > 0:
		batch_sizes.append(batch_size)
		remaining_grading_actions -= batch_size
	batch_sizes.append(remaining_grading_actions)
	return batch_sizes

def create_algorithm(name, question_id, subspace_param_list, grading_actions, iterations, possible_grades,
	random_seed=None, load_subspaces=False, **kwargs):
	version = kwargs.get('version')
	version_variant = kwargs.get('version_variant')
	if version is not None and version_variant is not None:
		try:
			# version_variant = globals()['V{}_{}'.format(version, version_variant)](params=params)
			version_variant = getattr(version_variants, 'V{}_{}'.format(version, version_variant))(
				params=kwargs, subspace_param_list=subspace_param_list)
			kwargs.update(version_variant.params)
		except Exception as e:
			raise utils.InvalidNameError('version_variant', 'V{}_{}'.format(version, version_variant))

	if load_subspaces:
		path, file_name = get_saved_subspaces_path(name, question_id, subspace_param_list, kwargs,
			random_seed=random_seed)
		if path is not None and os.path.exists(path / file_name):
			kwargs['subspaces'] = utils.load(path / file_name)
			kwargs['reset_subspaces'] = False
	
	for params in subspace_param_list:
		random_dimension = params.get('random_dimension')
		if random_dimension is None:
			encoder, model = params['encoder'], params.get('model', None),
			compress_factor = params.get('compress_factor', None)
			compress_method = params.get('compress_method', 'pca')

			data, reference_data, compressed_data, compressed_reference_data \
				= load_data(name, question_id, encoder, model=model, compress_factor=compress_factor,
				compress_method=compress_method)
			data = compressed_data if compressed_data is not None else data
			params['data'] = data

	return GAL(subspace_param_list, grading_actions, iterations, possible_grades, random_seed=random_seed,
		**kwargs)

def get_saved_subspaces_path(name, question_id, subspace_param_list, params, random_seed=None):
	if random_seed is None:
		return None, None
	dimension_string, encoder = get_subspace_dimension_string(subspace_param_list)
	distance_function = params.get('distance_function', config.DISTANCE_FUNCTION)
	# encoder = subspace_param_list[0]['encoder'].lower()
	# file_name = 'gal_{}_{}_{}_{}_{}_seed{}.dat'.format(name.lower(), question_id,
	# 	'guse' if encoder == 'google_universal_sentence_encoder' else encoder, dimension_string,
	# 	distance_function, random_seed)
	# return Path('{}/data/algorithms/gal/{}/{}/{}/{}/{}'.format(config.ROOT_PATH, name, question_id,
	# 	subspace_param_list[0]['encoder'], dimension_string, distance_function)), file_name
	file_name = 'gal_{}_{}_{}_{}_{}_seed{}.dat'.format(name.lower(), question_id, encoder,
		dimension_string, distance_function, random_seed)
	return Path('{}/data/algorithms/gal/{}/{}/{}/{}/{}'.format(config.ROOT_PATH, name, question_id,
		encoder, dimension_string, distance_function)), file_name

def save_subspaces(name, question_id, subspace_param_list, grading_actions, iterations, possoble_labels,
	random_seed, args):
	if random_seed is None:
		return None
	path, file_name = get_saved_subspaces_path(name, question_id, subspace_param_list, args,
		random_seed=random_seed)
	algorithm = create_algorithm(name, question_id, subspace_param_list, grading_actions, iterations,
		possoble_labels, random_seed=random_seed, **args)
	utils.save(algorithm.subspaces, str(path), file_name)
	return algorithm

def get_distributions(labels, indices):
	indices = indices or list(range(len(labels)))
	
	distributions = dict()
	for index in indices:
		if not labels[index] in distributions:
			distributions[labels[index]] = 0
		distributions[labels[index]] += 1
	return distributions

def generate_param_conbinations(arg_dict):
	import itertools
	keys = list(arg_dict.keys())
	plain_list = list(itertools.product(*list(arg_dict.values())))
	return [{keys[i]: p[i] for i in range(len(keys))} for p in plain_list]

def print_selection_values(name, question_id, algorithm, gp_dict, labels, text_list, explicit_list=None,
	save_mode=0, folder_name=None):
	reduced_labels = np.array(labels)[algorithm.reduced_data_indices]
	reduced_text_list = np.array(text_list)[algorithm.reduced_data_indices]

	indices, reduced_indices = [], []
	if explicit_list is not None:
		if len(explicit_list) <= 0:
			return None, [], []
		
		if type(explicit_list[0]) == str:
			for text in explicit_list:
				for i in range(len(text_list)):
					if text_list[i] == text:
						indices.append(i)
						reduced_indices.append(algorithm.original_to_reduced_index_map[i])
						break
		else:
			indices = explicit_list
			reduced_indices = [algorithm.original_to_reduced_index_map[index] for index in explicit_list]
	
	headers = ['Question ID', 'Version']
	if algorithm.version_variant is not None:
		headers += ['Version Variant']
	headers += ['Data Dim', 'Random Seed', 'Grade Assignment Method']
	if algorithm.grade_assignment_method == 'parent_breaks':
		headers += ['Delta Link Threshold Factor', 'Delta Link Threshold',
			'Delta Link Threshold Window Size Factor', 'Delta Link Threshold Window Size']
	headers += ['Voting Ver.', 'Grading Actions', 'Batch Sizes', 'GP', 'Index']
	if algorithm.version != 6:
		headers += ['Selected by', 'RDExclude Factor At Selection', 'RDExclude At Selection']
	headers += ['True Grade', 'Text', 'RDGrade']
	if algorithm.version != 6:
		headers += ['MARlocal N', 'MARlocal RD Factor']
	headers += ['Representing N', 'Normalized Density', 'Spaciousness']
	headers += ['PGV' if algorithm.version == 6 else 'Information']
	headers += ['DISTTRUE', 'DELTA']
	for subspace in algorithm.subspaces:
		encoder = subspace.get_name()
		headers.append('DENlocal {}'.format(encoder))
		headers.append('Spaciousnesslocal {}'.format(encoder))
		headers.append('DELTAlocal {}'.format(encoder))
		headers.append('DISTTRUElocal {}'.format(encoder))
		headers.append('DISTTRUElocal {} Index'.format(encoder))
		headers.append('DISTTRUElocal {} Text'.format(encoder))
	# headers += ['MARlocal']
	# headers += ['MARlocal {}'.format(s.get_name()) for s in algorithm.subspaces]
	value_name = 'IG' if algorithm.version == 6 else 'MARlocal'
	headers += [value_name]
	headers += ['{} {}'.format(value_name, s.get_name()) for s in algorithm.subspaces]
	headers += ['MARglobal']
	for subspace in algorithm.subspaces:
		encoder = subspace.get_name()
		headers.append('RDlocal {}'.format(encoder))
		headers.append('MAX DELTAlocal {}'.format(encoder))

	lines, value_list = [], []
	for key, value in gp_dict.items():
		if explicit_list is None:
			indices = value['indices']
			reduced_indices = value['reduced_indices']
		selection_param_dict = value['selection_param_dict']
		informations = value['informations']
		nearest_ground_truth_distances = value['nearest_ground_truth_distances']
		local_nearest_ground_truth_distance_list = value['local_nearest_ground_truth_distance_list']
		local_nearest_ground_truth_index_map_list = value['local_nearest_ground_truth_index_map_list']
		local_marginalness_list = value['local_marginalness_list']
		global_marginalness = value['global_marginalness']

		normalized_local_delta_distance_list = [s.delta_distances / s.max_delta \
			for s in algorithm.subspaces]
		normalized_delta_distances = np.mean(normalized_local_delta_distance_list, axis=0)

		lines.append('Question ID: {}'.format(question_id))
		lines.append('Version: {}'.format(algorithm.version))
		if algorithm.version_variant is not None:
			lines.append('Version Variant'.format(algorithm.version_variant))
		lines.append('Data Dimension: {}'.format(algorithm.subspaces[0].data_dim))
		lines.append('Random Seed: {}'.format(algorithm.random_seed))
		lines.append('Grade Assignment Method: {}'.format(algorithm.grade_assignment_method))
		if algorithm.grade_assignment_method == 'parent_breaks':
			lines.append('Delta Link Threshold Factor: {}'.format(algorithm.delta_link_threshold_factor))
			lines.append('Delta Link Threshold: {}'.format(algorithm.delta_link_threshold))
			lines.append('Delta Link Threshold Window Size Factor: {}'.format(
				algorithm.delta_link_threshold_window_size_factor))
			lines.append('Delta Link Threshold window Size: {}'.format(
				algorithm.delta_link_threshold_window_size))
		lines.append('Voting Version: {}'.format(algorithm.voting_version))
		lines.append('Grading Actions: {}'.format(algorithm.grading_actions))
		lines.append('Batch Sizes: {}'.format(algorithm.batch_sizes))
		lines.append('Relevant Subspace Number: {}'.format(algorithm.relevant_subspace_number))
		if algorithm.version != 6:
			lines.append('Exclusion RD Factor: {}'.format(algorithm.exclusion_rd_factor))
			lines.append('Min. Exclusion RD Factor: {}'.format(algorithm.min_exclusion_rd_factor))
		lines.append('RD Cutoff Window Size: {}'.format(algorithm.subspaces[0].rd_cutoff_window_size))
		if algorithm.version != 6:
			lines.append('MARlocal N: {}'.format(algorithm.subspaces[0].marginalness_neighbourhood_size))
			lines.append('MARlocal RD Factor: {}'.format(algorithm.subspaces[0].marginalness_rd_factor))
		lines.append('')
		lines.append('[GP{}]'.format(key))
		for j in range(len(indices)):
			index = indices[j]
			reduced_index = reduced_indices[j]
			lines.append('{}, {}, {}'.format(index, labels[index], text_list[index]))
			if algorithm.version != 6:
				lines.append('Selected By: {}'.format(selection_param_dict[reduced_index]['method']))
				lines.append('Exclusion RD Factor At Selection: {}'.format(selection_param_dict[reduced_index]['exclusion_rd_factor']))
				lines.append('Exclusion RD At Selection: {}'.format(selection_param_dict[reduced_index]['exclusion_rd']))
			lines.append('RDGrade: {}'.format(algorithm.rd_cutoff))
			lines.append('Representing Number: {}'.format(algorithm.subspaces[0].get_data_representation_number(reduced_index, 2)))
			lines.append('Density: {}'.format(algorithm.densities[reduced_index]))
			lines.append('Spaciousness: {}'.format(algorithm.spaciousness[reduced_index]))
			lines.append('Information: {}'.format(informations[reduced_index]))
			lines.append('DISTTRUE: {}'.format(nearest_ground_truth_distances[reduced_index]))
			lines.append('DELTA: {}'.format(normalized_delta_distances[reduced_index]))
			local_marginalness = 0
			for s in range(len(algorithm.subspaces)):
				subspace = algorithm.subspaces[s]
				encoder = subspace.get_name()
				lines.append('{} Density: {}'.format(encoder, subspace.densities[reduced_index]))
				# lines.append('{} Spaciousness: {}'.format(encoder, subspace.spaciousness[reduced_index]))
				lines.append('{} Spaciousness: {}'.format(encoder, subspace.normalized_spaciousness[reduced_index]))
				d = '' if local_nearest_ground_truth_distance_list[s] is None \
					else local_nearest_ground_truth_distance_list[s][reduced_index]
				lines.append('{} DELTAlocal: {}'.format(encoder, normalized_local_delta_distance_list[s][reduced_index]))
				lines.append('{} DISTTRUElocal: {}'.format(encoder, d))

				local_nearest_ground_truth_index = local_nearest_ground_truth_index_map_list[s][reduced_index]
				local_nearest_ground_truth_index = '' if local_nearest_ground_truth_index < 0 \
					else algorithm.get_original_data_indices([local_nearest_ground_truth_index])[0]
				local_nearest_ground_truth_text = '' if local_nearest_ground_truth_index == '' \
					else text_list[local_nearest_ground_truth_index]
				lines.append('{} DISTTRUElocal Index: {}'.format(encoder, local_nearest_ground_truth_index))
				lines.append('{} DISTTRUElocal Text: {}'.format(encoder, local_nearest_ground_truth_text))
				local_marginalness += local_marginalness_list[s][reduced_index]

			local_marginalness /= len(algorithm.subspaces)
			# lines.append('MARlocal: {}'.format(local_marginalness))
			# for s in range(len(algorithm.subspaces)):
			# 	lines.append('{} MARlocal: {}'.format(encoder, local_marginalness_list[s][reduced_index]))
			value_name = 'IG' if algorithm.version == 6 else 'MARlocal' 
			lines.append('{}: {}'.format(value_name, local_marginalness))
			for s in range(len(algorithm.subspaces)):
				lines.append('{} {}: {}'.format(encoder, value_name, local_marginalness_list[s][reduced_index]))

			lines.append('MARglobal: {}'.format(global_marginalness[reduced_index]))
			
			lines += ['{} RD: {}'.format(s.get_name(), s.rd_cutoff) for s in algorithm.subspaces]
			lines += ['{} MAX DELTAlocal: {}'.format(s.get_name(), s.max_delta) \
				for s in algorithm.subspaces]
			lines.append('')
			
			values = [question_id, algorithm.version]
			if algorithm.version_variant is not None:
				values += [algorithm.version_variant]
			values += [algorithm.subspaces[0].data_dim, algorithm.random_seed,
				algorithm.grade_assignment_method]
			if algorithm.grade_assignment_method == 'parent_breaks':
				values += [algorithm.delta_link_threshold_factor, algorithm.delta_link_threshold,
					algorithm.delta_link_threshold_window_size_factor,
					algorithm.delta_link_threshold_window_size]
			values += [algorithm.voting_version, algorithm.grading_actions, str(algorithm.batch_sizes), key, index]
			if algorithm.version != 6:
				values += [selection_param_dict[reduced_index]['method'],
				selection_param_dict[reduced_index]['exclusion_rd_factor'],
				selection_param_dict[reduced_index]['exclusion_rd']]
			values += [labels[index], text_list[index]]
			nearest_ground_truth_distance = nearest_ground_truth_distances[reduced_index] \
				if not np.isinf(nearest_ground_truth_distances[reduced_index]) else 'inf'
			values += [algorithm.rd_cutoff]
			if algorithm.version != 6:
				values += [algorithm.subspaces[0].marginalness_neighbourhood_size,
					algorithm.subspaces[0].marginalness_rd_factor]
			values += [algorithm.subspaces[0].get_data_representation_number(reduced_index, 2),
				algorithm.densities[reduced_index], algorithm.spaciousness[reduced_index],
				informations[reduced_index], nearest_ground_truth_distance,
				normalized_delta_distances[reduced_index]]
			for s in range(len(algorithm.subspaces)):
				subspace = algorithm.subspaces[s]
				nearest_ground_truth_distance = local_nearest_ground_truth_distance_list[s][reduced_index] \
					if not np.isinf(local_nearest_ground_truth_distance_list[s][reduced_index]) else 'inf'
				local_nearest_ground_truth_index = local_nearest_ground_truth_index_map_list[s][reduced_index]
				local_nearest_ground_truth_index = '' if local_nearest_ground_truth_index < 0 \
					else algorithm.get_original_data_indices([local_nearest_ground_truth_index])[0]
				local_nearest_ground_truth_text = '' if local_nearest_ground_truth_index == '' \
					else text_list[local_nearest_ground_truth_index]
				values += [subspace.densities[reduced_index]]
				# values += [subspace.spaciousness[reduced_index]]
				values += [subspace.normalized_spaciousness[reduced_index]]
				values += [normalized_local_delta_distance_list[s][reduced_index],
					nearest_ground_truth_distance, local_nearest_ground_truth_index,
					local_nearest_ground_truth_text]
			values += [local_marginalness]
			values += [local_marginalness_list[s][reduced_index] for s in range(len(algorithm.subspaces))]
			values += [global_marginalness[reduced_index]]
			for subspace in algorithm.subspaces:
				values += [subspace.rd_cutoff, subspace.max_delta]
			value_list.append(values)
		lines.append('')

	value_list.insert(0, headers)

	if save_mode <= 0:
		map(print, lines)
		return None, lines, value_list

	timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	folder_name = folder_name or timestamp
	path = '{}/data/results/{}/{}/multiple/{}/{}'.format(config.ROOT_PATH, name, question_id,
		algorithm.name, folder_name)
	path = utils.create_directories(path)
	
	file_name = 'selection_values_v{}_{}_{}'.format(algorithm.version, algorithm.grading_actions,
		timestamp)
	if algorithm.random_seed is not None:
		file_name += '_seed{}'.format(algorithm.random_seed)

	if save_mode == 1 or save_mode == 3:
		utils.write(str(path), '{}.txt'.format(file_name), lines)

	if save_mode < 2:
		return folder_name, lines, value_list

	workbook = xlsxwriter.Workbook('{}/{}.xlsx'.format(path, file_name))
	sheet = workbook.add_worksheet('{}'.format('{} {} seed {}'.format(question_id[:10],
		algorithm.grading_actions, algorithm.random_seed)))

	header_cell_format = workbook.add_format()
	header_cell_format.set_bold()
	header_cell_format.set_bg_color('#DDDDDD')

	cell_format = workbook.add_format()
	
	for i in range(len(value_list)):
		for j in range(len(value_list[i])):
			if i == 0 and 'Text' in value_list[i][j]:
				sheet.set_column(j, j, 50)
			sheet.write(i, j, value_list[i][j], header_cell_format if i == 0 else cell_format)

	workbook.close()
	return folder_name, lines, value_list

def print_relevant_subspace_values(name, question_id, algorithms, labels, text_list, save_mode=0,
	folder_name=None):
	reduced_labels = np.array(labels)[algorithms[0].reduced_data_indices]
	reduced_text_list = np.array(text_list)[algorithms[0].reduced_data_indices]

	headers = ['Question ID', 'Distance Function', 'Version', 'Random Seed', 'Relevant Subspace Nummber', 'Index',
		'True Grade', 'Text', 'Relevant Subspaces', 'AVG DenRelevant', 'AVG SpacRelevant', 'MIN DenRelevant',
		'MAX DenRelevant', 'MIN SpacRelevant', 'MAX SpacRelevant', 'AVG DenAll', 'AVG SpacAll', 'MIN DenAll',
		'MAX DenAll', 'MIN SpacAll', 'MAX SpacAll']
	
	algorithm_value_list = []
	
	for algorithm in algorithms:
		value_list = []
		for i in range(algorithm.reduced_num_data):
			values = [question_id, algorithm.distance_function, algorithm.version, algorithm.random_seed,
				algorithm.relevant_subspace_number]
			values += [i, reduced_labels[i], reduced_text_list[i]]
			values += [str(algorithm.relevant_subspaces[i]), algorithm.densities[i], algorithm.spaciousness[i]]
			relevant_densities, relevant_spaciousness = [], []
			for s in algorithm.relevant_subspaces[i]:
				# relevant_densities.append(algorithm.subspaces[s].densities[i])
				# relevant_spaciousness.append(algorithm.subspaces[s].spaciousness[i])
				relevant_densities.append(algorithm.subspaces[s].normalized_densities[i])
				relevant_spaciousness.append(algorithm.subspaces[s].normalized_spaciousness)
			values += [min(relevant_densities), max(relevant_densities), min(relevant_spaciousness),
				max(relevant_spaciousness)]

			density_list, spaciousness_list = [], []
			for s in algorithm.subspaces:
				# density_list.append(s.densities / s.densities.max())
				# spaciousness_list.append(s.spaciousness / s.spaciousness.max())
				density_list.append(s.normalized_densities)
				spaciousness_list.append(s.normalized_spaciousness)
			densities = algorithm.calculate_weighted_average_subspace_values(density_list)
			densities = np.log(densities) / np.log(math.exp(1))
			densities -= densities.min()
			densities /= densities.max()
			spaciousness = algorithm.calculate_weighted_average_subspace_values(spaciousness_list)
			density_list = np.array(density_list)
			spaciousness_list = np.array(spaciousness_list)
			values += [densities[i], spaciousness[i], min(density_list[:, i]), max(density_list[:, i]),
				min(spaciousness_list[:, i]), max(spaciousness_list[:, i])]
			value_list.append(values)

		value_list.insert(0, headers)
		algorithm_value_list.append(value_list)
		
	if save_mode <= 0:
		return algorithm_value_list

	folder_name = folder_name or datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	path = '{}/data/results/{}/{}/multiple/{}/{}'.format(config.ROOT_PATH, name, question_id,
		algorithm.name, folder_name)
	path = utils.create_directories(path)
	
	file_name = 'subspace_relevant_values_v{}_r{}_{}_seeds_{}'.format(algorithm.version,
		algorithm.relevant_subspace_number,
		algorithm.distance_function, '_'.join([str(algorithm.random_seed) for algorithm in algorithms]))
	workbook = xlsxwriter.Workbook('{}/{}.xlsx'.format(path, file_name))
	
	for a in range(len(algorithms)):
		sheet = workbook.add_worksheet('{}'.format('{} seed {}'.format(question_id,
			algorithms[a].random_seed)))

		header_cell_format = workbook.add_format()
		header_cell_format.set_bold()
		header_cell_format.set_bg_color('#DDDDDD')

		cell_format = workbook.add_format()
		
		value_list = algorithm_value_list[a]
		for i in range(len(value_list)):
			for j in range(len(value_list[i])):
				if i == 0 and 'Text' in value_list[i][j]:
					sheet.set_column(j, j, 50)
				sheet.write(i, j, value_list[i][j], header_cell_format if i == 0 else cell_format)
	workbook.close()
	
	return algorithm_value_list

def print_delta_link_thresholds(question_params, subspace_param_list, random_seeds, load_subspaces=False,
	**args):
	headers = ['Dataset', 'Question ID', 'Data Number', 'Data Numer (Encap)', 'Window Size Factor',
		'Random Seed', 'RD', 'Window Size', 'Average Start Position', 'Average Start Rank',
		'Average Threshold (Start)', 'Average Factor to RD (Start)', 'Average Median Position',
		'Average Median Rank', 'Average Threshold (Median)', 'Average Factor to RD (Median)',
		'Average End Position', 'Average End Rank', 'Average Threshold (End)', 'Average Factor to RD (End)',
		'Average Curvatures', 'Window Sizes', 'Positions (Start)', 'Thresholds (Start)',
		'Factors to RD (Start)', 'Positions (Median)', 'Thresholds (Median)', 'Factors to RD (Median)']

	value_list = []
	for name, question_id in question_params:
		for random_seed in random_seeds:
			print('Running {} {}, Random Seed: {}'.format(name, question_id, random_seed))
			algorithm = create_algorithm(name, question_id, subspace_param_list, grading_actions=10,
				iterations=10, possible_grades=['Correct', 'Wrong'],
				random_seed=random_seed, load_subspaces=load_subspaces, **args)
			values = [name, question_id, algorithm.num_data, algorithm.reduced_num_data,
				algorithm.delta_link_threshold_window_size_factor, algorithm.random_seed,
				algorithm.rd_cutoff]
			window_sizes = []
			start_positions, start_ranks, thresholds_start, factors_to_rd_start = [], [], [], []
			median_positions, median_ranks, thresholds_median, factors_to_rd_median = [], [], [], []
			end_positions, end_ranks, thresholds_end, factors_to_rd_end = [], [], [], []
			curvatures = []
			for subspace in algorithm.subspaces:
				if subspace.weight <= 0:
					continue
				w = subspace.delta_link_threshold_window_size or 0
				s = subspace.delta_link_threshold_start_position or 0
				m = s + int(w / 2)
				e = s + w
				window_sizes.append(w)
				start_positions.append(s)
				start_ranks.append(subspace.spaciousness_ranks[s])
				thresholds_start.append(subspace.ranked_spaciousness[s])
				factors_to_rd_start.append(thresholds_start[-1] / subspace.rd_cutoff)
				median_positions.append(m)
				median_ranks.append(subspace.spaciousness_ranks[m])
				thresholds_median.append(subspace.ranked_spaciousness[m])
				factors_to_rd_median.append(thresholds_median[-1] / subspace.rd_cutoff)
				end_positions.append(e)
				end_ranks.append(subspace.spaciousness_ranks[e])
				thresholds_end.append(subspace.ranked_spaciousness[e])
				factors_to_rd_end.append(thresholds_end[-1] / subspace.rd_cutoff)
				curvatures.append(subspace.delta_link_threshold_curvature or 0)
			values += [np.mean(window_sizes), np.mean(start_positions), np.mean(start_ranks),
				np.mean(thresholds_start), np.mean(factors_to_rd_start), np.mean(median_positions),
				np.mean(median_ranks), np.mean(thresholds_median), np.mean(factors_to_rd_median),
				np.mean(end_positions), np.mean(end_ranks), np.mean(thresholds_end),
				np.mean(factors_to_rd_end), np.mean(curvatures)]
			values += [str(window_sizes), str(start_positions), str(thresholds_start),
				str(factors_to_rd_start), str(median_positions), str(thresholds_median),
				str(factors_to_rd_median)]
			value_list.append(values)
		value_list.append([''])
	value_list.insert(0, headers)

	# return None, value_list

	# folder_name = folder_name or datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	folder_name = timestamp
	# path = '{}/data/results/{}/{}/multiple/{}/{}'.format(config.ROOT_PATH, name, question_id,
	# 	algorithm.name, folder_name)
	# path = Path('D:/data/Desktop/')
	path = Path('/Users/user/Desktop/')
	path = utils.create_directories(path)
	
	file_name = 'delta_link_thresholds_{}'.format(timestamp)
	workbook = xlsxwriter.Workbook('{}/{}.xlsx'.format(path, file_name))

	sheet = workbook.add_worksheet('Sheet1')

	header_cell_format = workbook.add_format()
	header_cell_format.set_bold()
	header_cell_format.set_bg_color('#DDDDDD')

	cell_format = workbook.add_format()
	
	for i in range(len(value_list)):
		for j in range(len(value_list[i])):
			if i == 0 and 'Text' in value_list[i][j]:
				sheet.set_column(j, j, 50)
			sheet.write(i, j, value_list[i][j], header_cell_format if i == 0 else cell_format)
	workbook.close()

	return folder_name, value_list

def print_wrongly_graded_answers(name, question_id, algorithms, labels, text_list, gp_dict_list, save_mode=0,
	folder_name=None):
	reduced_labels = np.array(labels)[algorithms[0].reduced_data_indices]
	reduced_text_list = np.array(text_list)[algorithms[0].reduced_data_indices]

	unique_labels = np.unique(labels).tolist()

	algorithm_value_list = []
	for a in range(len(algorithms)):
		algorithm = algorithms[a]
		gp_dict = gp_dict_list[a]

		headers = ['Question ID', 'Distance Function', 'Version']
		if algorithm.version_variant is not None:
			headers += ['Version Variant']
		headers +=['Grading Actions', 'Batch Sizes', 'Random Seed', 'Relevant Subspace Number']
		if algorithm.version != 6:
			headers += ['Exclusion RD Deduction Factor', 'Min. Exclusion RD Factor']
		headers += ['Grade Assignment Method']
		if algorithm.grade_assignment_method == 'parent_breaks':
			headers += ['Delta Link Threshold Factor', 'Delta Link Threshold',
				'Delta Link Threshold Window Size Factor', 'Delta Link Threshold Window Size']
		headers += ['Voting Ver.']
		headers += ['Index', 'True Grade', 'Text', 'Assigned Grade']
		headers += ['{} Count'.format(l) for l in unique_labels]
		headers += ['Representing N', 'Densities', 'Spaciousness', 'DISTTrue']
		headers += ['Count(Source Density == 0)', 'Count(Parent == Ground Truth)']
		
		subspace_source_types = []
		for subspace in algorithm.subspaces:
			subspace_source_types += [subspace.source_map[index][1] for index in range(algorithm.reduced_num_data)]
		subspace_source_types = sorted(list(set(subspace_source_types)))
		subspace_source_types.remove('self')

		# for i in range(algorithm.num_data):
		# 	reduced_index = algorithm.original_to_reduced_index_map[i]
		# 	if reduced_labels[reduced_index] != labels[i]:
		# 		print('Original, {}, Label: {}, Text: {}'.format(i, labels[i], text_list[i]))
		# 		print('Reduced, {}, Label: {}, Text: {}'.format(reduced_index, reduced_labels[reduced_index],
		# 			reduced_text_list[reduced_index]))

		# wrong_count = 0
		# for i in range(algorithm.reduced_num_data):
		# 	if algorithm.true_grade_labels[i] != reduced_labels[i]:
		# 		wrong_count += 1
		# print('known count', len(algorithm.known_data_labels))
		# print('wrong_count', wrong_count)
		# import sys
		# sys.exit()
		
		value_list = []
		indices = []
		for i in range(algorithm.reduced_num_data):
			if algorithm.true_grade_labels[i] != reduced_labels[i]:
				values = [question_id, algorithm.distance_function, algorithm.version]
				if algorithm.version_variant is not None:
					values += [algorithm.version_variant]
				values += [algorithm.grading_actions, str(algorithm.batch_sizes), algorithm.random_seed,
					algorithm.relevant_subspace_number]
				if algorithm.version != 6:
					values += [algorithm.exclusion_rd_deduction_factor, algorithm.min_exclusion_rd_factor]
				values += [algorithm.grade_assignment_method]
				if algorithm.grade_assignment_method == 'parent_breaks':
					values += [algorithm.delta_link_threshold_factor, algorithm.delta_link_threshold,
						algorithm.delta_link_threshold_window_size_factor,
						algorithm.delta_link_threshold_window_size]
				values += [algorithm.voting_version]
				values += [i, str(reduced_labels[i]), reduced_text_list[i], str(algorithm.true_grade_labels[i])]
				if algorithm.relevant_subspaces is not None:
					subspace_index_list = algorithm.relevant_subspaces[i]
				else:
					subspace_index_list = [s for s in range(len(algorithm.subspaces)) if algorithm.subspaces[s].weight > 0]
				label_counts = np.zeros(shape=len(unique_labels), dtype=int)
				source_den_0_count, source_true_count = 0, 0
				source_types = [0] * len(subspace_source_types)

				assigned_as_outlier_count = 0
				for s in subspace_index_list:
					subspace = algorithm.subspaces[s]
					label_counts[unique_labels.index(subspace.true_grade_labels[i])] += 1
					source = subspace.source_map[i][0]
					if subspace.source_map[i][1] == 'outlier':
						assigned_as_outlier_count += 1
					if subspace.normalized_densities[source] == 0.:
						source_den_0_count += 1
					# if source in algorithm.reduced_known_data_labels:
					if subspace.source_map[i][2]:
						source_true_count += 1
					source_types[subspace_source_types.index(subspace.source_map[i][1])] += 1
				values += label_counts.tolist()

				values += [algorithm.subspaces[0].get_data_representation_number(i, 2),
					algorithm.densities[i], algorithm.spaciousness[i]]
				values += [gp_dict[max(list(gp_dict.keys()))]['nearest_ground_truth_distances'][i]]

				values += [source_den_0_count, source_true_count]
				values += source_types
				
				value_list.append(values)
				indices.append(i)
		print(indices)

		value_list.insert(0, headers + ['Source \'{}\''.format(t) for t in subspace_source_types])
		algorithm_value_list.append(value_list)

	if save_mode <= 0:
		return None, algorithm_value_list

	timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	folder_name = folder_name or timestamp
	path = '{}/data/results/{}/{}/multiple/{}/{}'.format(config.ROOT_PATH, name, question_id,
		algorithm.name, folder_name)
	path = utils.create_directories(path)

	file_name = 'wrongly_graded_values_v{}_r{}_{}_seeds_{}_{}'.format(algorithm.version,
		algorithm.relevant_subspace_number,
		algorithm.distance_function, '_'.join([str(algorithm.random_seed) for algorithm in algorithms]),
		timestamp)
	workbook = xlsxwriter.Workbook('{}/{}.xlsx'.format(path, file_name))
	
	for a in range(len(algorithms)):
		sheet = workbook.add_worksheet('{}'.format('{} seed {} {}'.format(question_id[:10],
			algorithms[a].random_seed, algorithms[a].grading_actions)))

		header_cell_format = workbook.add_format()
		header_cell_format.set_bold()
		header_cell_format.set_bg_color('#DDDDDD')

		cell_format = workbook.add_format()
		
		value_list = algorithm_value_list[a]
		for i in range(len(value_list)):
			for j in range(len(value_list[i])):
				if i == 0 and 'Text' in value_list[i][j]:
					sheet.set_column(j, j, 50)
				sheet.write(i, j, value_list[i][j], header_cell_format if i == 0 else cell_format)
	workbook.close()

	return folder_name, algorithm_value_list

def print_accuracy_batch(name, question_id, result_dict, labels, text_list, reduced_labels,
	reduced_text_list, explicit_param_list=[], save_mode=0, folder_name=None):
	explicit_params = explicit_param_list[0] if explicit_param_list is not None \
		and len(explicit_param_list) > 0 else []
	a = list(list(result_dict.values())[0].values())[0]['algorithm']
	values = [['Dataset', name], ['Question', question_id], ['Total', len(labels)],
		['Total (Encap)', len(reduced_labels)]]
	for pkey, pvalue in a.get_printable_params().items():
		# values += [[pkey, pvalue]]
		values += [[pkey] + [str(o['algorithm'].get_printable_params()[pkey]) \
			for o in list(result_dict.values())[0].values()]]

	for _, v in result_dict.items():
		a = list(v.values())[0]['algorithm']
		for pkey, pvalue in a.get_printable_params().items():
			if pkey.replace(' ', '_').lower() in explicit_params:
				values += [[pkey] + ([str(pvalue)] if isinstance(pvalue, list) else [pvalue])]
		for pkey, pvalue in a.get_subspace_printable_params().items():
			values += [[pkey] + (pvalue if isinstance(pvalue, list) else [pvalue])]
		# for pkey, pvalue in a.get_subspace_printable_params().items():
		# 	if pkey.replace(' ', '_').lower() in explicit_params:
		# 		values += [[pkey] + (pvalue if isinstance(pvalue, list) else [pvalue])]
		
		grading_actions = list(v.keys())
		values += [['Grading Actions'] + grading_actions]
		if a.version == 5.5:
			phrase1_selection_list = ['Phrase1 Selection Grading Actions']
			phrase2_selection_list = ['Phrase2 Selection Grading Actions']
			for k in grading_actions:
				phrase1_selection_list.append(v[k]['algorithm'].phrase1_selection_grading_actions)
				phrase2_selection_list.append(v[k]['algorithm'].phrase2_selection_grading_actions)
			values += [phrase1_selection_list, phrase2_selection_list]
		# values += [['Accuracy'] + [v[k]['result'].accuracy for k in grading_actions], ['']]
		accuracy_list = ['Accuracy']
		tp_list, tn_list, fp_list, fn_list, mse_list = ['TP'], ['TN'], ['FP'], ['FN'], ['MSE']
		for k in grading_actions:
			result = v[k]['result']
			accuracy_list.append(result.accuracy)
			tp_list.append(result.tp)
			tn_list.append(result.tn)
			fp_list.append(result.fp)
			fn_list.append(result.fn)
			mse_list.append(result.mse)
		# values += [accuracy_list, tp_list, tn_list, fp_list, fn_list, ['']]
		values += [accuracy_list]
		if len(np.unique(labels)) <= 2:
			values += [tp_list, tn_list, fp_list, fn_list]
		values += [mse_list, ['']]
		# values += [['Time Used'] + [str(v[k]['time_used']) for k in grading_actions], ['']]

	if save_mode < 1:
		return values, None

	timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
	path = '{}/data/results/{}/{}/multiple/{}/{}'.format(config.ROOT_PATH, name, question_id,
		a.name, folder_name if folder_name is not None else timestamp)
	path = utils.create_directories(path)
	
	file_name = 'accuracy_{}_v_{}_w_{}_vo_{}_{}'.format(question_id, a.version,
		a.subspaces[0].rd_cutoff_window_size, a.voting_version, timestamp)
	if a.random_seed is not None:
		file_name += '_seed{}'.format(a.random_seed)

	workbook = xlsxwriter.Workbook('{}/{}.xlsx'.format(path, file_name))
	sheet = workbook.add_worksheet(question_id[:31])
	sheet.set_column(0, 0, 30)
	for i in range(len(values)):
		for j in range(len(values[i])):
			sheet.write(i, j, values[i][j])
	
	if save_mode >= 2:
		header_cell_format = workbook.add_format()
		header_cell_format.set_bold()
		header_cell_format.set_bg_color('#DDDDDD')
		cell_format = workbook.add_format()
		
		for key, result in result_dict.items():
			grading_actions = max(list(result.keys()))
			v = result[grading_actions]
			key = key.replace('*', 'x')
			key = key.replace('/', '_')

			if save_mode == 2:
				sheet = workbook.add_worksheet('{} {}'.format(key, grading_actions))
				_, values = print_selection_values(name, question_id, v['algorithm'], v['gp_dict'], labels,
					text_list, explicit_list=None, save_mode=0)
				for i in range(len(values)):
					for j in range(len(values[i])):
						sheet.write(i, j, values[i][j], header_cell_format if i == 0 else cell_format)

			if save_mode == 3:
				sheet = workbook.add_worksheet('{} {}'.format(key, v['algorithm'].random_seed))
				_, values = print_relevant_subspace_values(name, question_id, v['algorithm'],
					labels, text_list, save_mode=0)
			
			# if save_mode == 3:
			# 	sheet = workbook.add_worksheet('{} outliers'.format(key))
			# 	headers, values = None, []
			# 	for s in range(len(v['algorithm'].subspaces)):
			# 		for g in v['gp_dict'].keys():
			# 			_, outlier_values = print_selection_values(name, question_id, v['algorithm'], v['gp_dict'], labels,
			# 			text_list, explicit_list=v['gp_dict'][g]['outlier_indices_list'][s], save_mode=0)
			# 			if headers is None and len(outlier_values) > 0:
			# 				headers = outlier_values[0]
			# 			for l in outlier_values:
			# 				l.insert(headers.index('Index'), get_subspace_name(v['algorithm'].subspaces[s]))
			# 		values += outlier_values[1:]
			# 	values = headers + values
			# 	for i in ranage(len(values)):
			# 		for j in range(len(values[i])):
			# 			sheet.write(i, j, values[i][j], header_cell_format if i == 0 else cell_format)

	workbook.close()
	return values, timestamp

def evaluate(name, question_id, mapping, subspace_param_list, grading_actions, iterations,
	possible_grades, random_seed=None, load_subspaces=False, **kwargs):
	start_time = datetime.datetime.now()
	
	batch_size = kwargs.get('batch_size', None)
	# iterations = int(grading_actions / batch_size) if batch_size is not None else iterations
	iterations = kwargs.get('iterations', None)
	algorithm = create_algorithm(name, question_id, subspace_param_list, grading_actions=grading_actions,
		iterations=iterations, possible_grades=possible_grades, random_seed=random_seed,
		load_subspaces=load_subspaces, **kwargs)

	param = subspace_param_list[0]
	dataset, labels, text_list, _, _ = load_dataset(name, question_id, param['encoder'], None,
		mapping=mapping)
	reduced_labels = np.array(labels)[algorithm.reduced_data_indices]
	reduced_text_list = np.array(text_list)[algorithm.reduced_data_indices]

	global_indices, reduced_global_indices = [], []

	i = 1
	gp_dict = dict()
	while True:
		indices, reduced_indices, selection_param_dict, informations, nearest_ground_truth_distances, \
		local_nearest_ground_truth_distance_list, local_nearest_ground_truth_index_map_list, \
		local_marginalness_list, global_marginalness, outlier_indices_list = algorithm.run()

		if len(indices) <= 0:
			break

		print('GP{}, valuable indices distributions: {}'.format(i, get_distributions(labels, indices)))
		print('Current Iteration: {}, Total Iterations: {}'.format(algorithm.current_iteration,
			algorithm.iterations))
		algorithm.mark_answers({k: labels[k] for k in indices})

		gp_dict[i] = dict()
		gp_dict[i]['indices'] = np.copy(indices)
		gp_dict[i]['reduced_indices'] = np.copy(reduced_indices)
		gp_dict[i]['selection_param_dict'] = copy.copy(selection_param_dict)
		gp_dict[i]['informations'] = np.copy(informations)
		gp_dict[i]['nearest_ground_truth_distances'] = np.copy(nearest_ground_truth_distances)
		gp_dict[i]['local_nearest_ground_truth_distance_list'] = np.copy(local_nearest_ground_truth_distance_list)
		gp_dict[i]['local_nearest_ground_truth_index_map_list'] = np.copy(local_nearest_ground_truth_index_map_list)
		gp_dict[i]['local_marginalness_list'] = np.copy(local_marginalness_list)
		gp_dict[i]['global_marginalness'] = np.copy(global_marginalness)
		gp_dict[i]['outlier_indices_list'] = np.copy(outlier_indices_list)
		i += 1
		
		global_indices += indices
		reduced_global_indices += reduced_indices

	results = algorithm.get_result()
	time_used = datetime.datetime.now() - start_time

	print('[Global]')
	for index in global_indices:
		print('{}, {}, {}'.format(index, labels[index], text_list[index]))
	print()

	print('Question: {}, Distributions: {}'.format(question_id, get_distributions(labels, None)))
	print('Reduced Num of Data: {}'.format(algorithm.reduced_num_data))
	distributions = dict()
	for k, v in algorithm.known_data_labels.items():
		if not v in distributions:
			distributions[v] = 1
		else:
			distributions[v] += 1
	
	if algorithm.random_seed is not None:
		print('Random Seed: {}'.format(algorithm.random_seed))
	print('Version: {}'.format(algorithm.version))
	if algorithm.version_variant is not None:
		print('Version Variant: {}'.format(algorithm.version_variant))
	print('Voting Version: {}'.format(algorithm.voting_version))
	print('Grading Actions: {}'.format(algorithm.grading_actions))
	print('Known Distributions: {}'.format(distributions))
	print('Unique: {}'.format(np.unique(results, return_counts=True)))
	print('Time Used: {}'.format(time_used))

	from evaluation.algorithms import GALSolution
	from evaluation.results import GALResult
	solution = GALSolution(results)
	result = GALResult(solution, dataset, labels)
	print('RD: {}'.format(algorithm.rd_cutoff))
	if algorithm.grade_assignment_method == 'parent_breaks':
		print('Delta link Threshold: {}'.format(algorithm.delta_link_threshold))
	print('TP: {}, FP: {}, TN: {}, FN: {}'.format(result.tp, result.fp, result.tn, result.fn))
	print('Precision: {}'.format(result.precision))
	print('Recall: {}'.format(result.recall))
	print('F1-Score: {}'.format(result.f1_score))
	print('Accuracy: {}'.format(result.accuracy))
	print('MSE: {}'.format(result.mse))

	return algorithm, labels, text_list, reduced_labels, reduced_text_list, result, gp_dict, time_used

def evaluate_batch(name, question_id, mapping, subspace_param_list, grading_actions_list,
	batch_size_one_length=5, batch_size=5, possible_grades=['Correct', 'Wrong'], explicit_param_list=[],
	random_seed=None, load_subspaces=False):
	result_dict = dict()
	
	if explicit_param_list is None or len(explicit_param_list) <= 0:
		explicit_param_list = generate_param_conbinations({'Sheet': [1]})
	
	for params in explicit_param_list:
		key = ''.join([str(p)[:(5 if not type(p) == bool else 1)] for p in params.values()])
		
		if key in result_dict:
			count = 1
			k = '{}{}'.format(count, key)
			while k in result_dict:
				k = '{}{}'.format(count, key)
				count += 1
			key = k
		result_dict[key] = dict()

		init_subspace_param_list = None
		for grading_actions in grading_actions_list:
			if batch_size_one_length > 0:
				iterations = None
				batch_sizes = calculate_batch_sizes(grading_actions, batch_size_one_length, batch_size)
				params['batch_sizes'] = batch_sizes
			else:
				# iterations = int(grading_actions / batch_size)
				iterations = None
				params['batch_size'] = batch_size
			
			if 'random_seed' not in params:
				algorithm, labels, text_list, reduced_labels, reduced_text_list, result, gp_dict, time_used \
					= evaluate(name, question_id, mapping,
						init_subspace_param_list if init_subspace_param_list is not None else subspace_param_list,
						grading_actions, iterations, possible_grades, random_seed=random_seed,
						load_subspaces=load_subspaces, **params)
			else:
				algorithm, labels, text_list, reduced_labels, reduced_text_list, result, gp_dict, time_used \
					= evaluate(name, question_id, mapping,
						init_subspace_param_list if init_subspace_param_list is not None else subspace_param_list,
						grading_actions, iterations, possible_grades, load_subspaces=load_subspaces, **params)

			if algorithm.version == 4:
				algorithm.reset_subspaces = True
			
			if init_subspace_param_list is None:
				init_subspace_param_list = copy.copy(algorithm.subspace_param_list)
			
			result_dict[key][grading_actions] = dict()
			result_dict[key][grading_actions]['algorithm'] = algorithm
			result_dict[key][grading_actions]['result'] = result
			result_dict[key][grading_actions]['gp_dict'] = gp_dict
			result_dict[key][grading_actions]['time_used'] = time_used
			result_dict[key][grading_actions]['init_subspace_param_list'] = copy.copy(init_subspace_param_list)
	
	return result_dict, labels, text_list, reduced_labels, reduced_text_list

def run_comparisons(name, question_id, encoder, model, mapping, grading_actions_list,
	comparison_algorithms=['kmeans', 'birch'], save_mode=0):
	values = [['Question ID', question_id], ['']]

	section1_headers = ['Clustering Algorithm', 'Grading Standard', 'Grading Action Per Clusters']
	section2_headers = ['Grading Actions', 'Actual Grading Actions',
		'Correct Grading (Original) (Count)', 'Error Grading (Original) (Count',
		'Grading Accuracy (Original)']
	
	for comparison_algorithm in comparison_algorithms:
		args = dict()
		args['algorithm'] = comparison_algorithm
		v  = [comparison_algorithm, 'Centroid', 1]
		for i in range(len(section1_headers)):
			values.append([section1_headers[i], v[i]])
		values.append([''])
		
		row = len(values)
		for h in section2_headers:
			values.append([h, ''])
		values.append([''])

		for grading_actions in grading_actions_list:
			args['cluster_num'] = grading_actions
			result_set = run(name, question_id, encoder, model, args)
			values[row].append(grading_actions)
			values[row+1].append(len(result_set.results[0].clusters))
			values[row+2].append(result_set.results[0].dataset.num_data - result_set.results[0].error_count)
			values[row+3].append(result_set.results[0].error_count)
			values[row+4].append(result_set.results[0].accuracy)
		args.pop('algorithm')

	values.insert(1, ['Total (Original)', result_set.results[0].dataset.num_data])

	if save_mode > 0:
		folder_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
		path = '{}/data/results/{}/{}/multiple/{}/{}'.format(config.ROOT_PATH, name, question_id,
			'gal', folder_name)
		path = utils.create_directories(path)

		file_name = 'other_algorithms_{}_{}'.format('_'.join(comparison_algorithm), folder_name)
		workbook = xlsxwriter.Workbook('{}/{}.xlsx'.format(path, file_name))
		sheet = workbook.add_worksheet('Sheet1')

		for i in range(len(values)):
			for j in range(len(values[i])):
				if j == 0:
					sheet.set_column(j, j, 50)
				sheet.write(i, j, values[i][j])
		workbook.close()

	return values

if __name__ == '__main__':
	name = 'USCIS'
	question_id = '3'
	# name = 'SEB2'
	# question_id = 'BULB_C_VOLTAGE_EXPLAIN_WHY1'
	mapping = None
	subspace_param_list = [
		{'encoder': 'google_universal_sentence_encoder'},
		{'encoder': 'google_universal_sentence_encoder', 'compress_factor': 128},
		{'encoder': 'google_universal_sentence_encoder', 'compress_factor': 32},
		# {'encoder': 'google_universal_sentence_encoder'},
		# {'encoder': 'google_universal_sentence_encoder', 'compress_factor': 128, 'compress_method': 'svd'},
		# {'encoder': 'google_universal_sentence_encoder', 'compress_factor': 32, 'compress_method': 'svd'},
		# {'encoder': 'google_universal_sentence_encoder', 'distance_function': 'euclidean'},
		# {'encoder': 'google_universal_sentence_encoder', 'distance_function': 'euclidean',
		# 	'compress_factor': 128},
		# {'encoder': 'google_universal_sentence_encoder', 'distance_function': 'euclidean',
		# 	'compress_factor': 32},
		# {'encoder': 'tfidf', 'model': '1_3_min_df_2'},	# 77, 117 all zero
		# {'encoder': 'tfidf', 'model': '1_3_min_df_2', 'distance_function': 'euclidean'},	# 77, 117 all zero
		# {'encoder': 'jaccard_similarity', 'model': 'nl_nr', 'distance_function': 'jaccard_similarity'}
	]
	grading_actions = 40
	iterations = None
	batch_size = 5
	possible_grades = ['Correct', 'Wrong']

	# args = {'batch_size': batch_size, 'representation_version': '*', 'information_version': 'max',
	# 	'pgv_version': '+', 'voting_version': 'average'}
	# algorithm, labels, text_list, reduced_labels, reduced_text_list, result, gp_dict \
	# 	= evaluate(name, question_id, mapping, subspace_param_list, grading_actions, iterations, \
	# 	possible_grades, **args)
	# print_neighbours(name, question_id, labels, text_list, reduced_labels, reduced_text_list, algorithm)
	# sys.exit()
	
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
	# explicit_list = None
	
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
	# print_guse_distance_test_result(name, question_id, mapping=mapping,
	# 	selected_indices=[577, 196], method='angular')
	# sys.exit()

	grading_actions_list = list(range(20, 110, 10))
	
	# separation1_version_list = ['min', 'max', '*', '+']
	# information_version_list = ['max', 'min', '*', '+']
	# pgv_version_list = ['*', '+']
	# voting_version_list = ['average']
	# args = {'separation1_version': separation1_version_list,
	# 	'information_version': information_version_list, 'pgv_version': pgv_version_list,
	# 	'voting_version': voting_version_list}
	# explicit_param_list = generate_param_conbinations(args)

	representation_version_list = ['+', '*']
	information_version_list = ['max', 'min', '*', '+']
	pgv_version_list = ['+', '*']
	voting_version_list = ['average']
	args = {'representation_version': representation_version_list,
		'information_version': information_version_list, 'pgv_version': pgv_version_list,
		'voting_version': voting_version_list}
	explicit_param_list = generate_param_conbinations(args)

	# separation1_version_list = ['min', 'min', 'min', 'min', 'min', 'min', 'min', 'max', 'max', 'max', 'max', '*', '*',	'*', '*', '*', '+', '+', '+']
	# information_version_list = ['max', 'min', 'max', '*', 'min', '+', '+', 'max', 'max', '+', '+', 'max', 'max', 	'*', '+', '+', 'min', '*', '+']
	# pgv_version_list = ['*', '+', 	'+',  '+', '*', '*', '+', '*', '+', '*', '+', '*', '+', '*', '+', '+', '+', '+', '+']
	# voting_version = ['average'] * len(separation1_version_list)
	# plain_list = list(zip(separation1_version_list, information_version_list, pgv_version_list,
	# 	voting_version))
	# explicit_param_list = [{'separation1_version': p[0], 'information_version': p[1],
	# 	'pgv_version': p[2], 'voting_version': p[3]} for p in plain_list]
	
	result_dict, labels, text_list, reduced_labels, reduced_text_list \
		= evaluate_batch(name, question_id, mapping, subspace_param_list,
			grading_actions_list=grading_actions_list, batch_size=batch_size, possoble_labels=possoble_labels,
			explicit_param_list=explicit_param_list)
	print_accuracy_batch(name, question_id, result_dict, labels, text_list, reduced_labels,
		reduced_text_list, explicit_param_list=explicit_param_list, save_mode=2)
