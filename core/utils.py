import os, csv, pickle, math, random, csv
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import scipy.spatial.distance as sd

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class InvalidNameError(Exception):
	def __init__(self, value_type, name):
		self.type = value_type
		self.name = name

	def __str__(self):
		return 'INVALID {} NAME: {}'.format(self.type.upper(), self.name)

def create_directories(path):
	# parts = os.path.split(path)
	# parts = path.split('/')
	path = os.path.normpath(path)
	parts = path.split(os.sep)

	# p = None if not path.startswith('/') else Path('/')
	if os.sep == '\\':
		p = Path(parts.pop(0) + os.sep)
	else:
		p = None if not path.startswith('/') else Path('/')
	for part in parts:
		p = Path(part) if p is None else p / part
		if not os.path.exists(p):
			# temp fix for symlink on windows
			try:
				os.mkdir(p)
			except Exception as e:
				pass
	return p

def read_file(path):
	lines = []
	path = Path(path) if not isinstance(path, Path) else path
	with open(path, 'r', encoding='utf8', errors='ignore') as file:
		lines = file.readlines()
	
	return [line.strip() for line in lines]

def read_csv_file(path, delimiter=','):
	rows = []
	with open(path, 'r', encoding='utf8', errors='ignore') as file:
		for row in csv.reader(file, delimiter=delimiter):
			rows.append(row)
	return rows

def write_csv_file(path, file_name, rows, delimiter=','):
	path = create_directories(path)

	with open(path / file_name, 'w', newline='\n') as csvfile:
		writer = csv.writer(csvfile, delimiter=delimiter)
		for row in rows:
			writer.writerow(row)

def write(path, file_name, lines):
	path = create_directories(path)

	with open(path / file_name, 'w') as file:
		for line in lines:
			file.write('{}\n'.format(line))

def save(data, output_path, file_name):
	path = create_directories(output_path)

	with open(path / file_name, 'wb') as file:
		# pickle.dump(data, file, fix_imports=True)
		pickle.dump(data, file)

def save_zip(data, output_path, file_name, compression=ZIP_DEFLATED, allowZip64=True, compresslevel=9):
	_ = save(data, output_path, file_name)
	path = Path(output_path)
	input_file_path = path / file_name
	output_file_name = '.'.join(file_name.split('.')[:-1]) + '.zip'
	with ZipFile(path / output_file_name, compression=compression, allowZip64=allowZip64,
		compresslevel=compresslevel, mode='w') as z:
		z.write(input_file_path, arcname=file_name, compresslevel=compresslevel)
	os.remove(input_file_path)

def load(path):
	with open(Path(path), 'rb') as file:
		# return pickle.load(file, fix_imports=True)
		return pickle.load(file)

def load_zip(path, file_name):
	with ZipFile(path, mode='r') as z:
		data = z.read(file_name)
	return pickle.loads(data)

def is_empty_string(text):
	return text.strip() == ''

def validate_option(options, value, name):
	if not value in options:
		raise InvalidNameError(name, value)

# def compress_data(data, factor, data_path, is_reference_answers=False, save_to_file=False, method='pca'):
# 	try:
# 		file_path = '{}/compressed/{}_{}'.format(data_path, method, factor)
# 		# file_name = 'data.dat'
# 		file_name = 'data.npy' if not is_reference_answers else 'reference_answer_data.npy'

# 		data_dim, num_data = len(data[0]), len(data)

# 		from sklearn import decomposition
# 		factor = int(data_dim * factor) if factor < 1. else factor
# 		# pca = decomposition.PCA(n_components=min(int(data_dim * factor), num_data))
# 		# pca = decomposition.PCA(n_components=min(factor, num_data))
# 		if method == 'pca':
# 			pca = decomposition.PCA(n_components=min(factor, num_data), svd_solver='full')
# 			data = pca.fit_transform(np.array(data))
# 		elif method == 'svd':
# 			svd = decomposition.TruncatedSVD(n_components=min(factor, num_data - 1), algorithm='arpack',
# 				random_state=0)
# 			data = svd.fit_transform(np.array(data))
# 		# svd = decomposition.TruncatedSVD(n_components=min(int(self.data_dim * self.pca), self.num_data))
# 		# self.data = svd.fit_transform(np.array(self.data))
# 		if save_to_file:
# 			# save(data, file_path, file_name)
# 			create_directories(str(file_path))
# 			np.save(Path(file_path) / file_name, data, allow_pickle=False)

# 		return data
# 	except Exception as e:
# 		print('n_components: {}'.format(min(factor, num_data)))
# 		print(e)

# 		import sys
# 		sys.exit(1)

def compress_data(data, reference_answer_data, factor, data_path, save_to_file=False, method='pca'):
	try:
		file_path = '{}/compressed/{}_{}'.format(data_path, method, factor)
		# file_name = 'data.dat'
		# file_name = 'data.npy' if not is_reference_answers else 'reference_answer_data.npy'
		file_name = 'data.npy'
		reference_answer_file_name = 'reference_answer_data.npy'

		all_data = np.append(data, reference_answer_data, axis=0)
		data_dim, num_data = len(all_data[0]), len(all_data)

		from sklearn import decomposition
		factor = int(data_dim * factor) if factor < 1. else factor
		# pca = decomposition.PCA(n_components=min(int(data_dim * factor), num_data))
		# pca = decomposition.PCA(n_components=min(factor, num_data))
		if method == 'pca':
			pca = decomposition.PCA(n_components=min(factor, num_data), svd_solver='full')
			all_data = pca.fit_transform(np.array(all_data))
		elif method == 'svd':
			svd = decomposition.TruncatedSVD(n_components=min(factor, num_data - 1), algorithm='arpack',
				random_state=0)
			all_data = svd.fit_transform(np.array(all_data))
		# svd = decomposition.TruncatedSVD(n_components=min(int(self.data_dim * self.pca), self.num_data))
		# self.data = svd.fit_transform(np.array(self.data))
		if save_to_file:
			# save(data, file_path, file_name)
			create_directories(str(file_path))
			np.save(Path(file_path) / file_name, all_data[:len(data)], allow_pickle=False)
			if reference_answer_data is not None:
				np.save(Path(file_path) / file_name, all_data[len(data):], allow_pickle=False)
		return all_data[:len(data)], all_data[len(data):]
	except Exception as e:
		print('n_components: {}'.format(min(factor, num_data)))
		print(e)

		import sys
		sys.exit(1)

def dist(x, y, method):
	# if self.dist_method == 'euclidean':
	if method == 'euclidean':
		# return self.dist_euclidean(x, y)
		return math.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
	elif method == 'cosine':
		# return self.dist_cosine(x, y, data_type)
		return sd.cdist(x, y, 'cosine')
	elif method == 'angular':
		return 2 * np.arccos(1 - dist(x, y, 'cosine')) / math.pi
	elif method == 'jaccard_similarity':
		set_x, set_y = set(x), set(y)
		intersection = set_x.intersection(set_y)
		union = set_x.union(set_y)
		distance = float(len(intersection) / len(union))
		return 1 - distance # to make the value consistence with other methods
	elif method == 'manhattan':
		# return sd.cityblock(x, y)
		return sd.cdist(x, y, 'cityblock')
	else:
		raise InvalidNameError('Distance Method', method)

def get_data_distances(data, method):
	data_len = len(data)
	if method == 'euclidean' or method == 'jaccard_similarity':
		# print('DATA DISTANCE: {}'.format(data_distances))
		data_distances = np.zeros(shape=(data_len, data_len))
		# data_distances[:] = np.nan
		for i in range(data_len):
			for j in range(i + 1, data_len):
				distance = dist(data[i], data[j], method)
				data_distances[i][j] = distance
				data_distances[j][i] = distance
	else:
		data_distances = dist(data, data, method)
		data_distances = data_distances.clip(min=0)  # added Andrew
		np.fill_diagonal(data_distances, 0.)
	return data_distances

# def calculate_data_distances(data, distance_function):
# 	data_len = len(data)
# 	data_distances = get_data_distances(data=data, method=distance_function)
# 	mean_local_distances = np.zeros(shape=data_len)
# 	max_local_dist = None
# 	for i in range(data_len):
# 		distances = np.sort(data_distances[i])
# 		coefficient = config.DENSITY_COEFFICIENT if data_len > config.DENSITY_COEFFICIENT else (data_len - 1)
# 		mean_local_distances[i] = np.sum(distances[:coefficient]) / coefficient  # Andrew: densities actually inversed density 0 = most dense
# 		if max_local_dist is None or max_local_dist < mean_local_distances[i]:
# 			max_local_dist = mean_local_distances[i]
# 	outlier_factor = mean_local_distances / max_local_dist

# 	outlier_adjusted_distances = np.zeros(shape=(data_len, data_len))
# 	for i in range(data_len):
# 		for j in range(i + 1, data_len):
# 			adjusted_distance = (1 + math.sqrt(outlier_factor[i] * outlier_factor[j])) * data_distances[i][j]
# 			outlier_adjusted_distances[i][j] = adjusted_distance
# 			outlier_adjusted_distances[j][i] = adjusted_distance
   
# 	return data_distances, outlier_factor

# Andrew
def calculate_entropy(data_len, solutions, valid_from=1):
	#data_len = len(data)
	label_list = [solution.labels for solution in solutions] # Andrew
	entropy = [0.] * data_len
	counts = np.zeros(shape=(data_len, data_len))
	for labels in label_list:
		for i in range(data_len):
			for j in range(i + 1, data_len):
				if labels[i] == labels[j] and labels[i] >= valid_from:
					counts[i][j] += 1
					counts[j][i] += 1

	probabilities = counts / len(label_list)
	for i in range(data_len):
		for j in range(data_len):
			entropy[i] += -1 * probabilities[i][j] * (math.log2(probabilities[i][j]) if probabilities[i][j] > 0. else 0.)
	return entropy

def get_tsne_data(data, mode='2d', perplexity=5.):
	if mode == '3d':
		tsne_data = TSNE(n_components=3, perplexity=perplexity).fit_transform(data)
	else:
		tsne_data = TSNE(n_components=2, perplexity=perplexity, early_exaggeration = 1).fit_transform(data)
	return tsne_data

def plot(data, labels, title, core_data_indices=[]):
	unique_labels = sorted(list(set(labels)))
	# colors = [plt.cm.Spectral(i) for i in np.linspace(0, 1, len(unique_labels))]
	
	if len(unique_labels) <= 3:
		if len(data[0]) <= 2:
			colors = [
				(0.3686274509803922, 0.30980392156862746, 0.6352941176470588, 1.0),
				(0.998077662437524, 0.9992310649750096, 0.7460207612456747, 1.0),
				(0.6352941176470588, 0.30980392156862746, 0.3686274509803922, 1.0)
			]
		else:
			colors = [(0.2, 0.2, 0.8, 1.0), (0.8, 0.2, 0.2, 1.0), (0.2, 0.8, 0.2, 1.0)]
	else:
		colors = [plt.cm.Spectral(i) for i in np.linspace(0, 1, len(unique_labels))]

	core_data_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]

	figure = plt.figure()
	ax = figure.add_subplot(111, projection='3d') if len(data[0]) > 2 else None
	# ax = Axes3D(figure)

	core_data_markers = ['^', 'v']
	core_data_marker_index = 0

	for i in range(len(unique_labels) + len(core_data_indices)):
		if i < len(unique_labels):
			color = colors[i]
			xy = data[labels == unique_labels[i]]
			marker = 'o'
			size = 6
			label = unique_labels[i]
			print('label: {}, color: {}'.format(list(unique_labels)[i], color))
		else:
			color = core_data_colors[(i - len(unique_labels)) % len(core_data_colors)]
			print('index: {}, color: {}'.format((i - len(unique_labels) % len(core_data_colors)), color))
			xy = data[core_data_indices[i - len(unique_labels)]]
			# marker = '^'
			marker = core_data_markers[core_data_marker_index]
			core_data_marker_index += 1
			size = 6
			label = ''

		if len(data[0]) == 2:
			plt.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=tuple(color), markeredgecolor='k', 
				markersize=size, label=label)
			# ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=6)
		else:
			# ax = Axes3D(figure)
			# ax.plot(
			# 	xy[:, 0], xy[:, 1], zs=xy[:, 2], marker=marker, markerfacecolor=tuple(color),
			# 	markeredgecolor='k', markersize=5
			# )
			ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], c=tuple(color), marker='o', label=label)

	min_x, max_x = -10, 10
	min_y, max_y = -10, 10
	min_z, max_z = -10, 10

	plt.grid(True)
	plt.title(title)
	plt.legend()

def plot_multiple_lines(title, x, y_list, x_label=None, y_label=None, x_ticks=None, y_ticks=None,
	legend=[]):
	import pandas

	data = {legend[0] if len(legend) > 0 else 'x': x}
	for i in range(len(y_list)):
		data[legend[i+1] if i+1 < len(legend) else 'y{}'.format(i+1)] = y_list[i]
	
	data = pandas.DataFrame(data)
	colors = [plt.cm.Spectral(i) for i in np.linspace(0, 1, len(y_list))]

	figure = plt.figure()

	for i in range(len(y_list)):
		color = tuple(colors[i])
		plt.plot(
			legend[0] if len(legend) > 0 else 'x',
			legend[i+1] if i+1 < len(legend) else 'y{}'.format(i+1),
			data=data, marker='o', markerfacecolor=color, markeredgecolor='k', color=color, linewidth=4,
			alpha=0.7
		)

	plt.grid(True)
	plt.title(title)
	if x_label:
		plt.xlabel(x_label)
	if y_label:
		plt.ylabel(y_label)
	if x_ticks:
		plt.xticks(x_ticks)
	if y_ticks:
		plt.yticks(y_ticks)
	plt.legend()

def plot_scatter(points_lists, x_label, y_label, legends, title, x_ticks=None, y_ticks=None, size=20):
	# if len(points_lists) <= 3:
	# 	colors = [
	# 		(1., 0.3, 0.3, 1.0),
	# 		(0.3, 1., 0.3, 1.0),
	# 		(0.3, 0.3, 1., 1.0)
	# 	]
	# else:
	# 	colors = [plt.cm.Spectral(i) for i in np.linspace(0, 1, len(points_lists))]
	colors = [plt.cm.Spectral(i) for i in np.linspace(0, 1, len(points_lists))]

	figure, ax = plt.subplots()
	# ax = Axes3D(figure)

	for i in range(len(points_lists)):
		color = colors[i]
		xy = np.array(points_lists[i])
		marker = 'o'
		print('label: {}, color: {}'.format(legends[i], color))

		# plt.plot(
		# 	xy[:, 0], xy[:, 1], marker, markerfacecolor=tuple(color), markeredgecolor='k', 
		# 	markersize=size
		# )
		ax.scatter(xy[:, 0], xy[:, 1], color=color, s=size, label=legends[i], alpha=1.0, edgecolors='none')
		# ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(color), markeredgecolor='k', markersize=6)

	ax.legend()
	ax.grid(True)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	# plt.xticks([0, 2.])
	# plt.yticks([0, 2.])
	if x_ticks is not None:
		plt.xticks(x_ticks)
	if y_ticks is not None:
		plt.y_ticks(y_ticks)
	plt.title(title)
	plt.show()

def plot_histogram(title, x, bins, labels=['Y', 'X'], log_scale=True, x_ticks=None, y_ticks=None):
	plt.figure()

	for i in range(len(x)):
		plt.hist(x[i], density=False, bins=bins, alpha=0.7, label=labels[i+1] if i+1 < len(labels) else 'X{}'.format(x+1))

	# plt.xlabel(labels[0])
	# plt.ylabel(labels[1])
	plt.ylabel(labels[0])

	# axes = plt.axes()
	# axes.set_ylim([0, 1])

	if log_scale:
		plt.yscale('log', nonposy='clip')

	plt.grid(True)
	plt.title(title)
	plt.legend()

	if x_ticks:
		plt.xticks(x_ticks)
	if y_ticks:
		plt.yticks(y_ticks)

def bar(title, x, height, width=0.8, bottom=0, align='center'):
	plt.figure()
	plt.bar(x=x, height=height, width=width, bottom=bottom, align=align)

	plt.grid(True)
	plt.title(title)
	plt.legend()
