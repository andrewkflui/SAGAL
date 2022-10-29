import os, sys, argparse
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..{}..{}..{}'.format(os.sep, os.sep, os.sep)))

from pathlib import Path
import xlrd, openpyxl, xlsxwriter

if os.name == 'nt':
	import win32api, win32con

def is_hidden_file(path, file_name):
	if os.name == 'nt':
		attribute = win32api.GetFileAttributes(os.path.join(path, file_name))
		return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
	else:
		return file_name.startswith('.') or file_name.startswith('~')

def consolidate_results_single_version(path, reorganize_folders=None, apply_file_count=10):
	file_names = []
	for file_name in os.listdir(path):
		if is_hidden_file(path, file_name) or file_name.startswith('consolidated'):
			continue
		file_names.append(file_name)
	file_names = sorted(file_names)

	if reorganize_folders is not None and len(file_names) > apply_file_count \
		and len(file_names) % apply_file_count == 0:
		file_path_list = [[] for _ in range(len(reorganize_folders))]
		for i in range(len(file_names)):
			file_name = file_names[i]
			index = i % len(reorganize_folders)
			folder_name = reorganize_folders[index]
			if not os.path.exists(os.path.join(path, folder_name)):
				os.mkdir(os.path.join(path, folder_name))
			file_path = os.path.join(path, folder_name, file_name)
			os.rename(os.path.join(path, file_name), file_path)
			file_path_list[index].append(file_path)
	else:
		file_path_list = [[os.path.join(path, file_name) for file_name in file_names]]

	for file_paths in file_path_list:
		descriptions, results = None, None
		for file_path in file_paths:
			v, s, d, r = process_excel_file_single_version(file_path)
			if descriptions is None:
				descriptions = d
			if results is None:
				results = [['Version', v]]
			results += r + [['']]
		
		file_name = file_names[0][:file_names[0].index('_seed')]
		workbook = xlsxwriter.Workbook(os.path.join(os.sep.join(file_path.split(os.sep)[:-1]),
			'consolidated_{}.xlsx'.format(file_name)))
		sheet = workbook.add_worksheet('Sheet1')

		values = descriptions + [['']] + results
		for i in range(len(values)):
			for j in range(len(values[i])):
				if j == 0:
					sheet.set_column(j, j, 30)
				sheet.write(i, j, values[i][j])
		workbook.close()

def process_excel_file_single_version(file_path):
	version, random_seed = None, None
	descriptions, results = [], []
	result_started = False

	workbook = openpyxl.load_workbook(file_path)
	worksheet = workbook.active
	for row in worksheet.values:
		header = row[0].lower()
		if header == 'random seed':
			random_seed = row[1]
		elif header.startswith('version') and random_seed is not None:
			if version is None:
				version = 'V{}'.format(int(row[1]))
			else:
				version += '_' + row[1]
		elif header == 'subspaces':
			result_started = True
			results.append(['Random Seed', random_seed])

		if not result_started:
			descriptions.append(list(row))
		else:
			results.append(list(row))

	# with xlrd.open_workbook(os.path.join(path, file_name)) as workbook:
	# 	sheet = workbook.sheets()[0]
	# 	for i in range(sheet.nrows):
	# 		header = sheet.cell_value(i, 0).lower()
	# 		if header == 'random seed':
	# 			random_seed = sheet.cell_value(i, 1)
	# 		elif header.startswith('version') and random_seed is not None:
	# 			if version is None:
	# 				version = 'V{}'.format(int(sheet.cell_value(i, 1)))
	# 			else:
	# 				version += '_' + sheet.cell_value(i, 1)
	# 		elif header == 'subspaces':
	# 			result_started = True
	# 			results.append(['Random Seed', random_seed])

	# 		row_values = [sheet.cell_value(i, j) for j in range(sheet.ncols)]
	# 		if not result_started:
	# 			descriptions.append(row_values)
	# 		else:
	# 			results.append(row_values)

	return version, random_seed, descriptions, results

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', help='path containing excel files', default='.')

	args = parser.parse_args()
	path = args.path

	folders = ['glove_64Vx16', 'bert_64Vx16', 'guse_64Vx12_glove_64Vx4', 'guse_64Vx12_bert_64Vx4',
		'guse_glove_0.5_0.5_128Vx16', 'guse_bert_0.5_0.5_128Vx16']
	consolidate_results_single_version(path, reorganize_folders=folders)