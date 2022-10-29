import os, sys, math, statistics
from pathlib import Path
from enum import Enum

import xlrd, docx2txt
from xml.etree import ElementTree

import numpy as np
import scipy.spatial.distance as sd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from core import utils
from basis import config

LABELS_2WAY = ['Correct', 'Wrong']
LABELS_3WAY = {'0': 'Wrong', '1': 'Partial', '2': 'Correct'}
LABELS_5WAY = {'0': 'Wrong', '1': 'Partially Wrong', '3': 'Marginal', '4': 'Partially Correct', '5': 'Correct'}

class ScoreEncoding(Enum):
	TEXT = 0
	NUMERIC = 1
	UNKNOWN = 2

class Answer(object):
	def __init__(self, answer_id, text, scores, is_reference=False, metadata=dict()):
		self.id = answer_id
		self.text = text
		self.scores = scores
		self.is_reference = is_reference
		self.metadata = metadata
		# self.encodings = dict()
		self.encoding = None
		self.words = None

	def __str__(self):
		string = '[REFERENCE ANSWER]\n' if self.is_reference else '[ANSWER]\n'
		string += 'ID: {}\n'.format(self.id)
		string += 'SCORES: {}\n'.format(self.scores)
		string += '{}{}\n'.format(self.text[:100] if self.text else None, '...' if self.text and len(self.text) >= 100 else '')
		return string

	def get_dominated_score(self):
		return max(self.scores, key=self.scores.count)

	def get_answer_class(self, mapping, scale, dataset, question):
		score_classes = dataset.metadata.get('score_classes')
		if score_classes is not None:
			score = self.get_dominated_score()
			# score = 'non_domain' if dataset.name == 'SEB3' and score == 'incorrect' else score
			if dataset.name == 'SEB2' or mapping == '2way':
				threshold = (len(score_classes) - 1) * scale
				return LABELS_2WAY[0] if score_classes.index(score) >= threshold else LABELS_2WAY[1]
			elif mapping == '3way' and dataset.name.startswith('SEB'):
				if dataset.name == 'SEB3':
					return score_classes.index(score)
				else:
					position = score_classes.index(score)
					return 2 if position >= 4 else (1 if position >= 3 else 0)
			else:
				# return score
				return score_classes.index(score)
		else:
			max_value = question.metadata.get('max_value')
			score = sum(answer.scores) / len(answer.scores)
			if mapping == '2way':
				threshold = max_value * scale
				return LABELS_2WAY[0] if score >= threshold else LABELS_2WAY[1]
			else:
				return score

class Question(object):
	def __init__(self, question_id, domain, text, answers=None, metadata=None):
		self.id = question_id
		self.domain = domain
		self.text = text
		self.answers = answers if answers else list()
		self.metadata = metadata if metadata else dict()

	def __str__(self):
		string = '[QUESTION]\n'
		string += 'DOMAIN: {}\n'.format(self.domain)
		string += 'ID: {}\n'.format(self.id)
		for key, value in self.metadata.items():
			if key == 'description':
				string += '{}: {}{}\n'.format(key, value[:100], '...' if self.text and len(self.text) >= 100 else '')
			else:
				string += '{}: {}\n'.format(key, value)
		# string += 'METADATA: {}\n'.format(self.metadata)
		string += '{}\n'.format(self.text)
		return string

	# print_type: 1 for reference only, 2 for non-reference only, otherwise for all
	def print_answers(self, print_type=3):
		string = ''

		if print_type == 1:
			answers = [a for a in self.answers if a.is_reference]
		elif print_type == 2:
			answers = [a for a in self.answers if not a.is_reference]
		else:
			answers = sorted(self.answers, key=lambda x: x.is_reference, reverse=True)
		
		for answer in answers:
			string += '\n{}'.format(answer)
		
		print(string)

class Dataset(object):
	def __init__(self, name, score_encoding, extra_data, path):
		self.name = name
		self.metadata = extra_data if extra_data is not None else dict()
		if not 'has_reference_answer' in self.metadata:
			self.metadata['has_reference_answer'] = False
		self.metadata['score_encoding'] = score_encoding
		self.questions = []

		self.process(path)

	def __str__(self):
		string = '[DATASET]\n'
		string += 'NAME: {}\n'.format(self.name)
		string += 'HAS_REFERENCE_ANSWER: {}\n'.format(self.metadata.get('has_reference_answer'))
		string += 'SCORE CLASSES: {}\n'.format(self.metadata.get('score_classes'))
		string += 'SCORE ENCODING: {}\n'.format(self.metadata.get('score_encoding'))
		string += 'QUESTION COUNT: {}'.format(len(self.questions))
		return string

	def process(self, path):
		raise NotImplementedError('write function not implemented')

	def get_question(self, question_id):
		return next((q for q in self.questions if q.id == question_id), None)

class SciEntsBank2013Task7(Dataset):
	def __init__(self, subset_name, container_folder_name, score_classes, root_path):
		self.subset_name = subset_name
		self.container_folder_name = container_folder_name
		self.fixes_path = Path(config.ROOT_PATH) / 'data/datasets/fixes/SEB{}'.format(subset_name)
		self.grade_corrections_path = self.fixes_path / 'grade_corrections.csv'
		super().__init__(name='SEB{}'.format(subset_name), score_encoding=ScoreEncoding.TEXT, extra_data={'has_reference_answer': True, 'score_classes': score_classes}, path=root_path)

	def read_grade_corrections(self):
		if os.path.exists(self.grade_corrections_path):
			rows = utils.read_csv_file(self.grade_corrections_path, delimiter=',')
			return rows[0], np.array(rows[1:])
		return [], []

	def scan(self, path, directory=None, files=[]):
		directories = [x for x in os.listdir(path) if os.path.isdir(Path('{}/{}'.format(path, x)))]

		if len(directories) > 0 and not self.container_folder_name in directories:
			for d in directories:
				self.scan(Path('{}/{}'.format(path, d)), d, files)
		else:
			for r, d, f in os.walk(Path('{}/{}'.format(path, self.container_folder_name))):
				for file in [xml for xml in f if xml.endswith('.xml')]:
					parts = r.split('/')
					files.append((directory, Path('{}/{}'.format(r, file))))

		return files

	def process(self, path):
		corrections = self.read_grade_corrections()
		files = self.scan(path)

		self.questions = []
		for file in files:
			usage, file_path = file

			question_element = ElementTree.parse(file_path).getroot()
			question_text = question_element.find('questionText').text
			reference_answer_elements = question_element.find('referenceAnswers').findall('referenceAnswer')
			student_answer_elements = question_element.find('studentAnswers').findall('studentAnswer')

			# attributes
			question_id = None
			question_metadata = {'usage': usage}

			for key, value in question_element.attrib.items():
				if key == 'id':
					question_id = value
				elif key == 'module':
					domain = value
				else:
					question_metadata[key] = value

			answers = []
			
			# reference answers
			for answer in reference_answer_elements:
				answer_id = None
				answer_text = answer.text

				answer_metadata = dict()
				for key, value in answer.attrib.items():
					if key == 'id':
						answer_id = value
					else:
						answer_metadata[key] = value

				answers.append(Answer(answer_id=answer_id, text=answer_text, scores=['correct'], is_reference=True))

			# student answer
			for answer in student_answer_elements:
				answer_id = None
				answer_text = answer.text
				scores = []

				answer_metadata = dict()
				for key, value in answer.attrib.items():
					if key == 'id':
						answer_id = value
					elif key == 'accuracy':
						scores.append(value)
					else:
						answer_metadata[key] = value

				# read fixes and corrrect
				fix = next((f for f in corrections[1] if self.name == f[0] and answer_id == f[2] \
					and answer_text == f[3]), None)
				if fix is not None:
					scores = [fix[-1]]

				answers.append(Answer(answer_id=answer_id, text=answer_text, scores=scores))

			question = self.get_question(question_id)
			if question is None:
				self.questions.append(Question(question_id=question_id, domain=domain, text=question_text, answers=answers))
			else:
				question.answers += answers

class USCIS(Dataset):
	def __init__(self, root_path, include_100=False):
		self.file_names = ['questions_answer_key', 'studentanswers_grades_698']
		if include_100:
			self.file_names.append('studentanswers_grades_100')
		self.fixes_path = Path(config.ROOT_PATH) / 'data/datasets/fixes/USCIS'
		self.grade_corrections_path = self.fixes_path / 'grade_corrections.csv'
		super().__init__(name='USCIS{}'.format('_include_100' if include_100 else ''), score_encoding=ScoreEncoding.NUMERIC, extra_data={'has_reference_answer': True, 'score_classes': [-1, 0, 1]}, path=root_path)

	def parse_questions(self, rows):
		self.questions = []
		domain = 'USCIS'

		i = 0
		for row in rows:
			if i == 0:
				i += 1
				continue
			
			question_id = row[0]
			question_text = row[1]
			answer_text = row[2:]

			question = Question(question_id=question_id, domain=domain, text=question_text)

			for j in range(0, len(answer_text)):
				question.answers.append(Answer(answer_id='{}_ref_{}'.format(question.id, j+1), text=answer_text[j].lower(), scores=[1, 1, 1], is_reference=True))

			self.questions.append(question)

	def parse_student_answers(self, rows, corrections):
		i = 0
		for row in rows:
			if i > 0:
				question = next((q for q in self.questions if q.id == row[1]), None)
				
				# read fixes and corrrect
				fix = next((f for f in corrections[1] if question.id == f[1] \
					and row[0] == f[2]), None)
				if fix is not None:
					scores = list(map(int, fix[-1].replace('[', '').replace(']', '').split(',')))
				else:
					scores = [int(score) for score in row[3:]]
				
				answer_id = '{}_student_{}'.format(question.id, len(question.answers) + i)
				answer_text = row[2]

				if utils.is_empty_string(answer_text):
					print('Empty answer! question ID: {}, line: {}, answer_id: {}'.format(question.id, i, answer_id))
					continue

				# question.answers.append(Answer(answer_id=answer_id, text=answer_text.lower(), scores=[int(score) for score in row[3:]], is_reference=False, metadata={'student': row[0]}))
				question.answers.append(Answer(answer_id=answer_id, text=answer_text.lower(), scores=scores, is_reference=False, metadata={'student': row[0]}))
			i += 1

	# def read_grade_corrections(self):
	# 	rows = utils.read_csv_file(self.grade_corrections_path, delimiter=',')
	# 	return rows[0], np.array(rows[1:])

	def process(self, path):
		corrections = self.read_grade_corrections()

		for name in self.file_names:
			rows = utils.read_csv_file(Path('{}/{}.tsv'.format(path, name)), delimiter='\t')
			if name == 'questions_answer_key':
				self.parse_questions(rows)
			else:
				self.parse_student_answers(rows, corrections)

class Mobley(Dataset):
	def __init__(self, root_path):
		super().__init__(name='mobley', score_encoding=ScoreEncoding.NUMERIC, extra_data={'has_reference_answer': True}, path=root_path)

	def parse_line(self, line):
		parts = line.split(' ')
		return parts[0], ' '.join(parts[1:])

	def parse_questions(self, path):
		file_names = ['questions', 'answers']
		
		domain = 'Data Structure'
		self.questions = []

		for name in file_names:
			lines = utils.read_file(Path('{}/data/raw/{}'.format(path, name)))

			for line in lines:
				question_id, text = self.parse_line(line)

				if name == 'questions':
					self.questions.append(Question(question_id, domain=domain, text=text, metadata={'min_value': 0., 'max_value': 10.0}))
				else:
					question = self.get_question(question_id)
					if question is not None:
						question.answers.append(Answer(answer_id='{}_{}'.format(question.id, len(question.answers)+1), text=text.lower(), scores=[10.0, 10.0], is_reference=True))

	def parse_answers(self, path):
		lines = utils.read_file(Path('{}/data/raw/all'.format(path)))
		
		previous_question_id, question = None, None
		answers = []
		i = 0

		while True:
			line = lines[i] if i < len(lines) else None

			if line is not None:
				parts = line.split(' ')
				answer_text = ' '.join(parts[1:])
			
			if not parts[0] == previous_question_id or i >= len(lines):
				if previous_question_id is not None:
					self.parse_scores(question, answers, path)
					answers = []
				previous_question_id = parts[0]
			
			question = next((q for q in self.questions if q.id == parts[0]), None)
			if question is not None:
				answers.append(Answer(answer_id='{}_{}'.format(question.id, len(answers)+1), text=answer_text.lower(), scores=[]))

			if i >= len(lines):
				break

			i += 1

	def parse_scores(self, question, answers, path):
		if question is None:
			return

		for name in ['me', 'other']:
			lines = utils.read_file(Path('{}/data/scores/{}/{}'.format(path, question.id, name)))
			for i in range(len(lines)):
				answers[i].scores.append(float(lines[i]))

		for answer in answers:
			question.answers.append(answer)

	def process(self, path):
		self.parse_questions(path)
		self.parse_answers(path)

class ChakrabortyAndKonar(Dataset):
	def __init__(self, root_path):
		super().__init__(name='cak', score_encoding=ScoreEncoding.NUMERIC, extra_data=None, path=root_path)

	def parse_sheet(self, sheet):
		question = Question(question_id=sheet.name, domain='Data Structure in UG Year 1', text=sheet.cell_value(1, 2), metadata={'min_value': 0., 'max_value': 1.})

		for i in range(4, sheet.nrows):
			raw_answer = []
			
			# index = 0
			for j in range(1, sheet.ncols):
				value = sheet.cell_value(i, j)
				if j == 1:
					value = int(value)
				raw_answer.append(value)

			question.answers.append(Answer(answer_id='{}_{}'.format(question.id, len(question.answers)), text=raw_answer[1], scores=[raw_answer[2]], metadata={'Sl.No': raw_answer[0]}))

		self.questions.append(question)

	def process(self, path):
		workbook = xlrd.open_workbook(path / 'Data_SingleSentence.xlsx')
		for sheet in workbook.sheets():
			self.parse_sheet(sheet)

class ASAP(Dataset):
	def __init__(self, root_path):
		super().__init__('ASAP', ScoreEncoding.NUMERIC, extra_data=None, path=root_path)

	def parse_essay_detail_description(self, question, path):
		if question is None:
			return

		text = docx2txt.process(Path('{}/Essay_Set_Descriptions/Essay Set #{}--ReadMeFirst.docx'.format(path, question.id)))
		lines = [line.strip() for line in text.split('\n') if not line.strip() == '']

		i = 0
		start = False

		while i < len(lines):
			line = lines[i]
			if not start and (line == 'Prompt' or line == 'Source Essay'):
				start = True
				question.metadata['description'] = '{}\n'.format(line)
			elif start and not line.startswith('Essay Set #') and not line.isdigit():
				question.metadata['description'] += '{}\n'.format(line)
			i += 1

	def parse_essay_descriptions(self, path):
		workbook = xlrd.open_workbook('{}/Essay_Set_Descriptions/essay_set_descriptions.xlsx'.format(path))
		sheet = workbook.sheet_by_index(0)
		headers = []

		for i in range(sheet.nrows):
			question = None
			
			for j in range(sheet.ncols):
				value = sheet.cell_value(i, j)
				if i == 0:
					headers.append(value)
				elif j == 0:
					question = Question(question_id=int(value), domain='essay', text=None)
				else:
					question.metadata[headers[j]] = value

			if question is not None:
				self.parse_essay_detail_description(question, path)
				self.questions.append(question)

	# ======================
	# Note:
	# score calculation for essay set 1, 3 - 6:
	# domain1_score
	#
	# score calculation for essay set 2
	# (domain1_score + domain2_score) / 2
	#
	# score calculation for essay set 7 - 8:
	# traits are I, O, V, W, S, C
	# formula w/o 3rd rater: (I_R1+I_R2)  +  (O_R1+O_R2)  + (S_R1+S_R2)  +  2 (C_R1+C_R2)
	# formula w/ 3rd rater: 2 (I_R3) + 2 (O_R3) + 2 (S_R3) + 4 (C_R3)
	# ======================
	def parse_training_set(self, path):
		rows = utils.read_csv_file(Path('{}/training_set_rel3.tsv'.format(path)), delimiter='\t')
		headers = None

		for row in rows:
			if headers is None:
				headers = row
				continue
			
			answer_id = int(row[0])
			question_id = int(row[1])
			answer_text = row[2].strip() if row[2] is not None else None
			raw_scores = row[3:]
			raw_scores_float = [float(score) if not score == '' else 0. for score in raw_scores]
			scores = []

			domain1_score = raw_scores_float[headers.index('domain1_score') - 3]
			domain2_score = raw_scores_float[headers.index('domain2_score') - 3]
			has_rater3 = not (row[headers.index('rater3_domain1')] == '')

			# question = next((q for q in self.questions if q.id == question_id), None)
			question = self.get_question(question_id)
			# print('question id: {}, answer count: {}'.format(question.id, len(question.answers)))
			if question_id == 2:
				scores = [(domain1_score + domain2_score) * 0.5]
			elif question_id < 7:
				scores = [domain1_score]
			elif has_rater3:
				rater3_traits = raw_scores_float[headers.index('rater3_trait1') - 3:]
				scores = [2 * rater3_traits[0] + 2 * rater3_traits[1] + 2 * rater3_traits[4] + 4 * rater3_traits[5]]
			else:
				rater1_traits = raw_scores_float[headers.index('rater1_trait1') - 3:headers.index('rater2_trait1') - 3]
				rater2_traits = raw_scores_float[headers.index('rater2_trait1') - 3:headers.index('rater3_trait1') - 3]
				scores = [rater1_traits[0] + rater2_traits[0] + rater1_traits[1] + rater2_traits[1] + rater1_traits[4] + rater2_traits[4] + 2 * (rater1_traits[5] + rater2_traits[5])]

			question.answers.append(Answer(answer_id=answer_id, text=answer_text, scores=scores))
			question.metadata['raw_scores'] = raw_scores

	def process(self, path):
		self.parse_essay_descriptions(path)
		self.parse_training_set(path)

class EncodedDataset(Dataset):
	def __init__(self, dataset, encoder):
		self.name = dataset.name
		self.metadata = dataset.metadata
		self.questions = dataset.questions
		self.encoder = encoder

	def __str__(self):
		string = '[ENCODED DATASET]\n'
		string += 'NAME: {}\n'.format(self.name)
		string += 'ENCODER: {}\n'.format(self.encoder.name)
		string += 'SCORE CLASSES: {}\n'.format(self.metadata.get('score_classes'))
		string += 'SCORE ENCODING: {}\n'.format(self.metadata.get('score_encoding'))
		string += 'QUESTION COUNT: {}\n'.format(len(self.questions))
		return string

	def get_subset(self, question_id):
		return SubDataset(encoded_dataset=self, question=self.get_question(question_id))

# Remark: method options: euclidean / cosine'
class SubDataset(object):
	def __init__(self, encoded_dataset, question, exclude_reference=True):
		self.question = question

		self.name = encoded_dataset.name
		self.metadata = encoded_dataset.metadata
		self.encoder = encoded_dataset.encoder

		if exclude_reference:
			self.answers = [answer for answer in self.question.answers if not answer.is_reference]
		else:
			self.answers = self.question.answers
		# self.answers = sorted(self.question.answers, key=lambda x: x.is_reference, reverse=True)

		self.reference_answers = [answer for answer in self.question.answers if answer.is_reference]
		self.reference_answer_data = [answer.encoding for answer in self.reference_answers]
		# self.data = []
		# self.data_dim, self.num_data = 0, 0
		self.set_data([answer.encoding for answer in self.answers])

	def __str__(self):
		string = '[SUBDATASET]\n'
		string += 'Name: {}\n'.format(self.name)
		string += 'Has Reference: {}\n'.format(self.metadata.get('has_reference_answer'))
		string += 'SCORE CLASSES: {}\n'.format(self.metadata.get('score_classes'))
		string += 'SCORE ENCODING: {}\n'.format(self.metadata.get('score_encoding'))
		string += 'QUESTION ID: {}\n'.format(self.question.id)
		string += 'NUMBER OF DATA: {}\n'.format(self.num_data)
		string += 'ENCODER: {}\n'.format(self.encoder.name)
		string += 'DATA DIMENSION: {}'.format(self.data_dim)
		# if self.pca is not None:
		# 	string += '\nPCA FACTOR: {}'.format(self.pca)
		return string

	def set_data(self, data):
		self.data = data
		self.data_dim, self.num_data = len(data[0]), len(data)
		# self.centroid = [statistics.mean(x) for x in zip(*(self.data))]
		# self.centroid = np.mean(self.data, axis=0)
