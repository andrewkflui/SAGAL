import os, sys, csv, pickle, argparse
from pathlib import Path
from enum import Enum

import xlrd, docx2txt
from xml.etree import ElementTree

from core import utils
from basis import constants
from datasets import datasets

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--root', help='datasets folder', default='{}/data/datasets/raw'.format(os.path.abspath(os.path.dirname(__file__))))
	parser.add_argument('--name', help='dataset name, options={}'.format([key for key in constants.DATASETS.keys()]), required=True)

	args = parser.parse_args()
	root = Path(args.root)
	name = args.name

	utils.validate_option(constants.DATASETS.keys(), name, 'dataset')

	root_path = root / constants.DATASETS[name]['source_folder']

	if name == 'SEB2':
		dataset = datasets.SciEntsBank2013Task7('2', '2way', score_classes=['incorrect', 'correct'], root_path=root_path)
	elif name == 'SEB3':
		dataset = datasets.SciEntsBank2013Task7('3', '3way', score_classes=['incorrect', 'contradictory', 'correct'], root_path=root_path)
	elif name == 'SEB5':
		dataset = datasets.SciEntsBank2013Task7('5', 'Core', score_classes=['non_domain', 'irrelevant', 'contradictory', 'partially_correct_incomplete', 'correct'], root_path=root_path)
	elif name == 'USCIS':
		dataset = datasets.USCIS(root_path=root_path)
	elif name == 'USCIS':
		dataset = datasets.USCIS(root_path=root_path, include_100=True)
	elif name == 'Mobley':
		dataset = datasets.Mobley(root_path=root_path)
	elif name == 'CAK':
		dataset = datasets.ChakrabortyAndKonar(root_path=root_path)
	elif name == 'ASAP':
		dataset = datasets.ASAP(root_path=root_path)

	utils.save(dataset, '{}/data/datasets/processed/{}'.format(os.path.abspath(os.path.dirname(__file__)), constants.DATASETS[name]['processed_folder']), 'data.txt')
	print(dataset)

	# sciEntsBank2way = SciEntsBank2013Task7('2way', '2way', score_classes=['correct', 'incorrect'], root_path=Path('datasets/raw/semeval2013-Task7-2and3way'))
	# sciEntsBank3way = SciEntsBank2013Task7('3way', '3way', score_classes=['contradictory', 'non_domain', 'correct'], root_path=Path('datasets/raw/semeval2013-Task7-2and3way'))
	# sciEntsBank5way = SciEntsBank2013Task7('5way', 'Core', score_classes=['non_domain', 'irrelevant', 'contradictory', 'partially_correct_incomplete', 'correct'], root_path=Path('datasets/raw/semeval2013-Task7-5way'))
	# uscis = USCIS(root_path=Path('datasets/raw/Powergrading-1.0-Corpus'))
	# mobley = Mobley(root_path=Path('datasets/raw/ShortAnswerGrading_v2.0'))
	# chakrabortyAndKonar = ChakrabortyAndKonar(root_path=Path('datasets/raw/chakrabortyAndKonar/Data_SingleSentence.xlsx'))
	# asap = ASAP(root_path=Path('datasets/raw/asap-aes'))
