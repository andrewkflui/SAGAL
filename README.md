# Short Answer Grading with Active Learning (SAGAL)

Copyright (C) 2021 - Andrew Lui, Vanessa Ng, Stella Cheung Wing-Nga

The Open University of Hong Kong

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, see http://www.gnu.org/licenses/.

## Introduction

Short answer question is a popular assessnent type. It expects constructed natural language responses that recall specific facts (Livingston, 2009; Burrows et al., 2015) or express subjective opinions and justifications. Automated Short Answer Grading (ASAG) is the research area that aims to develop techniques relevant to automated grading according to the grading examples provided by human graders. 

This repository contains the prototype implementation and the experiments of the Short Answer Grading with Active Learning (SAGAL) algorithm proposed in the following paper:

> Lui, A.K., Ng, S.C. and S.W.N. Cheung (in press), Automated Short Answer Grading with Computer-Assisted Grading Example Acquisition based on Active Learning, Interactive Learning Environment

Existing machine learning short answer grading models are often based on supervised learning on annotated datasets, but such human annotated datasets are expensive to acquire in practice. Active learning is a machine learning approach that iteratively acquires annotated data and updates the model (Saar-Tsechansky and Provost, 2004; Cohn et al., 1994), SAGAL uses active learning with a number of heuristics to optimize the procedure of acquiring grading examples. It can train short answer grading models to an accuracy level with significantly fewer number of annotations. The following figure outlines the apporach.

<img width="480" alt="Screenshot 2022-10-30 at 1 53 03 PM" src="https://user-images.githubusercontent.com/8808539/198864651-63aefa09-e49b-4791-bab1-792dc62de912.png">

#### Related Software

* [Perceptive Grader](https://github.com/andrewkflui/PerceptiveGrader): A web system demonstrating interactive building of grading models using SAGAL.

### Prerequisites

#### Python 3.7.3
This project relies on Python 3.7.3. Please download and install it from https://www.python.org/downloads/release/python-373/

#### Packages
Please install the required packages listed in requirements_tf2.txt

## Installation

The following comprises instructions for these two procedures.
* Conversion of three gold standard datasets into vector representation (i.e., embedding) stored as a serialized object file.
* Execution of the SAGAL algorithm and the variants on the datasets.

The implementation supports three popular sentence embedding models for turning short answers into vector represetations. 

### Preparation of Datasets

The SAGAL prototype implementation assumes that the input of short answers is already in vector represetation. A dataset processing tool is provided for the conversion of these three datasets.

* The USCIS dataset (Basu et al., 2013) contains 20 questions sampled from the United States Citizenship Examination. High specific answers are asked.
* The SciEntsBank dataset (Dzikovska, Myroslava O et al., 2013) is large and covers many disciplines in science. 
* The Hewlett Foundation (HF) dataset (Peters and Jankiewicz, 2012) consists of more open-ended questions. 

Download the below datasets and put under data/datasets/raw

#### Powergrading Short Answer Grading Corpus
https://www.microsoft.com/en-us/download/details.aspx?id=52397

#### Semeval 2013 2 and 3 way
https://www.kaggle.com/datasets/smiles28/semeval-2013-2-and-3-way

#### Hewlett Foundation
https://www.kaggle.com/c/asap-sas![image](https://user-images.githubusercontent.com/8808539/198863647-fb9a938d-dba2-4e6f-b9ad-c53ad19a2458.png)

### Pre-Trained Models of Sentence Embedding
Download and install one or more of the pre-trained sentence embedding models, except that the Skip Thoughts package is included in this repository. Please copy the /frameworks/skip_thoughts_master/skip_thoughts folder to your package folder.Other required packages can be installed with requirements_tf1.txt and requirements_tf2.txt for tf1 and tf2 supported environments respectively.

Encoding with Skip Thoughts and BERT requires tf1 to work while Google Universal Sentence Encoder requires tf2.

For all pretrained models, please put under data/models/pretrained/{ENCODER_NAME}/

#### Google Universal Sentence Encoder
`{ENCODER_NAME} = google_universal_sentence_encoder`

https://tfhub.dev/google/universal-sentence-encoder/4

#### Skip Thoughts
`{ENCODER_NAME} = skip_thoughts`

This project uses the library https://github.com/elvisyjlin/skip-thoughts.

Please follow the instructions in this https://pypi.org/project/skip-thoughts/ to download the models.

#### GloVe
`{ENCODER_NAME} = glove`

https://nlp.stanford.edu/projects/glove/

#### nltk
For tokenizing for stop word removals.
```
nltk.download('punkt')
nltk.download('stopwords')
```
### Dataset Processing

Before running the experiments, the raw datasets need to be processed.

Example:
> python dataset_processor.py --name=USCIS
> python text_encoder.py --name=USCIS --encoder=google_universal_sentence_encoder


To view argument descriptions, please use --help

### Executing SAGAL

version:
* 5: The latest version
* 4: Remove a proportion of subspaces set by subspace_replacement_ratio

grade_assignment_method:
* parent: Delta link
* nearest_true_grade: Nearest true grade
* oc: One class
* moc: Multiple One class

label_search_boundary_factor:
* affects the searching boundary for 'moc' and 'oc'
* boundary is multiplied by RD cutoff in each subspace

exclusion_rd_reduction_factor:
* the value to be used in each exclusion RD deduction

relevant_subspace_number:
* affects the number of relevant subspaces
* no subspace selection if given 0 or None

For more details, please use --help

Example:
> python clustering.py --name=USCIS --question_id=3 --grading_actions=30 --grade_assignment_method='nearest_true_grade' --label_search_boundary_factor=0.25 --save_mode=0

#### Executing baseline unsupervised learning algorithms

Example (for DBSCAN):
> python clustering.py --name=USCIS --question_id=8 --encoder=google_universal_sentence_encoder --eps=0.1 --min_samples=10 --save_mode=1 --algorithm=dbscan --plot=2d

Example (for Birch):
> python clustering.py --name=USCIS --question_id=8 --encoder=google_universal_sentence_encoder --cluster_num=30 --save_mode=1 --algorithm=birch --plot=2d
