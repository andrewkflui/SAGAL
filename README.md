# GAL

### Prerequisites

#### Python 3.7.3
This project relies on Python 3.7.3. Please download and install it from https://www.python.org/downloads/release/python-373/

#### Packages
Please install the required packages listed in requirements_tf2.txt

#### Skip Thoughts
This project uses the library https://github.com/elvisyjlin/skip-thoughts.
The package has been included in this repository. Please copy the /frameworks/skip_thoughts_master/skip_thoughts folder to your package folder.

Other required packages can be installed with requirements_tf1.txt and requirements_tf2.txt for tf1 and tf2 supported environments respectively.
Encoding with Skip Thoughts and BERT requires tf1 to work while Google Universal Sentence Encoder requires tf2.


### Pre-trained models
For all pretrained models, please put under data/models/pretrained/{ENCODER_NAME}/

#### Google Universal Sentence Encoder

https://tfhub.dev/google/universal-sentence-encoder/4

#### Skip Thoughts

Please follow the instructions in this https://pypi.org/project/skip-thoughts/ to download the models.

#### GloVe

https://nlp.stanford.edu/projects/glove/

### Datasets

Please download the below datasets and put under *data/datasets/raw*

#### Powergrading Short Answer Grading Corpus

https://www.microsoft.com/en-us/download/details.aspx?id=52397

#### Semeval 2013 2 and 3 way

https://www.kaggle.com/datasets/smiles28/semeval-2013-2-and-3-way

### Usage

To view argument descriptions, please use --help

#### Dataset Processing

Before performing clustering, the raw datasets need to be processed and then converted to vectors.

Example:
> python dataset_processor.py --name=USCIS
> python text_encoder.py --name=USCIS --encoder=google_universal_sentence_encoder

#### Clustering (GAL)

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

#### Clustering (Others)

Example (for DBSCAN):
> python clustering.py --name=USCIS --question_id=8 --encoder=google_universal_sentence_encoder --eps=0.1 --min_samples=10 --save_mode=1 --algorithm=dbscan --plot=2d

Example (for Birch):
> python clustering.py --name=USCIS --question_id=8 --encoder=google_universal_sentence_encoder --cluster_num=30 --save_mode=1 --algorithm=birch --plot=2d
