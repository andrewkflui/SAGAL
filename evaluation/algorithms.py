import copy
import numpy as np
from abc import ABCMeta

# import hdbscan
from sklearn.cluster import DBSCAN, Birch, KMeans, OPTICS, SpectralClustering
from sklearn.mixture import GaussianMixture

from core import utils
from core.algorithms import Algorithm, GAL
from basis import config
from basis.constants import ALGORITHMS, UNCLASSIFIED_CLUSTER
from basis.solutions import Solution

def setup(data, args):
	algorithm = args.get('algorithm', ALGORITHMS[0])
	cluster_num = args.get('cluster_num', None)

	if algorithm == 'moac':
		if config.SCHEME == 2 or config.SCHEME == 3:
			from core.algorithms import MOACV2
			return MOACV2(data, gstar=None, **args)
		elif config.SCHEME == 1:
			from core.algorithms import MOACV1
			return MOACV1(data, **args)
		else:
			from core.algorithms import MOAC
			return MOAC(data, **args)
	elif algorithm == 'gal':
		return GAL(**args)
	elif algorithm == 'birch':
		return SklearnBirch(data, n_clusters=cluster_num, args=args)
	elif algorithm == 'dbscan':
		eps = args.get('eps', 0.5)
		min_samples = args.get('min_samples', 10)
		count_unclassified = args.get('count_unclassified', False)
		return SklearnDBSCAN(data, eps=eps, min_samples=min_samples,
			count_unclassified=count_unclassified, args=args)
	elif algorithm == 'kmeans':
		return SklearnKMeans(data, n_clusters=cluster_num, random_state=args.get('random_state', 0))
	elif algorithm == 'gaussian_mixture':
		# return SklearnGaussianMixture(data, n_components=cluster_num, covariance_type='tied',
		# 	tol=1e-3, reg_covar=1e-6, max_iter=100, n_init=30, init_params='kmeans')
		return SklearnGaussianMixture(data, n_components=cluster_num, args=args)
	elif algorithm == 'spectral_clustering':
		return SklearnSpectralClustering(data, n_clusters=cluster_num, args=args)
	elif algorithm == 'hdbscan':
		# count_unclassified = args.get('count_unclassified', False)
		# return HDBSCAN(data, count_unclassified=count_unclassified)
		return HDBSCAN(data, args=args)
	elif algorithm == 'optics':
		args['algorithm'] = args.get('neighbour_algorithm', 'auto')
		return SklearnOPTICS(data, args=args)
	else:
		raise utils.InvalidNameError('algorithm', algorithm)

class SklearnBirch(Algorithm, Birch):
	def __init__(self, data, n_clusters=None, args=dict()):
		Algorithm.__init__(self, 'birch', data)
		Birch.__init__(self, n_clusters=n_clusters)
		self.set_attributes(args)

	def __str__(self):
		string = super().__str__()
		string += '\nN Clusters: {}'.format(self.n_clusters)
		return string

	def run(self):
		self.result = self.fit_predict(self.data)

	def get_result(self):
		if self.result is None:
			return None

		max_cluster_num = self.n_clusters if self.n_clusters is not None else np.max(self.result) + 1
		# solution = BirchSolution(self.data, labels=self.result + 1, max_cluster_num=max_cluster_num)
		solution = BirchSolution(self.data, labels=self.result + 1, threshold=self.threshold,
			branching_factor=self.branching_factor, max_cluster_num=max_cluster_num)
		solution.update_variables()
		return [solution]

class SklearnDBSCAN(Algorithm, DBSCAN):
	def __init__(self, data, eps=0.01, min_samples=10, count_unclassified=False, args=dict()):
		metric = args.get('metric', 'euclidean')
		if metric == 'angular':
			metric = self.dist_angular
			self.metric_name = 'angular'
		else:
			self.metric_name = metric
		Algorithm.__init__(self, 'dbscan', data)
		DBSCAN.__init__(self, eps=eps, min_samples=min_samples, metric=metric)
		self.count_unclassified = count_unclassified

	def __str__(self):
		string = super().__str__()
		string += '\nEPS: {}'.format(self.eps)
		string += '\nMin Samples: {}'.format(self.min_samples)
		string += '\nMetric: {}'.format(self.metric)
		return string

	def dist_angular(self, x, y):
		return utils.dist([x], [y], 'angular')[0]

	def run(self):
		self.result = self.fit_predict(self.data)

	def get_result(self):
		if self.result is None:
			return None
		solution = DBSCANSolution(self.data, labels=self.result + 1, eps=self.eps,
			min_samples=self.min_samples, metric=self.metric_name, max_cluster_num=np.max(self.result) + 2,
			count_unclassified=self.count_unclassified)
		return [solution]

class SklearnKMeans(Algorithm, KMeans):
	def __init__(self, data, n_clusters, random_state=0):
		Algorithm.__init__(self, 'kmeans', data)
		KMeans.__init__(self, n_clusters=n_clusters, random_state=random_state)

	def __str__(self):
		string = super().__str__()
		string += '\nN Clusters: {}'.format(self.n_clusters)
		string += '\nRandom State: {}'.format(self.random_state)
		return string

	def run(self):
		self.result = self.fit_predict(self.data)

	def get_result(self):
		if self.result is None:
			return None
		solution = KMeansSolution(self.data, labels=self.result + 1, max_cluster_num=self.n_clusters,
			random_state=self.random_state)
		solution.update_variables()
		return [solution]

# class KMeansPlusPlus(Algorithm):
# 	def __init__(self, data, n_clusters, num_runs, converge_distance=0.1, random_state=0):
# 		Algorithm.__init__(self, 'kmeans++', data)
# 		self.n_clusters = n_clusters
# 		self.num_runs = num_runs
# 		self.converge_distance = converge_distance
# 		self.random_state = random_state

# 	def __str__(self):
# 		string = super().__str__()
# 		string += '\nN Clusters: {}'.format(self.n_clusters)
# 		string += '\nNumber of Runs in Parallel: {}'.format(self.num_runs)
# 		string += '\nConverge Distance: {}'.format(self.converge_distance)
# 		string += '\nRandom State: {}'.format(self.random_state)
# 		return string

# 	def compute_entropy(self, frequency_vector):
# 		frequency_vector = 1.0 * frequency_vector / frequency_vector.sum()
# 		return -np.sum(frequency_vector * np.log2(frequency_vector))

# 	def choice(self, probabilities):
# 		# Generate a random sample from [0, len(p))
# 		random_number = np.random.random()
# 		for i in range(len(probabilities)):
# 			s += probabilities[i]
# 			if s > random_number:
# 				return i
# 		return None

# 	def run(self):
# 		pass

# 	def get_result(self):
# 		if self.result is None:
# 			return None
# 		solution = KMeansPlusPlusSolution(self.data, labels=self.result + 1,
# 			max_cluster_num=self.n_clusters, random_state=self.random_state)
# 		solution.update_variables()
# 		return [solution]

class SklearnGaussianMixture(Algorithm, GaussianMixture):
	# def __init__(self, data, n_components=1, covariance_type='full', tol=1e-3, reg_covar=1e-6,
 # 		max_iter=100, n_init=1, init_params='kmeans'):
	# 	Algorithm.__init__(self, 'gaussian_mixture', data)
	# 	# GaussianMixture.__init__(self, n_components=n_components, covariance_type=covariance_type,
	# 	# 	tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params)
	# 	GaussianMixture.__init__(self, n_components=n_components, covariance_type=covariance_type)

	def __init__(self, data, n_components, args=dict()):
		Algorithm.__init__(self, 'gaussian_mixture', data)
		GaussianMixture.__init__(self, n_components=n_components, random_state=args.get('random_state', None))
		self.set_attributes(args)
	
	def __str__(self):
		string = super().__str__()
		return string

	def run(self):
 		self.result = self.fit_predict(self.data)

	def get_result(self):
		if self.result is None:
			return None
		solution = GaussianMixtureSolution(self.data, labels=self.result + 1,
			max_cluster_num=np.max(self.result) + 1, n_components=self.n_components,
			covariance_type=self.covariance_type, tol=self.tol, reg_covar=self.reg_covar,
			max_iter=self.max_iter, n_init=self.n_init, init_params=self.init_params,
			weights_init=self.weights_init, means_init=self.means_init,
			precisions_init=self.precisions_init, warm_start=self.warm_start, random_state=self.random_state)
		solution.update_variables()
		return [solution]

	def get_proba(self):
		return self.predict_proba(self.data)

class SklearnSpectralClustering(Algorithm, SpectralClustering):
	def __init__(self, data, n_clusters, random_state=0, args=dict()):
		Algorithm.__init__(self, 'spectral_clustering', data)
		#eigen_solver{‘arpack’, ‘lobpcg’, ‘amg’}, default=None (arpack)
		# n_component: number of eigen vectors ot use
		# n_init: number of time the k-means algorithm will be run with difference centroid seeds
		# gamma
		# kernal coefficient for rbf, plot, sigmod, laplacian and chi2 kernels.
		# ignored for affinity='nearest_neighbours'

		# affinity:
		# str or callable, default = 'rbf'
		# 'nearest_neighbors' / 'rgb' / 'precomputed' / 'precomputed_nearest_neighbors' /
		# one of the kernels supported by pairwaise_kernels

		# n_neighbors: default = 10
		
		# eigen_tol: defalut = 0.0
		# stopping criterion for eigndecomposition of the Laplacian matrix when eigen_solver='arpack'

		# assign_labels: {'kmeans' / 'discretize'}, default = 'kmeans'

		# degree: default = 3
		# degree of the plynomial kernal. ignored by other kernels

		# coef0: default = 1
		# zero coefficient for ploynomial and sigmoid kernels. ignored by other kernels

		# kernel_params: dict of str to any, default = None
		# parameters for keernal passed as callable object, ignored by other kernels

		# n_jobs: default = None
		# number of paraleel jobs to run when affinity='nearest_neighbors'
		SpectralClustering.__init__(self, n_clusters=n_clusters, random_state=random_state)
		for key, value in args.items():
			# if getattr(self, key, '') is not '':
			if hasattr(self, key):
				setattr(self, key, value)
	
	def __str__(self):
		string = super().__str__()
		for k, v in self.get_params().items():
			string += '{}: {}\n'.format(k, v)
		return string

	def get_params(self):
		visible_params = dict()
		for k, v in self.__dict__.items():
			if k[0] == '_' or k[len(k)-1] == '_' or k in ['version', 'name', 'data', 'result', 'args']:
				continue
			visible_params[k] = v
		return visible_params

	def run(self):
		self.result = self.fit_predict(self.data)

	def get_result(self):
		if self.result is None:
			return None
		solution = SpectralClusteringSolution(self.data, labels=self.result + 1,
			max_cluster_num=self.n_clusters, params=self.get_params())
		solution.update_variables()
		return [solution]

class HDBSCAN(Algorithm):
	def __init__(self, data, args=dict()):
		Algorithm.__init__(self, 'hdbscan', data)
		metric = args.get('metric', 'euclidean')
		if metric == 'angular':
			metric = self.dist_angular
			metric_name = 'angular'
		else:
			metric_name = metric
		self.clusterer = hdbscan.HDBSCAN(algorithm='best',
			alpha=args.get('alpha', 1.0), approx_min_span_tree=args.get('approx_min_span_tree', True),
			gen_min_span_tree=args.get('gen_min_span_tree', False), leaf_size=args.get('leaf_size', 40),
			metric=metric, min_cluster_size=args.get('min_cluster_size', 2),
			min_samples=args.get('min_samples', None), p=args.get('p', None))
		self.count_unclassified = args.get('count_unclassified', False)
		self.clusterer.metric_name = metric_name

	def __str__(self):
		string = super().__str__()
		string += 'Metric: {}\n'.format(self.clusterer.metric_name)
		string += 'Min Samples: {}\n'.format(self.clusterer.min_samples)
		return string

	def dist_angular(self, x, y):
		return utils.dist([x], [y], 'angular')[0]

	def run(self):
		self.clusterer.fit(self.data)
		self.result = self.clusterer.labels_

	def get_result(self):
		if self.result is None:
			return None
		solution = HDBSCANSolution(self.data, labels=self.result + 1,
			max_cluster_num=self.clusterer.labels_.max() + 2, count_unclassified=self.count_unclassified,
			clusterer=self.clusterer)
		solution.update_variables()
		return [solution]

class SklearnOPTICS(Algorithm, OPTICS):
	def __init__(self, data, args=dict()):
		Algorithm.__init__(self, 'optics', data)
		# min_samples: int > 1 or float between 0 and 1, default = 5

		# max_eps: float, default = np.inf
		# maximum distance between two samples for one to be considered as in the neighbourhood of
		# the other. smaller max_eps, shorter run times

		# metric: any metric from sklearn or scipy.spatial.distance can be used
		# for distance computation

		# p: 1 = ,manhattan_distance (l1), 2 = euclidean (l2), arbitrary = minkowski_distance (l_p)
		# parameter for the Minkowski metric from pairwise_distance

		# metric_params: dict, default = None
		# additional keyword arguments for the metric function

		# cluster_method: 'xi' or 'dbscan'
		# extraction method used to extract clusters using the calculated reachability and ordering

		# eps: float, default = max_eps
		# used only when cluster_method='dbscan'

		# xi: float bewteen 0 and 1, default = 0.05
		# determines the minimum steepness on the reachbility plot that constitutes a cluster boundary

		# predecessor_correction: bool, default = True
		# used only when cluster_method='xi'
		# correct clusters according to the predecessors calculated by OPTICS
		# has minial effec on most datasets

		# min_cluster_size, int > 1 or float between 0 and 1, default = None
		# used only when cluster_method = 'xi'
		# minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction
		# of the number of samples (rounded to be at least 2)
		# If None, value of min_samples is used

		# algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, default = 'auto'
		# fitting on sparse input will override the setting of this parameter, using brute force
		# 'ball_tree' = BallTree
		# 'kd_tree'  = KDTree
		# 'brute' = brute-force search
		# 'auto' = attempt to decide the most appropriate algorithm based on the fitted values

		# leaf_size: int, default  = 30
		# lef size passed to BallTree or KDTree
		# can affect the spped of the construction and query, as well as the memory required to store the tree
		# optimal value depends on the nature of the problem

		# n_jobs, int, default = None
		# number of parallel jobs to run for neighbours search
		# None means 1 unless in a joblib.parallel_backend context -1 means using all processors

		OPTICS.__init__(self, min_samples=args.get('min_samples', 5), max_eps=args.get('max_eps', np.inf),
			metric=args.get('metric', 'minkowski'), p=args.get('p', 2),
			metric_params=args.get('metric_params', None),
			cluster_method=args.get('cluster_method', 'xi'), eps=args.get('eps', None),
			xi=args.get('xi', 0.05), predecessor_correction=args.get('predecessor_correction', True),
			min_cluster_size=args.get('min_cluster_size', None), algorithm=args.get('algorithm', 'auto'),
			leaf_size=args.get('leaf_size', 30), n_jobs=args.get('n_jobs', None))
		
		self.count_unclassified = args.get('count_unclassified', False)
		self.param_keys = ['min_samples', 'max_eps', 'metric', 'p', 'metric_params', 'cluster_method',
			'eps', 'xi', 'prodecessor_correction', 'min_cluster_size', 'algorithm', 'leaf_size', 'n_jobs']
		self.params = dict()
		for key, value in self.__dict__.items():
			if key in self.param_keys:
				self.params[key] = value

	def __str__(self):
		string = super().__str__()
		return string

	def run(self):
		self.result = self.fit_predict(self.data)

	def get_result(self):
		if self.result is None:
			return None
		solution = OPTICSSolution(self.data, labels=self.result + 1,
			max_cluster_num=self.labels_.max() + 2, params=self.params,
			count_unclassified=self.count_unclassified)
		solution.update_variables()
		return [solution]

class LabelEncodingSolution(Solution):
	__meta__ = ABCMeta

	def __init__(self, data, max_cluster_num, reserved_cluster_number=0):
		super().__init__(data, encoding='label')
		self.max_cluster_num = max_cluster_num
		self.reserved_cluster_number = reserved_cluster_number

		self.labels = None
		self.cluster_size = None
		self.centroids = None
		self.cluster_data_indices = None
		self.count_cluster = 0
		
		self.cluster_scores = None

	def __str__(self):
		string = '[Solution]\n'
		string += 'Number of Data: {}\n'.format(len(self.data))
		string += 'Data Dimension: {}\n'.format(len(self.data[0]))
		string += 'Score: {}\n'.format(self.score)
		string += self.additional_string()
		string += 'Number of Clusters: {}\n'.format(self.count_cluster)
		return string

	def additional_string(self):
		return ''

	def set_attributes(self, dictionary):
		for key, value in dictionary.items():
			setattr(self, key, copy.copy(value))

	def update_variables(self, distance_function=None, clear_scores=True):
		self.cluster_size = [0] * self.max_cluster_num
		self.sse = [np.nan] * len(self.data)
		self.centroids = [None] * self.max_cluster_num
		self.cluster_data_indices = [[]] * self.max_cluster_num
		# self.cluster_nd = [0] * self.max_cluster_num
		self.cluster_scores = [np.nan] * self.max_cluster_num if clear_scores or self.cluster_scores is None else self.cluster_scores
		self.count_cluster = 0

		unique_counts = np.unique(self.labels, return_counts=True)
		for i in range(len(unique_counts[0])):
			label = unique_counts[0][i]
			count = unique_counts[1][i]

			index = label + self.reserved_cluster_number - 1

			self.cluster_size[index] = count
			if index >= self.reserved_cluster_number:
				self.count_cluster += 1
			
			self.cluster_data_indices[index] = np.where(np.array(self.labels) == label)[0].tolist()
			self.centroids[index] = np.average(np.array(self.data)[self.cluster_data_indices[index]], axis=0)
			# self.cluster_nd[index] = np.average(normalized_densities[self.cluster_data_indices[index]])
			if distance_function == 'euclidean':
				for d in self.cluster_data_indices[index]:
					self.sse[d] = utils.dist(self.data[d], self.centroids[index], distance_function)
			elif distance_function is not None:
				sse = utils.dist(self.data[self.cluster_data_indices[index]], [self.centroids[index]], distance_function)
				for i in range(len(self.cluster_data_indices[index])):
					self.sse[self.cluster_data_indices[index][i]] = sse[i]

		self.updated = True

	def update_max_cluster_num(self, max_cluster_num):
		self.cluster_size = [self.cluster_size[i] if i < len(self.cluster_size) else 0 for i in range(max_cluster_num)]
		self.centroids = [self.centroids[i] if i < len(self.centroids) else None for i in range(max_cluster_num)]
		self.cluster_data_indices = [self.cluster_data_indices[i] if i < len(self.cluster_data_indices) else [] for i in range(max_cluster_num)]
		self.cluster_states = [self.cluster_states[i] if i < len(self.cluster_states) else 1 for i in range(max_cluster_num)]
		self.max_cluster_num = max_cluster_num

class GradeEncodingSolution(LabelEncodingSolution):
	__meta__ = ABCMeta

	def __init__(self, data, assigned_grades, assignment_source_map):
		super().__init__(data, 0, 0)
		self.assigned_grades = assigned_grades
		self.assignment_source_map = assignment_source_map
		self.build_labels()
		self.max_cluster_num = self.count_cluster

	def __str__(self):
		return super().__str__()

	def assign_from_source(self, index):
		source_type, source_index = self.assignment_source_map[index]
		if type(source_index) == str:
			source_index = index
		elif self.labels[source_index] <= 0:
			self.assign_from_source(source_index)
		self.labels[index] = self.labels[source_index]

	def build_labels(self):
		self.count_cluster = 0
		self.default_cluster_exists = False
		
		self.labels = np.zeros(shape=len(self.assigned_grades), dtype=int)

		centroids = [i for i in range(len(self.assigned_grades)) \
			if self.assignment_source_map[i][1] == 'Self']
		for i in centroids:
			self.count_cluster += 1
			self.labels[i] = self.count_cluster

		for i in range(len(self.assigned_grades)):
			if i in centroids:
				continue
			
			assigned_grade = self.assigned_grades[i]
			source = self.assignment_source_map[i]
			if type(source[1]) == str:
				self.default_cluster_exists = True
				self.labels[i] = self.count_cluster + 1
			else:
				self.assign_from_source(i)

		if self.default_cluster_exists:
			self.count_cluster += 1

class TrueGradeEncodingSolution(Solution):
	__meta__ = ABCMeta

	def __init__(self, true_grade_labels):
		super().__init__(true_grade_labels, encoding='true_grade')
		self.labels = true_grade_labels

	def __str__(self):
		return super().__str__()

class DBSCANSolution(LabelEncodingSolution):
	def __init__(self, data, labels, eps, min_samples, metric, max_cluster_num, count_unclassified=False):
		super().__init__(data, max_cluster_num, 1)
		self.labels = labels
		self.eps = eps
		self.metric = metric
		self.min_samples = min_samples
		self.count_unclassified = count_unclassified

		self.unclassified_num = 0
		self.unclassified_indices = []

		self.update_variables()

	def get_printable_params(self):
		return {
			'Eps': self.eps,
			'Min Samples': self.min_samples,
			'Metric': self.metric,
			'Unclassified Number': self.unclassified_num,
			'Count Unclassified': self.count_unclassified
		}

	def update_variables(self):
		super().update_variables()
		if self.count_unclassified and 0 in self.labels:
			self.count_cluster += 1
		self.unclassified_num = len(self.cluster_data_indices[UNCLASSIFIED_CLUSTER])
		self.unclassified_indices = self.cluster_data_indices[UNCLASSIFIED_CLUSTER]

class BirchSolution(LabelEncodingSolution):
	def __init__(self, data, labels, threshold, branching_factor, max_cluster_num=None):
		super().__init__(data, max_cluster_num, 0)
		self.labels = labels
		self.threshold = threshold
		self.branching_factor = branching_factor
		self.update_variables()

	def get_printable_params(self):
		return {
			'Threshold': self.threshold,
			'Branching Factor': self.branching_factor,
			'Maximum Cluster Number': self.max_cluster_num
		}

class KMeansSolution(LabelEncodingSolution):
	def __init__(self, data, labels, max_cluster_num, random_state=0):
		super().__init__(data, max_cluster_num, 0)
		self.labels = labels
		self.random_state = random_state
		self.update_variables()

	def get_printable_params(self):
		return {
			'Maximum Cluster Number': self.max_cluster_num,
			'Random State': self.random_state
		}

class KMeansPlusPlusSolution(KMeansSolution):
	pass

class GaussianMixtureSolution(LabelEncodingSolution):
 	def __init__(self, data, labels, max_cluster_num, n_components=1, covariance_type='full', tol=1e-3,
 		reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans', weights_init=None,
 		means_init=None, precisions_init=None, warm_start=False, random_state=None):
 		super().__init__(data, max_cluster_num, 0)
 		self.labels = labels
 		self.covariance_type = covariance_type
 		self.tol = tol
 		self.reg_covar = reg_covar
 		self.max_iter = max_iter
 		self.n_init = n_init
 		self.init_params = init_params
 		self.weights_init = weights_init
 		self.means_init = means_init
 		self.precisions_init = precisions_init
 		self.warm_start = warm_start
 		self.random_state = random_state

 		self.means_init_count = len(self.means_init) if self.means_init is not None else 9
 		self.update_variables()

 	def get_printable_params(self):
 		return {
 			'Covariance Type': self.covariance_type,
 			'Covariance Threshold (tol)': self.tol,
 			'Non-negative Regularization Covariance': self.reg_covar,
 			'Maximum EM Iterations': self.max_iter,
 			'Number of initialization (n_init)': self.n_init,
 			'Method to initialize weights (init_params)': self.init_params,
 			'Initial Weights (weights_init)': self.weights_init,
 			'Inital Means (means_init) Count': len(self.means_init) if self.means_init is not None else 0,
 			'Initial Precisions (precisions_init)': self.precisions_init,
 			'Warm Start': self.warm_start,
 			'Random State': self.random_state
 		}

class SpectralClusteringSolution(LabelEncodingSolution):
	def __init__(self, data, labels, max_cluster_num, params):
		super().__init__(data, max_cluster_num, 0)
		self.labels = labels
		self.params = params
		self.update_variables()

class HDBSCANSolution(LabelEncodingSolution):
 	def __init__(self, data, labels, max_cluster_num, count_unclassified=False, clusterer=None):
 		super().__init__(data, max_cluster_num, 1)
 		self.labels = labels
 		self.count_unclassified = count_unclassified
 		self.clusterer = clusterer

 		self.unclassified_num = 0
 		self.unclassified_indices = []

 		self.update_variables()

 	def __str__(self):
 		string = super().__str__()
 		return string

 	def get_printable_params(self):
 		return {
 			'Metric': self.clusterer.metric_name,
 			'Min Samples': self.clusterer.min_samples,
 			'Count Unclassified': self.count_unclassified,
 			'Unclassified Number': self.unclassified_num,
 			'Unclassified Indices': self.unclassified_indices
 		}

 	def update_variables(self):
 		super().update_variables()
 		if self.count_unclassified and 0 in self.labels:
 			self.count_cluster += 1
 		self.unclassified_num = len(self.cluster_data_indices[UNCLASSIFIED_CLUSTER])
 		self.unclassified_indices = self.cluster_data_indices[UNCLASSIFIED_CLUSTER]

class OPTICSSolution(LabelEncodingSolution):
 	def __init__(self, data, labels, max_cluster_num, params, count_unclassified=False):
 		super().__init__(data, max_cluster_num, 1)
 		self.labels = labels
 		self.params = params
 		self.count_unclassified = count_unclassified

 		self.unclassified_num = 0
 		self.unclassified_indices = []

 		self.update_variables()

 	def __str__(self):
 		string = super().__str__()
 		return string

 	def update_variables(self):
 		super().update_variables()
 		if self.count_unclassified and 0 in self.labels:
 			self.count_cluster += 1
 		self.unclassified_num = len(self.cluster_data_indices[UNCLASSIFIED_CLUSTER])
 		self.unclassified_indices = self.cluster_data_indices[UNCLASSIFIED_CLUSTER]

class CoreDataSolution(LabelEncodingSolution):
	def __init__(self, problem, density_method, radius_grade, root_index, core_pop_indices,
		core_weak_pop_indices, non_core_indices, merging_criteria=None):
		self.problem = problem
		self.density_method = density_method
		self.radius_grade = radius_grade

		self.root_index = root_index
		self.core_pop_indices = core_pop_indices
		self.core_weak_pop_indices = core_weak_pop_indices
		self.non_core_indices = non_core_indices
		self.merging_criteria = merging_criteria
		
		self.centroid_indices = [root_index] + core_pop_indices + core_weak_pop_indices
		super().__init__(problem.data, len(self.centroid_indices), 0)

		self.reduced_labels = []
		self.labels = []
		self.cluster_data_indices = []
		
		self.update_variables()

	def get_original_data_indices(self, reduced_data_indices):
		data_indices = []
		for index in reduced_data_indices:
			data_indices += np.where(self.problem.original_to_reduced_index_map == index)[0].tolist()
		return data_indices

	def merge_clusters_by_centroid_data_index(self, source_position, target_position):
		cluster_data_indices = []
		for i in range(len(self.centroid_indices)):
			if i == source_position:
				continue
			elif i == target_position:
				cluster_data_indices.append(self.cluster_data_indices[i] + self.cluster_data_indices[source_position])
			else:
				cluster_data_indices.append(self.cluster_data_indices[i])

		self.cluster_data_indices = cluster_data_indices
		self.count_cluster = len(self.cluster_data_indices)
		self.centroid_indices.remove(self.centroid_indices[source_position])

	def merge_clusters_by_radius_grade(self):
		while True:
			cluster_distances = np.zeros(shape=(len(self.centroid_indices), len(self.centroid_indices)))
			for i in range(len(self.centroid_indices)):
				cluster_distances[i] = self.problem.reduced_data_distances[self.centroid_indices[i]][self.centroid_indices]
				cluster_distances[i][i] = np.inf

			min_distance_index = np.argwhere(cluster_distances == cluster_distances.min())[0]
			target_position, source_position = min_distance_index

			source_centroid_index = self.centroid_indices[source_position]
			target_centroid_index = self.centroid_indices[target_position]
			if self.problem.reduced_data_distances[source_centroid_index][target_centroid_index] \
				>= self.radius_grade:
				break

			self.merge_clusters_by_centroid_data_index(source_position, target_position)

	def merge_clusters_by_target_number(self, target_number=None):
		if target_number is None or self.count_cluster <= target_number:
			return

		while self.count_cluster > target_number:
			cluster_distances = np.zeros(shape=(len(self.centroid_indices), len(self.centroid_indices)))
			for i in range(len(self.centroid_indices)):
				cluster_distances[i] = self.problem.reduced_data_distances[self.centroid_indices[i]][self.centroid_indices]
				cluster_distances[i][i] = np.inf

			min_distance_index = np.argwhere(cluster_distances == cluster_distances.min())[0]
			target_position, source_position = min_distance_index

			self.merge_clusters_by_centroid_data_index(source_position, target_position)

	def update_variables(self):
		self.reduced_labels = np.zeros(shape=self.problem.reduced_num_data, dtype=int)
		self.labels = np.zeros(shape=self.problem.num_data, dtype=int)
		self.cluster_data_indices = []

		for i in range(len(self.centroid_indices)):
			self.cluster_data_indices.append([self.centroid_indices[i]])
			self.reduced_labels[self.centroid_indices[i]] = i + 1
			self.labels[self.get_original_data_indices([self.centroid_indices[i]])] = i + 1

		for non_core_index in self.non_core_indices:
			distances = self.problem.reduced_data_distances[non_core_index][self.centroid_indices]
			cluster_index = np.argmin(distances)
			self.cluster_data_indices[cluster_index].append(non_core_index)
			self.reduced_labels[non_core_index] = cluster_index + 1
			self.labels[self.get_original_data_indices([non_core_index])] = cluster_index + 1

		self.count_cluster = len(self.centroid_indices)
		self.updated = True

		if self.merging_criteria == 'radius_grade':
			self.merge_clusters_by_radius_grade()
		elif type(self.merging_criteria) == int:
			self.merge_clusters_by_target_number(target_number=self.merging_criteria)

		# update labels
		for c in range(len(self.cluster_data_indices)):
			reduced_indices = self.cluster_data_indices[c]
			self.reduced_labels[reduced_indices] = c + 1
			self.labels[self.get_original_data_indices(reduced_indices)] = c + 1

		self.updated = True

class ModesSolution(LabelEncodingSolution):
	def __init__(self, problem, labels, assigned_grades, centroid_indices):
		super().__init__(problem.data, np.max(labels), 0)
		self.problem = problem
		self.labels = labels
		self.assigned_grades = assigned_grades
		self.centroid_indices = centroid_indices
		self.update_variables()

	def __str__(self):
		string = super().__str__()
		return string

	def get_original_data_indices(self, reduced_data_indices):
		data_indices = []
		for index in reduced_data_indices:
			data_indices += np.where(self.problem.original_to_reduced_index_map == index)[0].tolist()
		return data_indices

class GALSolution(TrueGradeEncodingSolution):
	def __init__(self, true_grade_labels, params=None):
		super().__init__(true_grade_labels)
		self.params = params

	def __str__(self):
		string = super().__str__()
		return string

	def get_printable_params(self):
		return self.params or dict()
