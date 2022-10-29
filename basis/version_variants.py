from abc import ABC, ABCMeta, abstractmethod

class MissingParamerror(Exception):
	def __init__(self, version, variant, param_name):
		self.version = version
		self.variant = variant
		self.param_name = param_name

	def __str__(self):
		return '{} MUST BE PROVIDED FOR V{}_{}'.format(self.param_name, self.version, self.variant)

class VersionVariant(ABC):
	def __init__(self, version, variant, params=None, **kwargs):
		self.version = version
		self.variant = variant
		self.params = dict() if params is None else params

		for k, v in kwargs.items():
			setattr(self, k, v)

class V6_0812A(VersionVariant):
	def __init__(self, version=6, variant='0812A', params=None, **kwargs):
		VersionVariant.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'rd_cutoff': 0.4, 'diverse_answer_queries': True,
			'dynamic_grade_weights': False})

class V6_0812B(VersionVariant):
	def __init__(self, version=6, variant='0812B', params=None, **kwargs):
		VersionVariant.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'rd_cutoff': 0.4, 'delta_link_threshold_factor': 2.0,
			'diverse_answer_queries': False, 'dynamic_grade_weights': False})

class V6_0812C(VersionVariant):
	def __init__(self, version=6, variant='0812C', params=None, **kwargs):
		VersionVariant.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'rd_cutoff': 0.4, 'diverse_answer_queries': False,
			'dynamic_grade_weights': True})

class V6_0812D(V6_0812B):
	# def __init__(self, params=None):
	# 	VersionVariant.__init__(self, version=6, variant='0812D', params=params)
	# 	V6_0812B.__init__(self)
	def __init__(self, version=6, variant='0812D', params=None, **kwargs):
		V6_0812B.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': True})

class V6_0812E(VersionVariant):
	def __init__(self, version=6, variant='0812E', params=None, **kwargs):
		VersionVariant.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'rd_cutoff': 0.4, 'diverse_answer_queries': True,
			'dynamic_grade_weights': True})

class V6_0812F(VersionVariant):
	def __init__(self, version=6, variant='0812F', params=None, **kwargs):
		VersionVariant.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'rd_cutoff': 0.4, 'grade_assignment_method': 'parent',
			'delta_link_threshold_factor': 1.3, 'diverse_answer_queries': False,
			'dynamic_grade_weights': False})

class V6_0812G(V6_0812F):
	def __init__(self, version=6, variant='0812G', params=None, **kwargs):
		V6_0812F.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'grade_assignment_method': 'parent_breaks',
			'delta_link_threshold_factor': 1.1})

class V6_0812H(V6_0812G):
	def __init(self, version=6, variant='0812H', params=None, **kwargs):
		V6_0812H.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'delta_link_threshold_factor': 1.5})

class V6_0825A(VersionVariant):
	def __init__(self, version=6, variant='0825A', params=None, **kwargs):
		VersionVariant.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'grade_assignment_method': 'parent_breaks',
			'delta_link_threshold_factor': 1., 'rd_cutoff': None, 'rd_deriving_factor': 0.5,
			'diverse_answer_queries': True, 'diverse_answer_queries_all': False,
			'dynamic_grade_weights': True})

class V6_0825B(VersionVariant):
	def __init__(self, version=6, variant='0825B', params=None, **kwargs):
		VersionVariant.__init(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'rd_cutoff': 0.4, 'diverse_answer_queries': True,
			'diverse_answer_queries_all': False, 'dynamic_grade_weights': True})

class V6_0825B2(V6_0825B):
	def __init__(self, version=6, variant='0825B2', params=None, **kwargs):
		V6_0825B.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'diverse_answer_queries_outlier_counts': True})

class V6_0825C(V6_0825A):
	def __init__(self, version=6, variant='0825C', params=None, **kwargs):
		V6_0825A.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'rd_cutoff': None, 'rd_deriving_factor': 0.5,
			'delta_link_threshold_factor': 1.3})

class V6_0825D(V6_0825C):
	def __init__(self, version=6, variant='0825D', params=None, **kwargs):
		V6_0825C.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'delta_link_threshold_factor': None, 'rd_deriving_factor': 0.5,
			'weighted_global_entropy': False, 'weighted_label_assignment': False})

class V6_0825D1(V6_0825D):
	def __init__(self, version=6, variant='0825D1', params=None, **kwargs):
		V6_0825D.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_use_normalized_density': True})

class V6_0825D2(V6_0825D):
	def __init__(self, version=6, variant='0825D2', params=None, **kwargs):
		V6_0825D.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_use_normalized_density': False})

class V6_0825D1_eqw(V6_0825D1):
	def __init__(self, version=6, variant='0825D1_eqw', params=None, **kwargs):
		V6_0825D1.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': False, 'possible_grades_weights': None})

class V6_0825D2_eqw(V6_0825D2):
	def __init__(self, version=6, variant='0825D2_eqw', params=None, **kwargs):
		V6_0825D2.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': False, 'possible_grades_weights': None})

class V6_0825E1(V6_0825D1):
	def __init__(self, version=6, variant='0825E1', params=None, **kwargs):
		V6_0825D1.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'diverse_answer_queries_all': True})

class V6_0825E2(V6_0825D2):
	def __init__(self, version=6, variant='0825E2', params=None, **kwargs):
		V6_0825D2.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'diverse_answer_queries_all': True})

class V6_0825F1(V6_0825D1_eqw):
	def __init__(self, version=6, variant='0825F1', params=None, **kwargs):
		V6_0825D1_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'voting_version': 'weighted_ig_average'})

class V6_0825G1(V6_0825D2_eqw):
	def __init__(self, version=6, variant='0825G1', params=None, **kwargs):
		V6_0825D1_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'voting_version': 'majority'})

class V6_0903A(V6_0825D):
	def __init__(self, version=6, variant='0903A', params=None, **kwargs):
		V6_0825D.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_density_adjustment': 'k'})

class V6_0903B(V6_0903A):
	def __init__(self, version=6, variant='0903B', params=None, **kwargs):
		V6_0903A.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_density_adjustment': 'k_den_0'})

class V6_0903C(V6_0903A):
	def __init__(self, version=6, variant='0903C', params=None, **kwargs):
		V6_0903A.__init(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_density_adjustment': 'k_den'})

class V6_0903C_eqw(V6_0903C):
	def __init(self, version=6, variant='0903C_eqw', params=None, **kwargs):
		V6_0903C_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': False, 'possible_grades_weights': None})

class V6_0915A(V6_0825D):
	def __init__(self, version=6, variant='0915A', params=None, **kwargs):
		V6_0825D.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': False, 'weighted_global_entropy': True,
			'weighted_label_assignment': True})

class V6_0915A1(V6_0915A):
	def __init__(self, version=6, variant='0915A1', params=None, **kwargs):
		V6_0915A.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_use_normalized_density': True})

class V6_0915A2(V6_0915A):
	def __init__(self, version=6, variant='0915A2', params=None, **kwargs):
		V6_0915A.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_use_normalized_density': False})

class V6_0915A1_eqw(V6_0825D1_eqw):
	def __init__(self, version=6, variant='0915A1_eqw', params=None, **kwargs):
		V6_0825D1_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': False, 'weighted_global_entropy': False,
			'weighted_label_assignment': False})

class V6_0915A2_eqw(V6_0825D2_eqw):
	def __init__(self, version=6, variant='0915A2_eqw', params=None, **kwargs):
		V6_0825D1_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': False, 'weighted_global_entropy': False,
			'weighted_label_assignment': False})

class V6_0915B(V6_0825D):
	def __init__(self, version=6, variant='0915B', params=None, **kwargs):
		V6_0825D.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': False, 'possible_grades_weights': [1.2, 0.6, 1.2],
			'weighted_global_entropy': True, 'weighted_label_assignment': True})

class V6_0915B1(V6_0915B):
	def __init__(self, version=6, variant='0915B1', params=None, **kwargs):
		V6_0915B.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_use_normalized_density': True})

class V6_0915B2(V6_0915B):
	def __init__(self, version=6, variant='0915B2', params=None, **kwargs):
		V6_0915B.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_use_normalized_density': False})

class V6_0915C(V6_0825D):
	def __init__(self, version=6, variant='0915C', params=None, **kwargs):
		V6_0825D.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': False, 'possible_grades_weights': [0.75, 1.5, 0.75],
			'weighted_global_entropy': True, 'weighted_label_assignment': True})

class V6_0915C1(V6_0915C):
	def __init__(self, version=6, variant='0915C1', params=None, **kwargs):
		V6_0915C.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_use_normalized_density': True})

class V6_0915C2(V6_0915C):
	def __init__(self, version=6, variant='0915C2', params=None, **kwargs):
		V6_0915C.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'pgv_use_normalized_density': False})

class V6_0915D1(V6_0825D1_eqw):
	def __init__(self, version=6, variant='0915D1', params=None, **kwargs):
		V6_0825D.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'dynamic_grade_weights': False, 'weighted_global_entropy': False,
			'weighted_label_assignment': True, 'voting_version': 'weighted_ig_average'})

class V6_0915E1(V6_0915D1):
	def __init__(self, version=6, variant='0915E1', params=None, **kwargs):
		V6_0915D1.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'voting_version': 'majority'})

class V6_0915F1(V6_0915D1):
	def __init__(self, version=6, variant='0915F1', params=None, **kwargs):
		V6_0915D1.__init__(self, version=version, variant=variant, params=params, **kwargs)
		self.params.update({'voting_version': 'majority_ordinal'})

class V6_0924A(V6_0825D1_eqw):
	def __init__(self, version=6, variant='0924A', params=None, **kwargs):
		V6_0825D1_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		subspace_param_list = kwargs.get('subspace_param_list', None)
		if subspace_param_list is None:
			raise MissingParamerror(version, variant, 'subspace_param_list')
		subspace_param_list[0]['compress_factor'] = 100
		subspace_param_list[0]['compress_method'] = 'pca'

class V6_0924B(VersionVariant):
	def __init__(self, version=6, variant='0924B', params=None, **kwargs):
		VersionVariant.__init__(self, version=version, variant=variant, params=params, **kwargs)
		subspace_param_list = kwargs.get('subspace_param_list', None)
		if subspace_param_list is None:
			raise MissingParamerror(version, variant, 'subspace_param_list')
		for subspace_params in subspace_param_list:
			if 'random_dimension' in subspace_params:
				subspace_params['random_selection_method'] = 'horizontal'

class V6_0924B1(V6_0825D1_eqw, V6_0924B):
	def __init__(self, version=6, variant='0924B1', params=None, **kwargs):
		V6_0825D1_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		V6_0924B.__init__(self, version=version, variant=variant, params=params, **kwargs)

class V6_0924B2(V6_0825D2_eqw, V6_0924B):
	def __init__(self, version=6, variant='0924B2', params=None, **kwargs):
		V6_0825D2_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		V6_0924B.__init__(self, version=version, variant=variant, params=params, **kwargs)

class V6_0924C(VersionVariant):
	def __init__(self, version=6, variant='0924C', params=None, **kwargs):
		VersionVariant.__init__(self, version=version, variant=variant, params=params, **kwargs)
		subspace_param_list = kwargs.get('subspace_param_list', None)
		if subspace_param_list is None:
			raise MissingParamerror(version, variant, 'subspace_param_list')
		num_subspaces = len(subspace_param_list)
		half = int(num_subspaces / 2)
		for s in range(num_subspaces):
			subspace_params = subspace_param_list[s]
			if 'random_dimension' in subspace_params:
				subspace_params['random_selection_method'] = 'vertical' if s <= half else 'horizontal'

class V6_0924C1(V6_0825D1_eqw, V6_0924C):
	def __init__(self, version=6, variant='0924C1', params=None, **kwargs):
		V6_0825D1_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		V6_0924C.__init__(self, version=version, variant=variant, params=params, **kwargs)

class V6_0924C2(V6_0825D2_eqw, V6_0924C):
	def __init__(self, version=6, variant='0924C2', params=None, **kwargs):
		V6_0825D2_eqw.__init__(self, version=version, variant=variant, params=params, **kwargs)
		V6_0924C.__init__(self, version=version, variant=variant, params=params, **kwargs)
