from abc import ABC, ABCMeta, abstractmethod

class Solution(ABC):
	def __init__(self, data, encoding):
		self.data = data
		self.encoding = encoding
		self.count_cluster = 0
		
		self.labels = []
		self.clusters = []

		self.objectives = []
		self.score = None

	def set_attributes(self, dictionary):
		for key, value in dictionary.items():
			setattr(self, key, value)

	def get_printable_params(self):
		return {}

	def update_variables(self):
		raise NotImplementedError('update_variables not implemented')

	def update_objectives(self, objective_types):
		self.objectives = []
		for o in objective_types:
			if o == ObjectiveType.SCORE:
				if isinstance(self.score, list):
					self.objectives.extend(self.score)
				else:
					self.objectives.append(self.score)
			elif objective_type == ObjectiveType.CLUSTER_NUM:
				self.objectives.append(self.count_cluster)
