import random

def memoize(func):
	memoized = {}

	def memoized_func(*args):
		if args in memoized:
			return memoized[args]
		else:
			value = func(*args)
			memoized[args] = value
			return value

	return memoized_func

def choose_with_probabilities(choices, probabilities):
	if len(choices) != len(probabilities) or abs(sum(probabilities)-1) > 0.001:
		return None

	cdf = []
	cdf.append(probabilities[0])
	for i in range(1, len(probabilities)):
		cdf.append(cdf[i-1] + probabilities[i])

	cutoff = random.uniform(0,1)

	i = 0
	while cutoff > 0:
		cutoff -= cdf[i]
		i += 1

	return choices[i-1]


class RandomCompositionGenerator(object):

	def __init__(self, term_min, term_max):
		self.term_min = term_min
		self.term_max = term_max

	@memoize
	def __compute_num_compositions(self, n, k):
		if n < 0 or k < 0:
			return 0
		if n == 0 or k == 0:
			return 1 if (n == 0 and k == 0) else 0

		if n < k * self.term_min:
			return 0
		if n > k * self.term_max:
			return 0

		return sum([self.__compute_num_compositions(n-i,k-1) for i in range(self.term_min, self.term_max+1)])

	def __fill_num_compositions(self, n, k):
		for i in range(n+1):
			for j in range(k):
				self.__compute_num_compositions(i,j)

	def random_composition(self, n, k):
		self.__fill_num_compositions(n,k)
		return self.random_composition_recur(n, k)

	def random_composition_recur(self, n, k):
		if n < 0 or k < 0:
			return None
		if n == 0 or k == 0:
			return [] if (n == 0 and k == 0) else None

		firstTermProbabilities = [0]*(self.term_max - self.term_min + 1)
		total = self.__compute_num_compositions(n,k)
		for i in range(self.term_min, self.term_max+1):
			firstTermProbabilities[i-self.term_min] = float(self.__compute_num_compositions(n-i,k-1))/total

		firstTerm = choose_with_probabilities(range(self.term_min, self.term_max+1), firstTermProbabilities)
		return [firstTerm] + self.random_composition_recur(n-firstTerm, k-1)

a=RandomCompositionGenerator(1,1000)
print(sum(a.random_composition(100000,100)))