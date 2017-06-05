import random
import math

'''This module aims to solve the problem of choosing a random composition.
Suppose we have a non negative integer n that we wish to represent as a sum of
k non negative integers, each between a and b inclusive. We want to pick a
representation among all ways to write n as such a sum.

By subtracting a from each composition part, this is equivalent to finding a
composition of n - k * a into k parts, each between 0 and b - a inclusive. We
can pass to the continuous version of the problem and normalize to turn this
into finding a composition of 1 into k real number parts, each between 0 and
(b-a)/(n-k*a).

The continuous problem has the geometric interpretation of finding a random
point on a subset of the unit simplex in k dimensions. This module attempts
to solve the problem described above through this geometric interpretation.

'''

# Given non negative integers n, k, a, b, returns a composition of n
# into k parts, each between a and b inclusive. The composition is
# selected uniformly among all such compositions.
def random_composition(n, k, a, b):
	if n < 0 or k < 0 or a < 0 or b < 0 or a > b:
		raise ValueError('Composition must be of non negative integer into \
			non negative number of parts with non negative min and max, with \
			min <= max')
	if n < k * a or n > k * b:
		raise ValueError('No composition exists with given minimum and maximum \
			part value range')
	if n == k * a:
		return [a] * k
	if n == k * b:
		return [b] * k

	# Solve the problem in the continuous case, then convert back by rounding and
	# check if it still satisfies the condition that sum(point) == n
	point = []
	while sum(point) != n:
		point = random_point_region(k, float(b-a)/(n-k*a))
		point = [int(round(x * (n-k*a) + a)) for x in point]
	return point

def l2(numbers):
	return math.sqrt(sum(map(lambda x: x**2, numbers)))

# Returns a random point on the sphere in n dimensions
def random_point_sphere(n):
	unnormalized = [random.gauss(0,1) for _ in range(n)]
	normalized = [x/l2(unnormalized) for x in unnormalized]
	return normalized

# Picks a random point on the hyperplane x1 + ... + xn = 1.
# Picks a random point on a sphere, then pulls it down to the hyperplane along
# the great circle that passes through that point and the north pole.
def random_point_hyperplane(n):
	p = random_point_sphere(n)
	sumCoords = sum(p)
	pulled_down = [sumCoords/(math.sqrt(n)*(sumCoords - math.sqrt(n))) - math.sqrt(n)*y/(sumCoords - math.sqrt(n)) for y in p]
	pulled_down_normalized = [x/l2(pulled_down) for x in pulled_down]
	return pulled_down_normalized

def initialize(n, max_val):
	X = [float(1)/n for _ in range(n)]
	return X

# Hit and run algorithm for finding a random point in the subset of the unit
# n-simplex where all coordinates are between 0 and max_val.
def random_point_region(n, max_val):
	X = initialize(n, max_val)
	for _ in range(200):
		direction = random_point_hyperplane(n)
		paramRange = []
		for i in range(n):
			coordRange = [-X[i]/direction[i], (max_val-X[i])/direction[i]]
			if coordRange[0] > coordRange[1]:
				coordRange[0], coordRange[1] = coordRange[1], coordRange[0]
			paramRange.append(coordRange)
		minRange = max([x[0] for x in paramRange])
		maxRange = min([x[1] for x in paramRange])
		if minRange > maxRange:
			continue
		randomRange = random.uniform(minRange, maxRange)
		X = [X[i] + randomRange * direction[i] for i in range(n)]
		total = sum(X)
		X = [X[i]/total for i in range(n)]

	return X