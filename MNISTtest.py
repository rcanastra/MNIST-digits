import unittest
import numpy
import MNISTSequence
import MNISTDigits
from PIL import Image
from sklearn.cluster import KMeans

class TestMNIST(unittest.TestCase):

	def setUp(self):
		MNISTDigits.initialize()

	# Tests for verify_parameters()
	def test_digits_is_string(self):
		self.assertRaises(TypeError, MNISTSequence.generate_mnist_sequence, '12',(1,2),57)

	def test_spacing_range_is_list(self):
		self.assertRaises(TypeError, MNISTSequence.generate_mnist_sequence, [1,2],[1,2],57)

	def test_spacing_range_singleton(self):
		self.assertRaises(TypeError, MNISTSequence.generate_mnist_sequence, [1,2],(1,),57)

	def test_spacing_range_3_tuple(self):
		self.assertRaises(TypeError, MNISTSequence.generate_mnist_sequence, [1,2],(1,2,3),57)

	def test_image_width_too_small(self):
		self.assertRaises(ValueError, MNISTSequence.generate_mnist_sequence, [1,2],(1,2),56)

	def test_image_width_too_big(self):
		self.assertRaises(ValueError, MNISTSequence.generate_mnist_sequence, [1,2],(1,2),59)

	def test_digit_is_negative(self):
		self.assertRaises(ValueError, MNISTSequence.generate_mnist_sequence, [-1,2],(1,2),57)

	def test_output_image_width_1_digit(self):
		output_image = MNISTSequence.generate_mnist_sequence([6],(1,2),28)
		self.assertEqual(output_image.shape[1], 28)

	def test_output_image_width_3_digits(self):
		output_image = MNISTSequence.generate_mnist_sequence([6,3,4],(1,2),87)
		self.assertEqual(output_image.shape[1], 87)

	def test_random_spacing(self):
		di = MNISTSequence.DigitsImage([5,6,7],(1,4),89)
		di.generate_random_spacing()
		for space in di.spacing:
			self.assertTrue(space >= 1 and space <= 4)
		self.assertEqual(sum(di.spacing), 89 - 28*3)

	def test_output_image_spacing(self):
		pass

	def test_output_image_correctness(self):
		digits = [6,3,4,5,6,7,8,8]
		num_digits = len(digits)
		output_image = MNISTSequence.generate_mnist_sequence(digits,(1,6),32*num_digits)
		output_image_values = identify_digits(output_image, num_digits)
		print(output_image_values)
		num_digits_correct = sum([1 for i in range(num_digits) if output_image_values[i] == digits[i]])
		proportion_digits_correct = float(num_digits_correct)/num_digits
		self.assertTrue(proportion_digits_correct > 0.9)	

def check_image(output_image, min_spacing, max_spacing, expected_digits):
	num_digits = len(digits)
	image_width = output_image.shape[1]
	start_column = 0
	first_digit = output_image[0:MNISTDigits.MNIST_IMAGE_HEIGHT, 0,MNISTDigits.MNIST_IMAGE_WIDTH]
	digits = []
	digits.append(MNISTDigits.get_label(first_digit))
	while len(digits) < num_digits:
		curr_column = 0
		

# checks if image is all white colored (pixel value 1)
def is_all_background(image):
	return numpy.count_nonzero(image != 1.0) == 0

# Returns a view of the image with all black columns removed.
def trim_sides(image):
	left_column = 0
	right_column = image.shape[1] - 1
	while left_column < image.shape[1] and is_all_background(image[:, left_column]):
		left_column += 1
	while right_column > 0 and is_all_background(image[:, right_column]):
		right_column -= 1

	if left_column <= right_column:
		return image[:, left_column:right_column+1]
	else:
		return image

def group_trimmed_images_by_value():
	trimmed_image_to_value = {}
	untrimmed_images = MNISTDigits.get_images_copy()
	labels = MNISTDigits.get_labels_copy()
	for i, image in enumerate(untrimmed_images):
		trimmed_image = trim_sides(image)
		trimmed_image_hashable = tuple(tuple(x) for x in trimmed_image)
		trimmed_image_to_value[trimmed_image_hashable] = labels[i]

	return trimmed_image_to_value

def identify_digits(output_image, num_digits):
	individual_digit_images = approx_split_into_digits(output_image, num_digits)
	trimmed_image_to_value = group_trimmed_images_by_value()
	digit_values = []
	for image in individual_digit_images:
		trimmed_image = trim_sides(image)
		trimmed_image_hashable = tuple(tuple(x) for x in trimmed_image)
		if trimmed_image_hashable in trimmed_image_to_value:
			digit_values.append(trimmed_image_to_value[trimmed_image_hashable])
		else:
			digit_values.append(-1)

	return digit_values

# Applies k-means to find clusters of digits and splits image at midpoint
# of cluster centers.
def approx_split_into_digits(output_image, num_digits):
	image_height = output_image.shape[0]
	image_width = output_image.shape[1]

	my_kmeans = KMeans(num_digits)
	dark_pixels = [[y,x] for y in range(image_height) for x in range(image_width) if output_image[y,x] < 0.4]
	my_kmeans.fit(dark_pixels)

	# sort centers from left to right
	centers = sorted(my_kmeans.cluster_centers_, key= lambda x: x[1])
	centers_x = numpy.array([a[1] for a in centers])
	'''
	asdf = output_image
	for center in centers:
		for y in range(-3,3):
			for x in range(-3,3):
				asdf[int(center[0])+y,int(center[1])+x] = 0
	im = Image.fromarray(asdf*255)
	im.show()
	'''
	split_points = [0]
	num_spaces = num_digits - 1
	for i in range(num_spaces):
		split_points.append(int((centers_x[i]+centers_x[i+1])/float(2)))
	split_points.append(image_width)

	digits_list = []
	for i in range(len(split_points)-1):
		digit = output_image[:, split_points[i]:split_points[i+1]+1]
		digits_list.append(digit)

	return digits_list

if __name__ == '__main__':
	unittest.main()