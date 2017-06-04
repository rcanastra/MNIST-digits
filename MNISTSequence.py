import numpy
import MNISTDigits
from RandomPoints import random_composition

DIGITS_IMAGE_HEIGHT = MNISTDigits.MNIST_IMAGE_HEIGHT
DIGITS_IMAGE_WIDTH = MNISTDigits.MNIST_IMAGE_WIDTH

def verify_parameters(digits, spacing_range, image_width):
	verify_parameters_type(digits, spacing_range, image_width)
	verify_parameters_value(digits, spacing_range, image_width)

def verify_parameters_type(digits, spacing_range, image_width):
	if not hasattr(digits, '__iter__'):
		raise TypeError('digits must be an iterable of ints')
	for digit in digits:
		if not isinstance(digit, int):
			raise TypeError('digits must be an iterable of ints')
	if not isinstance(spacing_range, tuple):
		raise TypeError('spacing_range must be a 2-tuple of ints')
	if len(spacing_range) != 2:
		raise TypeError('spacing_range must be a 2-tuple of ints')
	if not isinstance(spacing_range[0], int) or not isinstance(spacing_range[1], int):
		raise TypeError('spacing_range must be a tuple of ints')
	if not isinstance(image_width, int):
		raise TypeError('image_width must be an int')

def verify_parameters_value(digits, spacing_range, image_width):
	for digit in digits:
		if digit < 0 or digit > 9:
			raise ValueError('digits must consist of ints between 0 and 9 inclusive')
	if spacing_range[0] < 0 or spacing_range[1] < 0:
		raise ValueError('spacing_range must be a tuple of non negative ints')
	if spacing_range[0] > spacing_range[1]:
		raise ValueError('spacing_range min cannot be greater than max')

	num_digits = len(digits)
	num_spaces = num_digits - 1
	if image_width < DIGITS_IMAGE_WIDTH * num_digits + spacing_range[0] * num_spaces or \
			image_width > DIGITS_IMAGE_WIDTH * num_digits + spacing_range[1] * num_spaces:
		raise ValueError('image_width must be within possible bounds given spacing_range \
			and MNIST image width')

class DigitsImage(object):
	def __init__(self, digits, spacing_range, image_width):
		verify_parameters(digits, spacing_range, image_width)

		self.digits = digits
		self.min_spacing = spacing_range[0]
		self.max_spacing = spacing_range[1]
		self.image_width = image_width

		self.digit_images = None
		self.spacing = None

		image_shape = (DIGITS_IMAGE_HEIGHT, image_width)
		self.image = numpy.ones(image_shape, dtype=numpy.float32)

	def fill_image_with_random_digits(self):
		self.choose_random_digit_images()
		self.generate_random_spacing()

		# pad spacing to be the same length as self.digit_images
		self.spacing.append(0)
		curr_column = 0
		for i, digit_image in enumerate(self.digit_images):
			digit_image_width = digit_image.shape[1]
			self.fill_image_with_digit(digit_image, curr_column)
			curr_column += digit_image_width + self.spacing[i]

	def fill_image_with_digit(self, digit_image, start_column):
		digit_image_width = digit_image.shape[1]
		digit_image_start = start_column
		digit_image_end = start_column + digit_image_width
		self.image[0:DIGITS_IMAGE_HEIGHT, digit_image_start:digit_image_end] = digit_image

	def generate_random_spacing(self):
		num_spaces = len(self.digits) - 1
		self.spacing = random_composition(self.get_total_spacing_width(), num_spaces, self.min_spacing, self.max_spacing)

	def choose_random_digit_images(self):
		self.digit_images = MNISTDigits.get_random_images(self.digits)

	def get_total_spacing_width(self):
		num_digits = len(self.digits)
		return self.image_width - num_digits * DIGITS_IMAGE_WIDTH

	def get_image(self):
		return self.image

def generate_mnist_sequence(digits, spacing_range, image_width):
	MNISTDigits.initialize_if_necessary()
	
	digits_image = DigitsImage(digits, spacing_range, image_width)
	digits_image.fill_image_with_random_digits()

	return digits_image.get_image()

generate_mnist_sequence([1,2,3],(0,2),87)