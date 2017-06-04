import sys
import numpy
from PIL import Image
import MNISTDigits
from RandomComposition import random_composition

DIGITS_IMAGE_HEIGHT = MNISTDigits.MNIST_IMAGE_HEIGHT
DIGITS_IMAGE_WIDTH = MNISTDigits.MNIST_IMAGE_WIDTH

class MNISTError(Exception):
	pass

def verify_parameters(digits, spacing_range, image_width):
	verify_parameters_type(digits, spacing_range, image_width)
	verify_parameters_value(digits, spacing_range, image_width)

def verify_parameters_type(digits, spacing_range, image_width):
	if not hasattr(digits, '__iter__'):
		raise MNISTError('digits must be an iterable of ints')
	if len(digits) == 0:
		raise MNISTError('digits must not be empty')
	for digit in digits:
		if not isinstance(digit, int):
			raise MNISTError('digits must be an iterable of ints')
	if not isinstance(spacing_range, tuple):
		raise MNISTError('spacing_range must be a 2-tuple of ints')
	if len(spacing_range) != 2:
		raise MNISTError('spacing_range must be a 2-tuple of ints')
	if not isinstance(spacing_range[0], int) or not isinstance(spacing_range[1], int):
		raise MNISTError('spacing_range must be a 2-tuple of ints')
	if not isinstance(image_width, int):
		raise MNISTError('image_width must be an int')

def verify_parameters_value(digits, spacing_range, image_width):
	for digit in digits:
		if digit < 0 or digit > 9:
			raise MNISTError('digits must consist of single digit ints')
	if spacing_range[0] < 0 or spacing_range[1] < 0:
		raise MNISTError('spacing_range must be a tuple of non negative ints')
	if spacing_range[0] > spacing_range[1]:
		raise MNISTError('spacing_range min cannot be greater than max')

	num_digits = len(digits)
	num_spaces = num_digits - 1
	min_spacing = spacing_range[0]
	max_spacing = spacing_range[1]
	min_image_width = DIGITS_IMAGE_WIDTH * num_digits + min_spacing * num_spaces
	max_image_width = DIGITS_IMAGE_WIDTH * num_digits + max_spacing * num_spaces
	
	if image_width < min_image_width or image_width > max_image_width:
		raise MNISTError('image_width must be within possible bounds given spacing_range \
			and individual digit image widths')

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

		# Pad spacing to be the same length as self.digit_images.
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

def main():
	try:
		digits = [int(i) for i in sys.argv[1]]
		min_spacing = int(sys.argv[2])
		max_spacing = int(sys.argv[3])
		image_width = int(sys.argv[4])
	except:
		print('Please pass in the following arguments: \n'
			'1. the sequence of digits to be generated\n'
			'2. the minimum spacing between digits\n'
			'3. the maximum spacing between digits\n'
			'4. the image width\n'
			'e.g., python MNISTSequence.py 123 1 4 88\n'
			'For more information, please see the readme.')
		sys.exit(-1)

	digits_image = generate_mnist_sequence(digits, (min_spacing, max_spacing), image_width)
	seq = Image.fromarray((digits_image*255).astype(numpy.uint8))
	seq.save(sys.argv[1] + '.png')


if __name__ == '__main__':
	main()