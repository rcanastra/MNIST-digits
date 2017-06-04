import unittest
import MNISTSequence
import MNISTDigits

class TestMNIST(unittest.TestCase):

	@classmethod
	def MNIST_initialize(cls):
		MNISTDigits.initialize_for_testing()

	# Test error handling

	def test_digits_is_string(self):
		self.assertRaises(MNISTSequence.MNISTError, MNISTSequence.generate_mnist_sequence, '12',(1,2),57)

	def test_spacing_range_is_list(self):
		self.assertRaises(MNISTSequence.MNISTError, MNISTSequence.generate_mnist_sequence, [1,2],[1,2],57)

	def test_spacing_range_1_tuple(self):
		self.assertRaises(MNISTSequence.MNISTError, MNISTSequence.generate_mnist_sequence, [1,2],(1,),57)

	def test_spacing_range_3_tuple(self):
		self.assertRaises(MNISTSequence.MNISTError, MNISTSequence.generate_mnist_sequence, [1,2],(1,2,3),57)

	def test_image_width_too_small(self):
		self.assertRaises(MNISTSequence.MNISTError, MNISTSequence.generate_mnist_sequence, [1,2],(1,2),56)

	def test_image_width_too_big(self):
		self.assertRaises(MNISTSequence.MNISTError, MNISTSequence.generate_mnist_sequence, [1,2],(1,2),59)

	def test_digit_is_negative(self):
		self.assertRaises(MNISTSequence.MNISTError, MNISTSequence.generate_mnist_sequence, [-1,2],(1,2),57)

	def test_digits_is_empty(self):
		digits = []
		self.assertRaises(MNISTSequence.MNISTError, MNISTSequence.generate_mnist_sequence, digits, (1,2), 0)

	# Test correctness of functions

	def test_output_image_width_1_digit(self):
		output_image = MNISTSequence.generate_mnist_sequence([6],(1,2),28)
		self.assertEqual(output_image.shape[1], 28)

	def test_output_image_width_3_digits(self):
		output_image = MNISTSequence.generate_mnist_sequence([6,3,4],(1,2),87)
		self.assertEqual(output_image.shape[1], 87)

	def test_spacing_within_range(self):
		di = MNISTSequence.DigitsImage([5,6,7],(1,4),89)
		di.generate_random_spacing()
		for space in di.spacing:
			self.assertTrue(space >= 1 and space <= 4)
		self.assertEqual(sum(di.spacing), 89 - 28*3)

	# Identify lengths of spacing from output image and check, e.g., 
	# it lies within spacing range.
	def test_output_image_spacing(self):
		pass

	def test_output_image_correct(self):
		digits = [6,3,4,5,6,7,8,8]
		num_digits = len(digits)
		output_image = MNISTSequence.generate_mnist_sequence(digits,(1,6),32*num_digits)
		self.assertTrue(image_represents_digits(output_image, (1,6), digits))	

	def test_output_image_incorrect(self):
		digits = [6,3,4,5,6,7,8,7]
		wrong_digits = [6,3,4,5,6,7,8,1]
		num_digits = len(digits)
		output_image = MNISTSequence.generate_mnist_sequence(digits,(1,6),32*num_digits)
		self.assertFalse(image_represents_digits(output_image, (1,6), wrong_digits))	


def image_represents_digits(output_image, spacing_range, expected_digits):
	image_width = output_image.shape[1]
	num_digits = len(expected_digits)
	if image_width == 0 and num_digits == 0:
		return True
	if image_width < MNISTDigits.MNIST_IMAGE_WIDTH or num_digits == 0:
		return False

	# handle first digit so that afterwards digits and spacing come in pairs
	first_digit = output_image[0:MNISTDigits.MNIST_IMAGE_HEIGHT, 0:MNISTDigits.MNIST_IMAGE_WIDTH]
	if MNISTDigits.check_image_is_digit(first_digit, expected_digits[0]):
		first_digit_removed = output_image[:, MNISTDigits.MNIST_IMAGE_WIDTH:]
		return image_represents_first_digit_recur(first_digit_removed, spacing_range, expected_digits[1:])
	return False

def image_represents_first_digit_recur(output_image, spacing_range, expected_digits):
	num_digits = len(expected_digits)
	image_width = output_image.shape[1]
	min_spacing = spacing_range[0]
	max_spacing = spacing_range[1]

	if image_width == 0 and num_digits == 0:
		return True
	if image_width < min_spacing + MNISTDigits.MNIST_IMAGE_WIDTH or num_digits == 0:
		return False

	spacing = min_spacing
	while spacing <= max_spacing:
		digit_start = spacing
		digit_end = spacing + MNISTDigits.MNIST_IMAGE_WIDTH
		if digit_end > image_width:
			return False
		potential_digit = output_image[0:MNISTDigits.MNIST_IMAGE_HEIGHT, digit_start:digit_end]
		if MNISTDigits.check_image_is_digit(potential_digit, expected_digits[0]):
			first_digit_and_spacing_removed = output_image[:, spacing+MNISTDigits.MNIST_IMAGE_WIDTH:]
			return image_represents_first_digit_recur(first_digit_and_spacing_removed, spacing_range, expected_digits[1:])
		else:
			spacing += 1

	return False

if __name__ == '__main__':
	TestMNIST.MNIST_initialize()
	unittest.main()