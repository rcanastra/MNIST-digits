import gzip
import numpy

'''Given a digit, allows selection of a random image representing that digit.
Requires files train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz, from
the MNIST homepage, to exist in the current directory.

Internally handles images in original scale (0 to 255) but public methods
return rescaled images.


From outside this module, please only run the public methods, indicated below.

'''

IMAGES_FILE_NAME = 'train-images-idx3-ubyte.gz'
LABELS_FILE_NAME = 'train-labels-idx1-ubyte.gz'

MNIST_PIXEL_MAX = 255
MNIST_IMAGE_HEIGHT = 28
MNIST_IMAGE_WIDTH = 28

MNIST_NUM_IMAGES = 60000
MNIST_IMAGES_HEADER_SIZE = 16
MNIST_LABELS_HEADER_SIZE = 8

has_initialized = False
loaded_images = None
loaded_labels = None
label_to_images = None
image_to_label = None

class InitializationException(Exception):
	pass

#-----------Private methods start-----------#

# MNIST image pixel values range from 0 (white) to 255 (black).
# Rescale the range to 1 (white) to 0 (black).
def rescale(image):
	return numpy.array((MNIST_PIXEL_MAX-image)/float(MNIST_PIXEL_MAX),dtype=numpy.float32)

def rescale_back(image):
	return numpy.rint(MNIST_PIXEL_MAX - MNIST_PIXEL_MAX*image).astype(int)

def load_images():
	images_file_uncompressed = gzip.open(IMAGES_FILE_NAME, 'rb')
	images_file_contents = images_file_uncompressed.read()
	images_file_bytearray = bytearray(images_file_contents)

	# remove header, see MNIST specs
	images_bytearray = images_file_bytearray[MNIST_IMAGES_HEADER_SIZE:]

	global loaded_images
	images_array_shape = (MNIST_NUM_IMAGES, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH)
	loaded_images = numpy.reshape(images_bytearray, images_array_shape)

def load_labels():
	labels_file_uncompressed = gzip.open(LABELS_FILE_NAME, 'rb')
	labels_file_contents = labels_file_uncompressed.read()
	labels_file_bytearray = bytearray(labels_file_contents)
	
	# remove header, see MNIST specs
	labels_bytearray = labels_file_bytearray[MNIST_LABELS_HEADER_SIZE:]

	global loaded_labels
	loaded_labels = numpy.reshape(labels_bytearray, (MNIST_NUM_IMAGES,))

# Creates a dict: label -> list of images. This allows fast selection of
# a random image representing a given digit.
def group_images_by_label():
	global label_to_images
	label_to_images = {}
	for image, label in zip(loaded_images, loaded_labels):
		group_image_with_label(image, label)

def group_image_with_label(image, label):
	global label_to_images
	if label in label_to_images:
		label_to_images[label].append(image)
	else:
		label_to_images[label] = [image]

def convert_image_hashable(image):
	return tuple(tuple(row) for row in image)

def map_images_to_label():
	global image_to_label
	image_to_label = {}
	for image, label in zip(loaded_images, loaded_labels):
		map_image_to_label(image, label)

def map_image_to_label(image, label):
	global image_to_label:
	hashable_image = convert_image_hashable(image)
	image_to_label[hashable_image] = label

#-----------Private methods end-----------#


#-----------Public methods start-----------#

# initialization requires processing the dataset, only run if necessary
def initialize_if_necessary():
	global has_initialized
	if not has_initialized:
		initialize()

def initialize():
	load_images()
	load_labels()
	group_images_by_label()
	map_images_to_label()

	global has_initialized
	has_initialized = True

def get_random_image(digit):
	global has_initialized
	if not has_initialized:
		raise InitializationException('Please run MNISTDigits.initialize() \
			or MNISTDigits.initialize_if_necessary() before calling \
			MNISTDigits.get_rescaled_random_image()')

	digit_images = label_to_images[digit]
	num_images = len(digit_images)
	random_index = numpy.random.randint(0,num_images)
	return rescale(digit_images[random_index])

def get_random_images(digits):
	num_digits = len(digits)
	images_array_shape = (num_digits, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH)
	images = numpy.empty((images_array_shape))
	for i, digit in enumerate(digits):
		images[i] = get_rescaled_random_image(digit)

	return images

def verify_image_is_digit(image, digit):
	hashable_image_rescaled_back = convert_image_hashable(rescaled_back(image))
	if hashable_image_rescaled_back in image_to_label:
		return image_to_label[hashable_image_rescaled_back] == digit
	return False

def get_label(image):
	hashable_image_rescaled_back = convert_image_hashable(rescaled_back(image))
	return image_to_label[hashable_image_rescaled_back]

#-----------Public methods end-----------#