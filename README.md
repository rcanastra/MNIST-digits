MNIST Digits Sequence Generator
================

Usage
-----------

Given a sequence of digits, generates an image of those digits composed from the MNIST data. These images would be used to train classifiers and generative deep learning models.

Requires files train-images-idx3-ubyte.gz and train-labels-idx1-ubyte.gz from MNIST homepage (http://yann.lecun.com/exdb/mnist/) to be placed in the same directory as the modules. File names must not be changed. 

There are two ways to run the application. One way is to run the generate_mnist_sequence() method located in the MNISTSequence module. This returns a float32 numpy array with values ranging from 0 (black, foreground) to 1 (white, background). The other is to run the MNISTSequence.py script with command line arguments via python. This produces a png file in the current directory. In either case, the resulting image is 28 rows by a user specified number of columns.

Spacing is applied between digits according to user specified parameters.

Arguments for generate_mnist_sequence():
digits: the digits to be represented in the output image, in an iterable
spacing_range: 2-tuple of ints storing the minimum and maximum spacing between digits, counted in pixels of width.
image_width: the desired width of the output image

Output: float32 numpy array of size 28 by image_width

The spacing between digits is selected uniformly at random from among all possible spacing within spacing_range that results in the desired image_width. Note that the individual digits from the MNIST dataset all have width 28 pixels.

The currently used algorithm for generating spacing is not very efficient when there are many digits and when the max possible spacing is large, so the code may run a while in those cases.

Arguments for command line:
1. the sequence of digits to be generated (no white space between digits)
2. the minimum spacing between digits
3. the maximum spacing between digits
4. the image width

Output: png file in the current directory with file name the same as the sequence of digits.

The same comments from above apply.

Architecture
----------

MNISTDigits.py reads and processes the data from the MNIST files. The interface provided by MNISTDigits.py is used by MNISTSequence.py to generate an image of the digits. The module RandomComposition.py solves the problem of picking random spacing widths between the digits.

Technical Choices
-------------------

MNISTDigits.py loads the entire dataset into memory, as opposed to accessing data from the disk only when it is necessary, or processing the dataset in chunks. This is possible because of the small size (~50MB) of the dataset. If the dataset were larger, we would have to be more careful with memory usage. For example, we could reference images not by their actual data but by their index/location in the file.

Also note that more memory is used for testing. (Specifically, an immutable copy of the data is created for hashing purposes.)

Within MNISTSequence.py, attempts were made to minimize the number of references to specifics of the MNIST dataset, e.g., the width of each image, to mitigate issues if the MNIST dataset were to be swapped with something else. (E.g., constants for the height and width can be found at the top of the module.) Perhaps these details could completely abstracted away with some kind of OOP.


Possible Improvements
---------------------
Currently, the widths of the random spacing is generated by an algorithm that is most certainly poorly implemented. Replacing the RandomComposition.py module, which handles the random spacing, with a proper library, or rewriting it with more thought, would improve the spacing aspect of the application. This module is not tested.

There is a file called RandomCompositionGenerator.py that contains a different algorithm than RandomComposition.py, which seems to do a better job of picking random spacing. However this module is not tested either, and the algorithm that it uses is not very time efficient.

There are also some aspects of the testing that are lacking. For example, within the testing module, the spacing portions of an output image is not identifed (although this can be somehow integrated into the test for the correctness of the digits).