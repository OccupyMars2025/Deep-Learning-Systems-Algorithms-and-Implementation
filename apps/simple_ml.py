import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # get 2-D image array
    with gzip.open(image_filename, 'rb') as fp:
        header = struct.unpack('>4i', fp.read(16))
        magic, number_of_images, height, width = header

        if magic != 2051:
            raise RuntimeError("'%s' is not an MNIST image set." % image_filename)
        images = struct.unpack('>%dB' % (number_of_images * height * width), fp.read())
        images = np.array(images).reshape([number_of_images, height * width])
        images = np.array(images / 255, dtype=np.float32)

    # get 1-D label array
    with gzip.open(label_filename, 'rb') as fp:
        header = struct.unpack('>2i', fp.read(8))
        magic, number_of_images = header

        if magic != 2049:
            raise RuntimeError("'%s' is not an MNIST label set." % label_filename)

        labels = struct.unpack('>%dB' % number_of_images, fp.read())
        labels = np.array(labels, np.uint8)

    return images, labels



def softmax_loss(Z: ndl.Tensor, y_one_hot: ndl.Tensor) -> ndl.Tensor:
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    part1 = ndl.log(ndl.summation(ndl.exp(Z), axes=-1, keep_axes=False)) 
    part2 = ndl.summation(Z * y_one_hot, axes=-1, keep_axes=False)
    batch_size = Z.shape[0]
    return ndl.summation(part1 - part2, axes=None, keep_axes=False) / batch_size


def nn_epoch(X: np.ndarray, y: np.ndarray, W1: ndl.Tensor, W2: ndl.Tensor, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X @ W1) @ W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of shape
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of shape (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    
    for i in range(0, (num_examples // batch) * batch, batch):
        print(f"batch {i}/{(num_examples // batch) * batch}")
        logits = ndl.relu(ndl.Tensor(X[i: i+batch]) @ W1) @ W2 

        y_one_hot = np.zeros((batch, num_classes))
        y_one_hot[np.arange(batch), y[i: i+batch]] = 1

        average_batch_loss = softmax_loss(logits, ndl.Tensor(y_one_hot))

        average_batch_loss.backward()

        # (failure)Implementation 1: create new tensors for W1 and W2 without stop,
        # while the new tensors can refer to the old tensors for W1 and W2 with its "inputs" attribute.
        # So the useless old tensors for W1 and W2 cannot be garbage-collected.
        # So the size of the computational graph will explode.
        # So it will consume more time with each batch.
        # It took me about 3 hours to traverse half of the 60000 examples and 
        # my computer memory seemed to explode if using Implementation 1
        # W1 -= lr * W1.grad
        # W2 -= lr * W2.grad

        # Implementation 2: For each mini-batch, after calling .backward(), you should compute 
        # the updated values for W1 and W2 in numpy,
        #  and then create new Tensors for W1 and W2 with these numpy values.
        # It takes about 5 seconds if using Implementation 2
        W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())

    return W1, W2



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
