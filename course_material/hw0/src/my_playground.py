# # import urllib.request
# # import os
# # import gzip
# # import struct
# # import numpy as np
# #
# #
# # class MNIST(object):
# #     def __init__(self, dataset_out_dir=".", base_url="http://yann.lecun.com/exdb/mnist/"):
# #         self.dataset_out_dir = dataset_out_dir
# #         train_images_file = "train-images-idx3-ubyte.gz"
# #         train_labels_file = "train-labels-idx1-ubyte.gz"
# #         test_images_file = "t10k-images-idx3-ubyte.gz"
# #         test_labels_file = "t10k-labels-idx1-ubyte.gz"
# #
# #         train_images_path = os.path.join(dataset_out_dir, train_images_file)
# #         train_labels_path = os.path.join(dataset_out_dir, train_labels_file)
# #         test_images_path = os.path.join(dataset_out_dir, test_images_file)
# #         test_labels_path = os.path.join(dataset_out_dir, test_labels_file)
# #
# #         self.download(base_url + train_images_file, train_images_path)
# #         self.download(base_url + test_images_file, test_images_path)
# #         self.download(base_url + train_labels_file, train_labels_path)
# #         self.download(base_url + test_labels_file, test_labels_path)
# #
# #         self.train_images = self.parse_images(train_images_path)
# #         self.train_labels = self.parse_labels(train_labels_path)
# #         self.test_images = self.parse_images(test_images_path)
# #         self.test_labels = self.parse_labels(test_labels_path)
# #
# #     def download(self, url, output):
# #         print("Downloading", url, "to", output, "...")
# #         os.makedirs(self.dataset_out_dir, exist_ok=True)
# #         if not os.path.exists(output):
# #             urllib.request.urlretrieve(url, output)
# #
# #     def parse_images(self, f):
# #         images = []
# #         with gzip.open(f, 'rb') as fp:
# #             header = struct.unpack('>4i', fp.read(16))
# #             # print(header)
# #             # exit()
# #             magic, size, height, width = header
# #
# #             if magic != 2051:
# #                 raise RuntimeError("'%s' is not an MNIST image set." % f)
# #
# #             chunk = width * height
# #             for _ in range(size):
# #                 img = struct.unpack('>%dB' % chunk, fp.read(chunk))
# #                 # print(img)
# #                 # print(type(img))
# #                 # print(len(img))
# #                 img_np = np.array(img, np.uint8)
# #                 # print(img_np)
# #                 # print(img_np.shape)
# #                 # exit()
# #                 images.append(img_np)
# #         return images
# #
# #     def parse_labels(self, f):
# #         with gzip.open(f, 'rb') as fp:
# #             header = struct.unpack('>2i', fp.read(8))
# #             magic, size = header
# #
# #             if magic != 2049:
# #                 raise RuntimeError("'%s' is not an MNIST label set." % f)
# #
# #             labels = struct.unpack('>%dB' % size, fp.read())
# #
# #         return np.array(labels, np.int32)
# #
# # def parse_mnist(image_filename, label_filename):
# #     """ Read an images and labels file in MNIST format.  See this page:
# #     http://yann.lecun.com/exdb/mnist/ for a description of the file format.
# #
# #     Args:
# #         image_filename (str): name of gzipped images file in MNIST format
# #         label_filename (str): name of gzipped labels file in MNIST format
# #
# #     Returns:
# #         Tuple (X,y):
# #             X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
# #                 data.  The dimensionality of the data should be
# #                 (num_examples x input_dim) where 'input_dim' is the full
# #                 dimension of the data, e.g., since MNIST images are 28x28, it
# #                 will be 784.  Values should be of type np.float32, and the data
# #                 should be normalized to have a minimum value of 0.0 and a
# #                 maximum value of 1.0. The normalization should be applied uniformly
# #                 across the whole dataset, _not_ individual images.
# #
# #             y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
# #                 labels of the examples.  Values should be of type np.uint8 and
# #                 for MNIST will contain the values 0-9.
# #     """
# #     with gzip.open(image_filename, 'rb') as fp:
# #         header = struct.unpack('>4i', fp.read(16))
# #         # print(header)
# #         # exit()
# #         magic, number_of_images, height, width = header
# #
# #         if magic != 2051:
# #             raise RuntimeError("'%s' is not an MNIST image set." % image_filename)
# #         images = struct.unpack('>%dB' % (number_of_images * height * width), fp.read())
# #         images = np.array(images).reshape([number_of_images, height * width])
# #         images = np.array(images / 255, dtype=np.float32)
# #     #     chunk = width * height
# #     #     for _ in range(size):
# #     #         img = struct.unpack('>%dB' % chunk, fp.read(chunk))
# #     #         # print(img)
# #     #         # print(type(img))
# #     #         # print(len(img))
# #     #         img_np = np.array(img, np.uint8)
# #     #         # print(img_np)
# #     #         # print(img_np.shape)
# #     #         # exit()
# #     #         images.append(img_np)
# #     # images = np.stack(images) / 255
# #
# #     with gzip.open(label_filename, 'rb') as fp:
# #         header = struct.unpack('>2i', fp.read(8))
# #         magic, number_of_images = header
# #
# #         if magic != 2049:
# #             raise RuntimeError("'%s' is not an MNIST label set." % label_filename)
# #
# #         labels = struct.unpack('>%dB' % number_of_images, fp.read())
# #
# #     labels = np.array(labels, np.int32)
# #     print(images)
# #     print(labels)
# #     print(f"images.shape: {images.shape} ")
# #     print("labels.shape: ", labels.shape)
# #     print(f"images.dtype: {images.dtype}")
# #     print(f"labels.dtype: {labels.dtype}")
# #     print(images.max(), images.min())
# #     print(labels.max(), labels.min())
# #
# # if __name__ == "__main__":
# #     # print(struct.calcsize("4i"))
# #     # import sys
# #     # print(sys.byteorder)
# #     # record = b'raymond   \x32\x12\x08\x01\x08'
# #     # name, serialnum, school, gradelevel = struct.unpack('<10sHHb', record)
# #     # # print(name, serialnum, school, gradelevel, sep='\n')
# #     # print(struct.unpack('<H', b'\x32\x12'))
# #     # print(int(0x12) * (2**8) + int(0x32))
# #     # print(struct.unpack('>H', b'\x32\x12'))
# #     # print(int(0x32) * (2**8) + int(0x12))
# #     # print(struct.pack('<qh6xq', 1, 2, 3))
# #     # print(struct.pack('<lhl', 1, 2, 3))
# #     # print(struct.pack('@lhl', 1, 2, 3))
# #     # print(struct.calcsize('<qqh6x'))
# #     # print(struct.calcsize('@llh0l'))
# #     # print(struct.pack('>llh0l', 1, 2, 3))
# #     # print(struct.pack('>qqh6x', 1, 2, 3))
# #     # exit()
# #     #
# #     # mnist = MNIST(dataset_out_dir="mnist")
# #     #
# #     # print("train dataset")
# #     # for label, perc in enumerate(np.histogram(mnist.train_labels, normed=True)[0]):
# #     #     print("%d: %.2f %%" % (label, 100 * perc))
# #     #
# #     # print("test dataset")
# #     # for label, perc in enumerate(np.histogram(mnist.test_labels, normed=True)[0]):
# #     #     print("%d: %.2f %%" % (label, 100 * perc))
# #     image_filename = r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw0\data\train-images-idx3-ubyte.gz"
# #     label_filename = r"C:\Users\Administrator\Desktop\Deep-Learning-Systems\source_code\hw0\data\train-labels-idx1-ubyte.gz"
# #     parse_mnist(image_filename=image_filename, label_filename=label_filename)
#
# import numpy as np
# Z = np.random.randn(4, 3)
# y = np.random.randint(0, 3, size=4)
# print(Z, y, Z[np.arange(4), y], sep='\n')
#
# batch_size = Z.shape[0]
# expZ = np.exp(Z)
# loss_for_each_sample = np.log(np.sum(expZ, axis=1)) - Z[np.arange(batch_size), y]
# total_loss = np.sum(loss_for_each_sample)
# print(expZ, loss_for_each_sample, total_loss, sep='\n')
#
# # return total_loss / batch_size

# num_examples = 100
# batch = 10
# for i in range(0, num_examples, batch):
#     print(i)

# N = -2
# def fun(n):
#     n += 10
#
# fun(N)
# print(N)

import numpy as np

N = np.zeros((2, 3))
def inc(n):
    n += 1

inc(N)
print(N)