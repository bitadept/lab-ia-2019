# https://fmi-unibuc-ia.github.io/ia/

cale = 'C:/Users/Student/Desktop/lab_ml_1/data/'

from skimage import io # pentru afisarea imaginii
import numpy as np
import matplotlib.pyplot as plt

train_images = np.loadtxt(cale + 'train_images.txt') # incarcam imaginile
train_labels = np.loadtxt(cale + 'train_labels.txt', 'int')  # incarcam etichetele avand tipul de date int
test_images = np.loadtxt(cale + 'test_images.txt') # incarcam imaginile
test_labels = np.loadtxt(cale + 'test_labels.txt', 'int')  # incarcam etichetele avand tipul de date int

class Naive_Bayes:
    def __init__(self, num_bins, max_value):
        self.bins = np.linspace(start = 0, stop = max_value, num = num_bins)
        self.num_bins = num_bins
    def values_to_bins(self, matrice_imagini):
        return np.digitize(matrice_imagini, self.bins)
    def fit(self, train_images, train_labels):
        # aplicam histograma
        # train_images = self.values_to_bins(train_images)
        # calculam P(c) si P(x|c)
        # dimensiunea lui P(c) = 1 * num_classes
        # dimensiunea lui P(x|c) = num_features * num_bins * num_classes
        pc = np.zeros((np.unique(train_labels).shape[0]))
        print('toate clasele: \n', np.unique(train_labels))
        for class_val in np.unique(train_labels):
            pc[class_val] = sum(train_labels == class_val) / train_labels.shape[0]
        # p(x|c)
        pxc = np.zeros((train_images.shape[1], self.num_bins, np.unique(train_labels).shape[0]))
        for i in range(train_images.shape[1]):
            for class_val in np.unique(train_labels):
                imgs_in_class_val = train_images[train_labels == class_val, :]
                # for k in range(16):
                #     plt.subplot(4, 4, k + 1)
                #     image = imgs_in_class_val[k, :]
                #     image = np.reshape(image, (28, 28))
                #     io.imshow(image.astype(np.uint8))
                # io.show()

                for j in range(self.num_bins):
                    nummar_bins_pe_feature = sum(imgs_in_class_val[:, i] == j)
                    pxc[i, j, class_val] = nummar_bins_pe_feature / imgs_in_class_val.shape[0]
        self.pc = pc
        self.pxc = pxc
        return pc, pxc

ob = Naive_Bayes(5, 255)
pc, pxc = ob.fit(train_images, train_labels)

print(pc)
print(sum(pc))
print(pxc)

# image = train_images[11, :] # prima imagine
# image = np.reshape(image, (28, 28))
#
# x = image
# num_bins = 5
# bins = np.linspace(start = 0, stop = 255, num = num_bins)
# x_to_bins = np.digitize(x, bins)
#
# print(x_to_bins)

# for i in range(16):
#     image = train_images[i, :] # prima imagine
#     image = np.reshape(image, (28, 28))
#     plt.subplot(4, 4, i + 1)
#     io.imshow(image.astype(np.uint8))
# io.show()
#
# etichete = []
# for i in range(16):
#     etichete.append(train_labels[i])
#
# for i in range(16):
#     if i % 4 == 0:
#         print()
#     print(etichete[i], end = " ")