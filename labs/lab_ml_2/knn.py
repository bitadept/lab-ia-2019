import numpy as np
import pdb
import matplotlib.pyplot as plt
from skimage import io
import sklearn.metrics as sm

class Knn_classifier:
    
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
    
    def classify_image(self, test_image, num_neighbors = 3, metric = 'l2'): 
     
        if(metric == 'l2'):   
            distances = np.sqrt(np.sum((self.train_images - test_image) ** 2, axis = 1))
        elif(metric == 'l1'):
            distances = np.sum(abs(self.train_images - test_image), axis = 1)
        else:
            print('Error! Metric {} is not defined!'.format(metric))
        
        sort_index = np.argsort(distances)
        sort_index = sort_index[:num_neighbors]
        nearest_labels = self.train_labels[sort_index] 
        histc = np.bincount(nearest_labels)
        
        return np.argmax(histc)
          
    def classify_images(self, test_images, num_neighbors = 3, metric = 'l2'):
        num_test_images = test_images.shape[0] 
        predicted_labels = np.zeros((num_test_images), np.int)
        
        for i in range(num_test_images): 
            if(i % 50 == 0):
                print('processing {}%...'.format(i / num_test_images * 100))
            predicted_labels[i] = self.classify_image(test_images[i, :], num_neighbors = num_neighbors, metric = metric)
        
        return predicted_labels
        

def accuracy_score(y_true, y_pred):
    return (y_true == y_pred).mean()

def confusion_matrix_v1(y_true, y_pred): 
    num_classes = max(y_true.max(), y_pred.max()) + 1
    conf_matrix = np.zeros((num_classes, num_classes)) 
    
    for i in range(num_classes):
        for j in range(num_classes):
            # cij, numarul exemplelor din clasa i care au fost clasificata ca fiind in clasa j. 
            conf_matrix[i, j] = len(np.intersect1d(np.where(y_true == i)[0], np.where(y_pred == j)[0]))
    
    return conf_matrix
    
def confusion_matrix_v2(y_true, y_pred): 
    num_classes = max(y_true.max(), y_pred.max()) + 1
    conf_matrix = np.zeros((num_classes, num_classes)) 
    
    for i in range(len(y_true)): 
        conf_matrix[int(y_true[i]), int(y_pred[i])] +=1
    return conf_matrix
    
train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', 'int')

image = train_images[0, :] # prima imagine
image = np.reshape(image, (28, 28)) 
io.imshow((image).astype(np.uint8))  
io.show()
 

# 1, 2, 3
classifier = Knn_classifier(train_images, train_labels)

test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int') 
 
predicted_labels = classifier.classify_images(test_images, 3, metric = 'l2')
accuracy = accuracy_score(test_labels, predicted_labels)
print('the classifier accuracy using l2 distance and 3 neighbors is ', accuracy)
np.savetxt('predictii_3nn_l2_mnist.txt', predicted_labels) 

# 4
predicted_labels = np.loadtxt('predictii_3nn_l2_mnist.txt')
conf_matrix = confusion_matrix_v1(test_labels, predicted_labels)
print(conf_matrix)
conf_matrix = confusion_matrix_v2(test_labels, predicted_labels)
print(conf_matrix)  


# 5 a
max_num_neighbors = 10
num_neighbors = [i for i in range(1, max_num_neighbors, 2)]
accuracy = np.zeros((len(num_neighbors)))

for n in range(len(num_neighbors)): 
    predicted_labels = classifier.classify_images(test_images, num_neighbors = num_neighbors[n], metric = 'l2')
    accuracy[n] = accuracy_score(test_labels, predicted_labels)
    
np.savetxt('acuratete_l2.txt', accuracy) 
# Ploteaza punctele
plt.plot(num_neighbors, accuracy)

# Adauga etichete pentru fiecare axa
plt.xlabel('number of neighbors')
plt.ylabel('accuracy') 

# Afiseaza figura
plt.show()


# 5 b

max_num_neighbors = 10
num_neighbors = [i for i in range(1, max_num_neighbors, 2)]
accuracy = np.zeros((len(num_neighbors)))

for n in range(len(num_neighbors)): 
    predicted_labels = classifier.classify_images(test_images, num_neighbors = num_neighbors[n], metric = 'l1')
    accuracy[n] = accuracy_score(test_labels, predicted_labels)


acuratete_l2 = np.loadtxt('acuratete_l2.txt')    
np.savetxt('acuratete_l1.txt', accuracy) 
# Ploteaza punctele
plt.plot(num_neighbors, accuracy) 
plt.plot(num_neighbors, acuratete_l2)
plt.gca().legend(('L1','L2'))
# Adauga etichete pentru fiecare axa
plt.xlabel('number of neighbors')
plt.ylabel('accuracy') 

# Afiseaza figura
plt.show()
    