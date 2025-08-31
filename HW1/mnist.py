import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm

#load the dataset
mnist_data = np.load("../data/mnist-data.npz")
#get a range from 0 to 59999
rand_seq = np.arange(mnist_data["training_data"].shape[0])
#shuffle the sequence of indices
np.random.shuffle(rand_seq)

#collect first 50000 random elements as training data and last 10000 random element as validation data
#flatten the 2d arrays to be able to use linear svm
mnist_training_data = np.reshape(mnist_data["training_data"][rand_seq[0:50000]],(50000,28*28))
mnist_validation_data = np.reshape(mnist_data["training_data"][rand_seq[50000:]],(10000,28*28))

#collect first 50000 corresponding labels for training labels and the rest for validation 
mnist_training_labels = mnist_data["training_labels"][rand_seq[:50000]]
mnist_validation_labels = mnist_data["training_labels"][rand_seq[50000:]]


#training with different number of elements
numbers = [100,200,500,1000,2000,5000,10000]

valditation_vec  =[]
training_vec  =[]
#calculates the training/validation accuracy
def accuracy(v1,v2):
	ret_score =0
	for i in range(v1.shape[0]):
		if v1[i]==v2[i]:
			ret_score +=1  
	return ret_score/v1.shape[0]

#note that training accuracy is 1 doesn't mean it's overfitting since mnist data is linearly separable 
#hence there is a line that classifies all training data correctly
for number in numbers:
	
	current_mnist_data = np.array(mnist_training_data[:number])
	current_mnist_labels = np.array(mnist_training_labels[:number])	
	mnist_svm = svm.LinearSVC(max_iter =20000)
	mnist_svm.fit(current_mnist_data,current_mnist_labels)
	
	p_vec = mnist_svm.predict(mnist_validation_data)
	valditation_vec.append(accuracy(p_vec,mnist_validation_labels))

	p_vec = mnist_svm.predict(current_mnist_data)
	training_vec.append(accuracy(p_vec,current_mnist_labels))


#training accuracy vs corresponding training size 
plt.plot(numbers,training_vec)
#validation accuracy vs corresponding training size 
plt.plot(numbers,valditation_vec)
plt.show()	



#with hyper parameter tunning
#as C shrinks the validation accuracy gets better and better
values = [10,1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001]
#here the training is done on 10000 elements (same training set) but with different values of hyperparameter C
current_mnist_data = np.array(mnist_training_data[:1000])
current_mnist_labels = np.array(mnist_training_labels[:1000])
valditation_vec = []
for value in values:	
	mnist_svm = svm.LinearSVC(max_iter =10000,C=value)
	mnist_svm.fit(current_mnist_data,current_mnist_labels)
	p_vec = mnist_svm.predict(mnist_validation_data)
	valditation_vec.append(accuracy(p_vec,mnist_validation_labels))

print(values)
print(valditation_vec)
