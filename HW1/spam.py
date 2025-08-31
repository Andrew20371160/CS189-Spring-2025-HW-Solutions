import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm

#loading the data set
spam_data = np.load("../data/spam-data.npz")
#80% for training and 20% for validation
num= int(0.8*spam_data["training_data"].shape[0])
#generate a sequence then shuffle them 
rand_seq = np.arange(spam_data["training_data"].shape[0])
np.random.shuffle(rand_seq)

#pick first shuffled 80% of data for training and rest for validation
spam_training_data = np.array(spam_data["training_data"][rand_seq[:num]])
spam_validation_data = np.array(spam_data["training_data"][rand_seq[num:]])

#pick first corresponding 80% of labels for training and rest for validation
spam_training_labels = spam_data["training_labels"][rand_seq[:num]]
spam_validation_labels = spam_data["training_labels"][rand_seq[num:]]


#traing and measuring accuracy against different sizes of training sets
numbers = [100,200,500,1000,2000,spam_training_data.shape[0]]

valditation_vec  =[]
training_vec  =[]

def accuracy(v1,v2):
	ret_score =0
	for i in range(v1.shape[0]):
		if v1[i]==v2[i]:
			ret_score +=1  
	return ret_score/v1.shape[0]


for number in numbers:
	#take first portion of the training and their corresponding labels
	current_spam_data = np.array(spam_training_data[:number])
	current_spam_labels = np.array(spam_training_labels[:number])
	
	spam_svm = svm.LinearSVC()
	spam_svm.fit(current_spam_data,current_spam_labels)
	#validation accuracy 
	p_vec = spam_svm.predict(spam_validation_data)
	valditation_vec.append(accuracy(p_vec,spam_validation_labels))
	#training accuracy 
	p_vec = spam_svm.predict(current_spam_data)
	training_vec.append(accuracy(p_vec,current_spam_labels))


plt.plot(numbers,training_vec)

plt.plot(numbers,valditation_vec)
plt.show()	





#k-cross validation 
spam_training_data  = np.array(spam_data["training_data"][rand_seq[:]])
spam_training_labels = np.array(spam_data["training_labels"][rand_seq[:]])

#data is divided into 5 equal sections
numbers = []
for i in range(5):
	numbers.append(int((i/5)*spam_data["training_data"].shape[0]))

#different values of C for each iteration
values =[10000,1000,500,10,0.1]

valditation_vec = []
for i in range(4):
	#each time ith section of training data is used as validation set
	#and the rest is used for training 
	current_spam_data=np.array(spam_training_data[numbers[(i+1)%5]:])
	current_spam_labels=np.array(spam_training_labels[numbers[(i+1)%5]:])
	if numbers[i]>0:
		#only at 2nd iteration we start concatinating what's before current validation set
		current_spam_data= np.concatenate((spam_training_data[:numbers[i]],current_spam_data))
		current_spam_labels= np.concatenate((spam_training_labels[:numbers[i]],current_spam_labels))

	current_spam_validation_data = np.array(spam_training_data[numbers[i]:numbers[i+1]])
	current_spam_validation_labels = np.array(spam_training_labels[numbers[i]:numbers[i+1]])

	for value in values:
		#for each iteration there are 5 c trials 
		spam_svm = svm.LinearSVC(C=value)
		spam_svm.fit(current_spam_data,current_spam_labels)
		p_vec =spam_svm.predict(current_spam_validation_data)
		valditation_vec.append(accuracy(p_vec,current_spam_validation_labels))


print(values)
print(valditation_vec)

avg_accuracy= 0
for i in range(20):
	avg_accuracy+=valditation_vec[i]


print(avg_accuracy/20)

	

