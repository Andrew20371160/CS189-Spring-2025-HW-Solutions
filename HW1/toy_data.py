import numpy as np
import matplotlib.pyplot as plt

#source: uc berkeley
def scatter_data(data):
	plt.scatter(y = data[:,1],x= data[:,0])


#source: uc berkeley
#Plot the decision boundary
def plot_decision_boundary(w, b):
	x = np.linspace(-5, 5, 100)
	y = -(w[0] * x + b) / w[1]
	plt.plot(x, y, 'k')
	

#the rest is my code
#set of x's in R2 such that w.x+b = -1 or 1 
def plot_margins(w, b):
	x = np.linspace(-5,5,1000)
	left_margin =[[],[]] 
	right_margin  = [[],[]]

	for i in range(x.shape[0]):
		for j in range(x.shape[0]):
			val  = x[i]*w[0] +x[j]*w[1] +b
			if abs(val)>=0.999 and abs(val)<=1.001 :
				if val<0:
					left_margin[0].append(x[i])
					left_margin[1].append(x[j])
				else:
					right_margin[0].append(x[i])
					right_margin[1].append(x[j])

	plt.plot(left_margin[0],left_margin[1])
	plt.plot(right_margin[0], right_margin[1])
	


#loading the dataset
toy_data = np.load(f"../data/toy-data.npz")



b= 0.1471
w=np.array([-0.4528,-0.5190])

scatter_data(toy_data["training_data"])
plot_decision_boundary(w,b)
plot_margins(w,b)

#indicate support vectors
for i in range(toy_data["training_data"].shape[0]):
	val = np.dot(toy_data["training_data"][i],w)+b 
	if abs(val)>=0.9 and abs(val)<=1.1 :
		plt.scatter(toy_data["training_data"][i,0],toy_data["training_data"][i,1])


plt.show()

