import numpy as np
# using Nearest neighbour algorithm to predict by using x as training data and y as testing data
x = np.array([[1,2,3],
             [1,2,3],
             [3,5,6],
             [1,0,1]])
x_label = np.array([1,2,3,4])
numrowsx = x.shape[0]
# x is a 4X3 where each row is an observation. Training starts with remember all
y = np.array([[3,0,3],
             [2,1,3],
             [0,1,3]])
# y is a 3X3 where each row is an test observation. Prediction starts with calling each row to calculate the distance
# with Training set
numrowsy = y.shape[0]
# Get how many rows in testing data
distance = np.zeros(numrowsx,dtype=x.dtype)
# define the distance array which will have the same length of x's iteration.
y_label = np.zeros(numrowsy,dtype=x.dtype)
# define the prediction outcome which will have the same format of x's label
for i in range(numrowsy):
    mindistance = 0
    for k in range(numrowsx):
       distance[k]=np.sum(np.abs(x[k,:]-y[i,:]),axis=0)
       # calculate the distance on each row in y from each row in x
    mindistance = min(distance)
    # calculate the min distance of each iteration in y
    y_label[i] = x_label[np.argmin(distance)]
    # find the smallest distance in each iteration on y

print(y_label)
# print prediction on y's label

