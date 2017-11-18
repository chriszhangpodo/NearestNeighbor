import numpy as np

class Nearestneighbor(object):

    def __init__(self, dimensionrow,dimensioncol):
        self.dimensionrow = dimensionrow
        self.dimensioncol = dimensioncol
    def train(self,data, target):
        self.data = data
        self.target = target

    def predict(self,test,trainmodel):
        if (trainmodel == self.data).all():


            numrows_test = test.shape[0]
            numrows_train = self.data.shape[0]
            prediction = np.zeros(numrows_test,dtype=self.target.dtype)
            distance = np.zeros(numrows_train,dtype=self.data.dtype)

            for i in range(numrows_test):

                for k in range(numrows_train):
                    distance[k] = np.sum(np.abs(self.data[k,:]-test[i,:]),axis=0)
                mindistance = min(distance)
                prediction[i] = self.target[np.argmin(distance)]

            return [prediction,mindistance]

        raise RuntimeError('using existing model or create one before predict!')



x = np.array([[1,2,3],
             [1,2,3],
             [3,5,6],
             [1,0,1]])
x_label = np.array([1,2,3,4])
y = np.array([[3,0,3],
             [2,1,3],
             [0,1,3]])

x1 = np.array([[6,2,3],
             [9,3,2],
             [2,4,6],
             [1,0,1]])
x1_label = np.array([1,2,3,4])

nn1 = Nearestneighbor(4,3)
nn1.train(x,x_label)

nn2 = Nearestneighbor(4,3)
nn2.train(x1,x1_label)

pred1 = nn1.predict(y,nn1.data)

print(pred1[0])


