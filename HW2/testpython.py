import numpy as np

def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
#   test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  # last will not count here.....
  # row in each vectors 
  # print(X_train)
  print(X_train)
  
  # X_train.shape[0] =-> column
  # X_train.shape[1] =-> row
  
  print(X_train.shape[0])
  print(X_train.shape[1])

  w = np.zeros( (X_train.shape[1],1) )
  # print(len(w))
  # print(w)
  y_train = train[:, -1]
  # the last of the column
  
#   print(y_train)
  
#   X_test = test
  
  return 0
  
#   return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


loadData()

