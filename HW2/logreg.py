import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_size=0.0008
max_iters=1000

def main():

  # Load the training data
  logging.info("Loading data")
  
  # get data from loadData (the bottom of the function)
  # train = np.loadtxt("train_cancer.csv", delimiter=",")
  # test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  X_train, y_train, X_test = loadData()

  logging.info("\n---------------------------------------------------------------------------\n")

  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (No Bias Term)")
  
  # X_train is each vectors (2-d array)
  # y_train is vector that the last columns 
  w, losses = trainLogistic(X_train,y_train)
  y_pred_train = X_train@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  
  logging.info("\n---------------------------------------------------------------------------\n")

  X_train_bias = dummyAugment(X_train)
 
  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (Added Bias Term)")
  w, bias_losses = trainLogistic(X_train_bias,y_train)
  y_pred_train = X_train_bias@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))


  plt.figure(figsize=(16,9))
  plt.plot(range(len(losses)), losses, label="No Bias Term Added")
  plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
  plt.title("Logistic Regression Training Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Negative Log Likelihood")
  plt.legend()
  plt.show()

  logging.info("\n---------------------------------------------------------------------------\n")

  logging.info("Running cross-fold validation for bias case:")

  # Perform k-fold cross
  for k in [2,3,4]:
    cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
    logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,cv_acc*100, cv_std*100))

  ####################################################
  # Write the code to make your test submission here
  ####################################################
  
  # make dummy set
  # dummy set -> def prediction(X_test, w)
  X_bias_test = dummyAugment(X_test)
  prediction(X_bias_test,w) # should be same dememsions
  
  
  # raise Exception('Student error: You haven\'t implemented the code in main() to make test predictions.')

def prediction(X, w):
  #sigmoid function
  i = 0
  f = open("kaggle_submit.csv","w")
  f.write("id" + ',' + "type" + '\n')
  for row in X:
    probability_1 = (1.0 / (1.0+np.exp(-1 * np.dot( w.T,X[i] ) )))
    
    if probability_1 >= 0.6438:
      result = 1
      # print(i, result)
      f.write(str(i) + ',' + str(result) + '\n')

    else:
      result = 0
      # print(i, result)
      f.write(str(i) + ',' + str(result) + '\n')
    i += 1
  
  f.close()

    
    
  

def dummyAugment(X):
  dummy = np.ones( (X.shape[0],X.shape[1]+1) )
  i = 0
  for row in X:
    for j in range(len(row)):
      dummy[i][j+1] = row[j]
    i += 1
  return dummy


def calculateNegativeLogLikelihood(X,y,w):

  # X  = n x d # each row OriginData[0...]
  # y  = n x 1 # the last item on the row (OriginialData[0][-1])
  
  # print(w)

  # It should return the result of computing Eq.8

  #  J = np.sum(-y * x * theta.T) + np.sum(np.exp(x * theta.T))+ np.sum(np.log(y))
  
  # devide 
  total = 0
  # result_calculate = np.sum(-y * X * w.T) + np.sum(np.exp(X * w.T)) + np.sum(np.log(y))
  i = 0
  
  for item in y:
    if item == 1:
      # print(np.shape(X[i]))
      partial_sum = np.log( 1.0 / (1.0+np.exp(-1 * np.dot( w.T,X[i] ) ) ) )
    else:
      partial_sum = np.log(1.0 - ( 1.0 / (1.0 + np.exp(-1 * np.dot(w.T,X[i]) ) ) ) )
          
    total = total + partial_sum
    # total += partial_sum
    i += 1
  # print(total)
  return (-1 * total)
  
  
  
  # raise Exception('Student error: You haven\'t implemented the negative log likelihood calculation yet.')
 

def trainLogistic(X, y, max_iters=max_iters, step_size=step_size):
  # X is each vectors (2-d array)
  # y is vector that the last columns 
  
    # Initialize our weights with zeros
    w = np.zeros( (X.shape[1],1) ) # get shape of the input vectors (number of row)
    # (466, 8)
    # w = np.random.random_sample((8,1)) - 0.5 # get shape of the input vectors (number of row)
    # print(w)
    
    # w = np.array([[0.1],[0.2],[0.3],[0.4],[0.3],[0.2],[0.1],[0]])
    
    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X,y,w)]
    # print(losses)
    
    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):
    
        # Make a variable to store our gradient
        w_grad = np.zeros( (X.shape[1],1) )
        # print(w_grad)
        # print(np.shape(w_grad))

        
        # Compute the gradient over the dataset and store in w_grad
        j = 0
        total_yields = np.zeros((X.shape[1],1))
        # print("values ", total_yields)
        # total_yields = 0
        # print("this is shape of total_tields before in for: " , np.shape(total_yields))
        
        for y_i in y:
          # print(np.shape(X[j]))
          # print(np.shape(( 1 / (1 + np.exp(-1 * np.dot(w.T,X[i])))) - y_i ))
          # print("print: ", np.shape(X[j]))
          # print("print (X[j]): ", np.transpose([X[j]]))
          # print("print (y_i): ", y_i)
          yields = (((1.0 / (1.0 + np.exp(-1 * np.dot(w.T , X[j] )))) - y_i ) * np.transpose([X[j]]))
          j = 1 + j
          # print("this is shape of yields: ", np.shape(yields))
          
          total_yields = total_yields + yields
        
        # print("this is outside of for (yields): ", yields)
        # print("this is shape of total_yields: ", np.shape(total_yields))
        w_grad = np.array(total_yields)
        # print("this is shape of w_grad: ", np.shape(w_grad))
        
        # print("This is expected shape:::: ", (X.shape[1],1) )
     
        # This is here to make sure your gradient is the right shape
        assert(w_grad.shape == (X.shape[1],1))

        # Take the update step in gradient descent
        w = w - step_size*w_grad
        
        # Calculate the negative log-likelihood with the 
        # new weight vector and store it for plotting later
        
        
        losses.append(calculateNegativeLogLikelihood(X,y,w))
    return w, losses




##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))
  
  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):
    
    start = min(len(X),fold_size*j)
    end = min(len(X),fold_size*(j+1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]
    
    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    w, losses = trainLogistic(X_fold_train, y_fold_train)

    # print(w)

    acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("train_cancer.csv", delimiter=",")
  test = np.loadtxt("test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test
  
  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
