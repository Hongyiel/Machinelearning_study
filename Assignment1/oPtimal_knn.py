from numpy import array
from numpy import linalg as LA

import math
import csv
import os
import numpy as np
import sys
import random

# 
def getRows(CSVPath):
    with open(CSVPath, newline='') as csvFile:
        rowReader = list(csv.reader(csvFile, delimiter=',', quotechar='|'))

        return rowReader



#Q5.
print("--------------------------------------------------")
print("kNN algorithm 4-folde cross validation:")
print("--------------------------------------------------")
#Explain how this might affect your model and how you interpret the results

# For instance, would you say a model that 
# achieved 70% accuracy is a good or poor model? 
#
# yes it will be good model achieved 70% accuracy
#
# How many dimensions does each data point have 
# (ignoring the id attribute and class label)?
#
# categorial (7)
# numerical (4)
# ordinal (1)
# Total 12 demensions ?
#

# # # range of workclass
# print(train_data[0][6])
# print(train_data[1][6:13])

# getWorkSpace_categorial = train_data[1][6:12]
# getMarried_categorial = train_data[1][13:20]
# getOccupation_categorial = train_data[1][21:33]
# getRelationship_categorial = train_data[1][34:40]

# print(train_data[0][6:12])
# print(train_data[0][13:20])
# print(train_data[0][21:33])
# print(train_data[0][34:40])

def one_hot_encoding(rawData,target_categorial):
    #get target length and find coding
    get_len = len(target_categorial)
    # print(get_len)
    g = 0
    result =0
    for i in target_categorial:
        #get each values for binary
        # 1 , 2 , 4 , 8 , 16 ...
        # 2^0, 2^1, 2^2, 2^3, 2^4 ...
        # check array is 1 or 0
        # if then add in result : 2^g

        if i == '1':
            result = result + (2)**int(g)
        # if not ignored
        g = g + 1
    return ((result-1)/(2**get_len-1))


# # range of Married
# print(train_data[0][14])
# print(train_data[0][21])
# print(train_data[1][14:21])

# # range of occupation
# print(train_data[0][22])
# print(train_data[0][33])
# print(train_data[1][22:33])

# # range of relationship
# print(train_data[0][34])
# print(train_data[0][39])
# print(train_data[1][34:39])


# get encoded value from here!

def get_vector_Generator(id,rawData,option):
    vector_set = []

    # Check data - should be delete

    getIdCheck = rawData[id][0]


    # Numerical

    getAge = rawData[id][1] 
    getEducation = rawData[id][2]
    getCapital_gain = rawData[id][3]
    getCapital_loss = rawData[id][4]
    getHoursPerWeek = rawData[id][5]

    # Categorial
    getWorkSpace_categorial= rawData[id][6:13]
    getMarried_categorial = rawData[id][13:20]
    getOccupation_categorial = rawData[id][20:34]
    getRelationship_categorial = rawData[id][34:40]


    # print("this is income data: " + str(getIncomeData))

    # encoded
    result_Work_encoded = one_hot_encoding(rawData,getWorkSpace_categorial)
    result_Married_encoded = one_hot_encoding(rawData,getMarried_categorial)
    result_Occupation_encoded = one_hot_encoding(rawData,getOccupation_categorial)
    result_Relationship_encoded = one_hot_encoding(rawData,getRelationship_categorial)

    vector_set.append(int(getIdCheck))

    # vector_set.append section
    vector_set.append(float(getAge))
    vector_set.append(float(getEducation))
    vector_set.append(float(getCapital_gain))
    vector_set.append(float(getCapital_loss))
    vector_set.append(float(getHoursPerWeek))

    # vectorset by encoded

    # temp = []

    vector_set.append(float(result_Work_encoded))
    vector_set.append(float(result_Married_encoded))
    vector_set.append(float(result_Occupation_encoded))
    vector_set.append(float(result_Relationship_encoded))
    
    # Normalization function
    # append normalzed values on vector_set 


    # for value in temp:
    #     vector_set.append((value - np.mean(temp)) / (max(temp) - min(temp)))
    # income
    if option == 1:
        getIncomeData = rawData[id][86]
        vector_set.append(int(getIncomeData))
    # print(vector_set)
    return vector_set


# def normalization(vector_set):
#     # x - mean / max - min
#     # vector_set[0] = work
#     # vector_set[1] = married
#     # vector_set[2] = occupation
#     # vector_set[3] = relationship

#     # normalization
#     normal_result = []
#     for value in vector_set:
#         normal_result.append(value - mean(vector_set)) / (max(vector_set) - min(vector_set))
        
#     # init vectorset again

#     return normal_result

# print(one_hot_encoding(rawData, getWorkSpace_categorial))

# get vector generation
def get_vector(rawData,option):
    i = 1
    vector = []
    for row in rawData[0:-1]:
        vector.append(get_vector_Generator(i,rawData,option))
        i = i + 1
    return vector


def kNN_decision(k, train_dataSet, validation_vector):
    sortDistanceList = []
    new_vector = []
    for each_vector in train_dataSet:
        i = 1
        distance = 0
        # get distance calculate between each_vector and ex_vector
        for i in range(1,10): # excluding id and income
            # 1 ~ 10 (which is 11 values  last will INCOME)
            distance += float((each_vector[i] - float(validation_vector[i]))**2)
        distance = math.sqrt(distance)
        # print(each_vector[10])

        # sortDistanceList.extend(distance)
        # sortDistanceList.extend(each_vector[0])
        # sortDistanceList.extend(each_vector[10])

        # new_vector.append(sortDistanceList)

        # ----------- distance,  id number    , income --------
        new_vector = [distance, each_vector[0], each_vector[10]]

        sortDistanceList.append(new_vector)
    sortDistanceList.sort()
    # counting of income references for voting
    k_result = 0
    k_income = 0
    for vote in range(k):
        if(int(sortDistanceList[vote][2]) == 1):
            k_income += 1
    # actual decision according k
    k_result = float(k_income/k)
    if (k_result > 0.5):
        return 1
    elif (k_result < 0.5):
        return 0
    else:
        return 1


# validation testing loop
def validation_loop(k, train_dataSet, validation_dataSet):
    kNN_temp = []
    kNN_result = []
    for validation_vector in validation_dataSet:
        decision = kNN_decision(k,train_dataSet,validation_vector)
        kNN_temp = [decision,validation_vector[10]]
        kNN_result.append(kNN_temp)
        
    err_score = 0
    total = 0
    i = 0

    # print(kNN_result)

    for row in kNN_result:
        # print(row)
        if ((row[0]+row[1]) == 1):
            err_score = err_score + 1

            # print("error_score :: " + str(err_score))
        i = i + 1
        total = total + 1

    # Data Accuracy
    return (int(100)*float(1-(float(err_score/total))))  

def k_decision_max(pre_k, pre_accuracy, cur_k, cur_accuracy):
    if(pre_accuracy < cur_accuracy):
        return cur_k
    else:
        return pre_k
############################
## - if argument 1 is '-v' then
option = sys.argv[1]
optimal_k = 1
if (option == "-v"):
    trainFile = sys.argv[2]
    train_data = getRows(trainFile)
    vector_train = get_vector(train_data,1)
    # vector_test  = get_vector(test_data,0)
    random.shuffle(vector_train)

    validating_vectorSet = []
    folded_vectorSet = []

    pre_k = 1
    pre_accuracy = 0
    max_k = 100
    for i in range(max_k): ## - it's for iterating of validation test with different fold
        k = 2*i+1
        numFold = 4
        nVector = int(8000/numFold)
        sum_accuracy = 0
        for curF in range(numFold): ## - it's for iterating of validation test with different fold
            validating_vectorSet = []
            folded_vectorSet = []
            for i in range(numFold): ## - it's for copy data set of folded data into test tray
                if(curF == i):
                    validating_vectorSet.extend(vector_train[i*nVector:(i+1)*nVector])
                else:
                    folded_vectorSet.extend(vector_train[i*nVector:(i+1)*nVector])
            print("validating vector set: " + str(curF))
            print("the other folding vector set will be TrainSet")
            accuracy_result = validation_loop(k, folded_vectorSet, validating_vectorSet)
            print("test result K = " + str(k),end="\n")
            print("              accuracy = " + str(accuracy_result),end="%\n")
            sum_accuracy += accuracy_result
        everage = sum_accuracy/numFold
        print("---- toal average: "+ str(sum_accuracy) + "/" + str(numFold) + "=" + str(everage),end="%\n")
        optimal_k = k_decision_max(pre_k, pre_accuracy, k, everage)
        pre_k = k
        pre_accuracy = everage
    print("optimal_k is : " + str(optimal_k))
## - else argment 1 is '-t' then
elif (option == "-t"):
    trainFile = sys.argv[2]
    testFile = sys.argv[3]
    train_data = getRows(trainFile)
    test_data = getRows(testFile)

    vector_train = get_vector(train_data,1)
    vector_test  = get_vector(test_data,0)
    random.shuffle(vector_train)

    print("id  income",end="\n")
    nTestVector = 2000
    k = 3
    for each_vector in vector_test:
        decision = kNN_decision(k,vector_train,each_vector)
        print(str(each_vector[0]) + " " + str(decision),end="\n")
else:
    print("------------------------------------------")
    print("- usage : cmd -v/-t traindata [test_data] ")
    print("------------------------------------------")



#### the end of code