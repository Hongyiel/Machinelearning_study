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
    # return (result)
    return ((result-1)/(2**get_len-1))



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
    #### align bit position to MSB side - smaller data loss
    #getWorkSpace_categorial= (rawData[id][12],rawData[id][9],rawData[id][6],rawData[id][11],rawData[id][7],rawData[id][10],rawData[id][8])
    #getMarried_categorial = (rawData[id][14],rawData[id][16],rawData[id][19],rawData[id][18],rawData[id][13],rawData[id][17],rawData[id][15])
    #getOccupation_categorial = (rawData[id][28],rawData[id][21],rawData[id][30],rawData[id][32],rawData[id][24],rawData[id][25],rawData[id][33],rawData[id][26],rawData[id][27],rawData[id][31],rawData[id][23],rawData[id][22],rawData[id][29],rawData[id][20])
    #getRelationship_categorial = (rawData[id][36],rawData[id][39],rawData[id][38],rawData[id][37],rawData[id][35],rawData[id][34])

    #### align bit position to LSB side - data loss is lager then MSB align
    # getWorkSpace_categorial= (rawData[id][8],rawData[id][10],rawData[id][7],rawData[id][11],rawData[id][6],rawData[id][9],rawData[id][12])
    # getMarried_categorial = (rawData[id][15],rawData[id][17],rawData[id][13],rawData[id][18],rawData[id][19],rawData[id][16],rawData[id][14])
    # getOccupation_categorial = (rawData[id][20],rawData[id][29],rawData[id][22],rawData[id][23],rawData[id][31],rawData[id][27],rawData[id][26],rawData[id][33],rawData[id][25],rawData[id][24],rawData[id][32],rawData[id][30],rawData[id][21],rawData[id][28])
    # getRelationship_categorial = (rawData[id][34],rawData[id][35],rawData[id][37],rawData[id][38],rawData[id][39],rawData[id][36])

    ### original binary one_hot_encoding
    getWorkSpace_categorial= rawData[id][6:13]
    getMarried_categorial = rawData[id][13:20]
    getOccupation_categorial = rawData[id][20:34]
    getRelationship_categorial = rawData[id][34:40]

    # getRace_categorial = rawData[id][40:45]


    # print("this is income data: " + str(getIncomeData))

    # encoded
    result_Work_encoded = one_hot_encoding(rawData,getWorkSpace_categorial)
    result_Married_encoded = one_hot_encoding(rawData,getMarried_categorial)
    result_Occupation_encoded = one_hot_encoding(rawData,getOccupation_categorial)
    result_Relationship_encoded = one_hot_encoding(rawData,getRelationship_categorial)
    # result_Race_encoded = one_hot_encoding(rawData,getRace_categorial)

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
    # vector_set.append(float(result_Race_encoded))
    
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


def get_vector(rawData,option):
    i = 1
    extract = []
    for row in rawData[0:-1]:
        extract.append(get_vector_Generator(i,rawData,option))
        i = i + 1
    return extract
#

def kNN_decision(k, train_dataSet, validation_vector):
    sortDistanceList = []
    new_vector = []
    for each_vector in train_dataSet:
        i = 1
        distance = 0
        # get distance calculate between each_vector and ex_vector
        #for i in range(1,11): # excluding id and income
            # 1 ~ 11 (which is 12 values  last will INCOME)
        for i in range(1,10): # excluding id and income
            # 1 ~ 10 (which is 11 values  last will INCOME)
            #if(i != 7): # which do you exclusive componant
            distance += float((each_vector[i] - float(validation_vector[i]))**2)
            
        distance = math.sqrt(distance)

        # ----------- distance,  id number    , income --------
        #new_vector = [distance, each_vector[0], each_vector[11]]
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

  
############################################################
# main routine start from here
############################################################
print("+------------------------------------------------------------------------")
print("| kNN algorithm                2021.04.19      ---------by Suh, HongYiel ")
print("| usage: python this_code_name.py option train_data.cvs [test_data.cvs #]")
print("|  option:       -v to validatiion to find out optimal k value")
print("|                -t to test test vectors ")
print("| [test_data #] when it test option used, # need to set as a number ")
print("| example) python3.0 this_code_name.py -v train.cvs")
print("| example) python3.0 this_code_name.py -t train.cvs test_pub.cvs 43   ")
print("+------------------------------------------------------------------------")
############################################################
## - if argument 1 is '-v' then
option = sys.argv[1]
optimal_k = 1
if (option == "-v"):
    trainFile = sys.argv[2]
    train_data = getRows(trainFile)
    vector_train = get_vector(train_data,1)
    # vector_test  = get_vector(test_data,0)
    # random.shuffle(vector_train)

    validating_vectorSet = []
    folded_vectorSet = []

    pre_k = 1
    pre_accuracy = 0
    max_k = 30
    start_k = 10
    for i in range(max_k): ## - it's for iterating of validation test with different fold
        k = 2*(i+start_k)+1
        numFold = 4
        total_train_vectors = 8000
        nVector = int(total_train_vectors/numFold)
        sum_accuracy = 0
        for curF in range(numFold): ## - it's for iterating of validation test with different fold
            validating_vectorSet = []
            folded_vectorSet = []
            for i in range(numFold): ## - it's for copy data set of folded data into test tray
                if(curF == i):
                    validating_vectorSet.extend(vector_train[i*nVector:(i+1)*nVector])
                else:
                    folded_vectorSet.extend(vector_train[i*nVector:(i+1)*nVector])
            #print("validating vector set: " + str(curF))
            #print("the other folding vector set will be TrainSet")
            accuracy_result = validation_loop(k, folded_vectorSet, validating_vectorSet)
            #print("test result K = " + str(k),end="\n")
            #print("              accuracy = " + str(accuracy_result),end="%\n")
            sum_accuracy += accuracy_result
        everage = sum_accuracy/numFold
         # + str(sum_accuracy) + "/" + str(numFold) + "=" 
        print(" when k is (" + str(k) + ")" + " average : " + str(everage),end="% \n")
        #print(" ----- pre_k is (" + str(pre_k) + ")" + "pre_average : " + str(pre_accuracy),end="% \n")
        optimal_k = k_decision_max(pre_k, pre_accuracy, k, everage)
        if( pre_k != optimal_k):
            pre_k = k
            pre_accuracy = everage
        #print(" ----- pre_k is (" + str(pre_k) + ")" + "pre_accuracy : " + str(pre_accuracy),end="% \n")
        print("optimal_k is : " + str(optimal_k))
    print("finally optimal_k is : " + str(optimal_k))
## - else argment 1 is '-t' then
elif (option == "-t"):
    optimal_k = sys.argv[4]
    trainFile = sys.argv[2]
    testFile = sys.argv[3]
    train_data = getRows(trainFile)
    test_data = getRows(testFile)

    vector_train = get_vector(train_data,1)
    vector_test  = get_vector(test_data,0)
    #random.shuffle(vector_train)

    f = open("kaggle_submit.csv","w")
    f.write("id" + ',' + "income" + '\n')
    print("id  income",end="\n")


    nTestVector = 2000
    k = int(optimal_k)
    change_decision = 0
    for each_vector in vector_test: # weighted short range
        decision1 = kNN_decision(k,vector_train,each_vector)
        decision2 = kNN_decision(k+10,vector_train,each_vector)
        decision3 = kNN_decision(k+24,vector_train,each_vector)
        decisionf = float(decision1 + decision2 + decision3)/3
        
        if(decisionf >= 0.5):
            print(str(each_vector[0]) + " 1",end="\n")
        else:
            print(str(each_vector[0]) + " 0",end="\n")
        decision = int(decisionf)
        # print("type of decision: " + type(decision) )
        f.write(str(each_vector[0]) + ',' + str(decision) + '\n')

    #print("total changing decision is " + str(change_decision))
    f.close()

else:
    print("------------------------------------------")
    print("- usage : cmd -v/-t traindata [test_data] ")
    print("------------------------------------------")



#### the end of code
