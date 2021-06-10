import numpy as np

# setting data set from hw document

# find p(y|x) is T / F

# find p(y|x)'s class is positive / negative

# get result

data = [[0, 0.1],[0, 0.1],[0, 0.25], [1, 0.25], [0, 0.3], [0, 0.33], [1, 0.4], [0, 0.52],
        [0, 0.55],[1,0.7],[1, 0.8], [1, 0.9], [1, 0.9], [1, 0.95], [1, 1.0]]

# t = [0, 0.2, 0.4, 0.6, 0.8, 1]

t = float(input())


# def if data[0][0] is positive: given if statement return 1 which mean positive
#                      negative: given if statement return 0 which mean negative
def distinquishPN(vector):
    if vector[0] == 1:
        return "positive"
    elif vector[0] == 0:
        return "negative"
    else:
        return -1
    

# def if data[0][1] is true: given if statement is satisfied
#                     false: given if statement is not satisfied

def distinquishTF(vector,PNresult):
    # for i in len(t)
    if PNresult == "positive":
        if vector[1] > t:
            return "True"
        elif vector[1] <= t:
            return "False"
        else:
            return -1
        
    elif PNresult == "negative":
        if vector[1] <= t:
            return "True"
        elif vector[1] > t:
            return "False"
        else:
            return -1
    
def count_r(result):
    falseNegative = 0
    trueNegative = 0
    falsePositive = 0
    truePositive = 0   
    i = len(result)

    print("incount") 

    for vector in result:
        if vector[0] == 'False':
            temp = 0
        else:
            temp = 1
                
        if vector[1] == 'negative':  
            if temp == 0:
                falseNegative = 1 + falseNegative
            else:
                trueNegative = 1+ trueNegative                
        if vector[1] == 'positive':
            if temp == 0:
                falsePositive = 1 + falsePositive
            else:
                truePositive = 1 + truePositive   
    print("falseNegative,falsePositive,trueNegative,truePositive")
    countList = [falseNegative,falsePositive,trueNegative,truePositive]
    return countList
# get all return value here

def Recall(data):
    
    # TP / (TP + FN)
    
    re = float(data[3]) / float((data[3]) + float(data[0]))
    return re
    
def Precision(data):
    
    pr = float(data[3]) / (float(data[3]) + float(data[1]))
    return pr

result = []

# merge results
for vector in data:
    PNresult = distinquishPN(vector)
    TFresult = distinquishTF(vector,PNresult)
    
    result.append([TFresult,PNresult])

print(result)    

# count all True negative / True positive / False negative / False positive


theResult = count_r(result)
print(theResult)            

# count Recall and Precision

r = Recall(theResult)
p = Precision(theResult)

print("Recall is ", r)
print("Precision is ", p)
        
            
            




