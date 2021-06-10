import numpy as np
testing = [[3, 2 ,3],[3,1,2],[2,1,5]]
assignments = [2,3,1,2,2,3,3,1,2,2,3,1,2]

print(len(testing))

print(range(len(testing)))

print(testing[:len(testing)])

print(testing[len(testing):])
k = 3
n = 10
centroids = []      
for j in range(k):
    print("this is j: ",j)
    for i in range(n):
        print(assignments[i] == j)
        print("This is assignments[i]: ", assignments[i])
        
        centroids[k] = np.mean(testing[assignments[i] == j], axis=0)
