import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

# Toy problem with 3 clusters for us to verify k-means is working well
def toyProblem():
  # Generate a dataset with 3 cluster
  X = np.random.randn(150,2)*1.5
  X[:50,:] += np.array([1,4])
  X[50:100,:] += np.array([15,-2])
  X[100:,:] += np.array([5,-2])

  # Randomize the seed
  np.random.seed()

  # Apply kMeans with visualization on
  k = 3
  max_iters=20
  centroids, assignments, SSE = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False)
  plotClustering(centroids, assignments, X, title="Final Clustering")
  
  # Print a plot of the SSE over training
  plt.figure(figsize=(16,8))
  plt.plot(SSE, marker='o')
  plt.xlabel("Iteration")
  plt.ylabel("SSE")
  plt.text(k/2, (max(SSE)-min(SSE))*0.9+min(SSE), "k = "+str(k))
  plt.show()


  #############################
  # Q5 Randomness in Clustering
  #############################
  k = 5
  max_iters = 20

  SSE_rand = []
  # Run the clustering with k=5 and max_iters=20 fifty times and 
  # store the final sum-of-squared-error for each run in the list SSE_rand.
  # raise Exception('Student error: You haven\'t implemented the randomness experiment for Q5.')
  
  for a in range(50):
    centroids, assignments, sse_t = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False)
    SSE_rand.append(sse_t)
    
  plotClustering(centroids, assignments, X, title="Second Clustering")

  
  # Plot error distribution
  plt.figure(figsize=(8,8))
  plt.hist(SSE_rand, bins=20)
  plt.xlabel("SSE")
  plt.ylabel("# Runs")
  plt.show()

  ########################
  # Q6 Error vs. K
  ########################

  SSE_vs_k = []
  # Run the clustering max_iters=20 for k in the range 1 to 150 and 
  # store the final sum-of-squared-error for each run in the list SSE_vs_k.
  k = 1
  for a in range(150):
    k = a + 1
    centroids, assignments, sse_t = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False)
    SSE_vs_k.append(sse_t)
  
  # raise Exception('Student error: You haven\'t implemented the randomness experiment for Q6.')

  # Plot how SSE changes as k increases
  plt.figure(figsize=(16,8))
  plt.plot(SSE_vs_k, marker="o")
  plt.xlabel("k")
  plt.ylabel("SSE")
  plt.show()
  print("END TOY")


def imageProblem():
  np.random.seed()
  # Load the images and our pre-computed HOG features
  data = np.load("img.npy")
  img_feats = np.load("hog.npy")


  # Perform k-means clustering
  k=3
  centroids, assignments, SSE = kMeansClustering(img_feats, k, 30, min_size=0)

  print("This is k = {{k}}'s sse",k , SSE)
  # Visualize Clusters
  for c in range(len(centroids)):
    # Get images in this cluster
    members = np.where(assignments==c)[0].astype(np.int)
    imgs = data[np.random.choice(members,min(50, len(members)), replace=False),:,:]
    
    # Build plot with 50 samples
    print("Cluster "+str(c) + " ["+str(len(members))+"]")
    _, axs = plt.subplots(5, 10, figsize=(16, 8))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img,plt.cm.gray)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    # Fill out plot with whitespace if there arent 50 in the cluster
    for i in range(len(imgs), 50):
      axs[i].axes.xaxis.set_visible(False)
      axs[i].axes.yaxis.set_visible(False)
    plt.show()



##########################################################
# initializeCentroids
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   k --  integer number of clusters to make
#
# Outputs:
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
##########################################################

def initalizeCentroids(dataset, k):
  idx = np.random.permutation(dataset.shape[0])
  centroids = dataset[idx[:k]]
  # print("This is centroids: ", centroids)
  # raise Exception('Student error: You haven\'t implemented initializeCentroids yet.')
  return centroids

##########################################################
# computeAssignments
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#
# Outputs:
#   assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
##########################################################

def computeAssignments(dataset, centroids):
  
  # How do I get num_clusters from outside
  clusters = centroids.shape[0] #k = j
  distance = np.zeros((dataset.shape[0], clusters))
  for k in range(clusters):
    dist = np.linalg.norm(dataset - centroids[k], axis=1)
    distance[:,k] = np.square(dist)
  # raise Exception('Student error: You haven\'t implemented computeAssignments yet.')
  # print(np.shape(assignments))
  # print("this is distance: ", distance)

  assignments = np.argmin(distance, axis=1)
  # print("This is shape of assignments: ", np.shape(assignments)) # n * 1

  return assignments

##########################################################
# updateCentroids
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#   assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
# Outputs:
#   centroids -- n x d matrix of centroid points where the
#                 j'th row represents c_j after being updated
#                 as the mean of assigned points
#   counts -- k x 1 matrix where the j'th entry is the number
#             points assigned to cluster j
##########################################################

def updateCentroids(dataset, centroids, assignments):
  
  # z_i = assignments that already after argmin (n x 1)
  # c_j = updated centroids (k x d)
  # j = k
  # X_i = dataset (n x d)
  k = centroids.shape[0]
  # Also return the count of datapoints assigned to each centroid as a vector
  counts = []
  j = 0
  for j in range(k):
    centroids[j] = np.mean(dataset[assignments == j], axis=0)
    counts.append(np.count_nonzero(dataset[assignments == j]))
  return centroids, counts
  

##########################################################
# calculateSSE
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#   assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
# Outputs:
#   sse -- the sum of squared error of the clustering
##########################################################

def calculateSSE(dataset, centroids, assignments):
  
  # print("This is SSE centroids: ", centroids)
  # print("This is SSE assignments: ", assignments)
  clusters = centroids.shape[0]
  # sse = np.zeros((dataset.shape[0], clusters))
  sse = []
  for k in range(clusters):
    sse_get = np.linalg.norm(dataset[assignments == k] - centroids[k], axis=0)
    sse_sum = np.sum(sse_get)
    sse_square = np.square(sse_sum)
    sse.append(sse_square)
    
  sse = np.sum(sse)
  # print("This is sse: ", sse)
  return sse
  

########################################
# Instructor Code: Don't need to modify 
# beyond this point but should read it
########################################

def kMeansClustering(dataset, k, max_iters=10, min_size=0, visualize=True):
  
  # Initialize centroids
  centroids = initalizeCentroids(dataset, k)
  # plotClustering(centroids,None,dataset,title=None)
  # Keep track of sum of squared error for plotting later
  SSE = []

  # Main loop for clustering
  for i in range(max_iters):

    # Update Assignments Step
    assignments = computeAssignments(dataset, centroids)
    # Update Centroids Step
    centroids, counts = updateCentroids(dataset, centroids, assignments)

    # Re-initalize any cluster with fewer then min_size points
    for c in range(k):
      if counts[c] <= min_size:
        centroids[c] = initalizeCentroids(dataset, 1)
        
    
    # print("This is after 3 steps of centroids: ", centroids)

    if visualize:
      plotClustering(centroids, assignments, dataset, "Iteration "+str(i))
    SSE.append(calculateSSE(dataset,centroids,assignments))

    # Get final assignments
    assignments = computeAssignments(dataset, centroids)

  return centroids, assignments, SSE

def plotClustering(centroids, assignments, dataset, title=None):
  plt.figure(figsize=(8,8))
  plt.scatter(dataset[:,0], dataset[:,1], c=assignments, edgecolors="k", alpha=0.5)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=5, edgecolors="k", s=250)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=2, edgecolors="w", s=200)
  if title is not None:
    plt.title(title)
  plt.show()


if __name__=="__main__":
  toyProblem()
  imageProblem()
