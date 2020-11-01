import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

########################## calculate_kMeans ##################################
# Purpose:
#   Calculate k-means using 2-d Euclidian objective function on 2-d
# Parameters:
#   I       Numpy Array     dPoints         2-d numerical
#   I       Int             k               # of clusters desired
#   I       Tuple Array     randClust       coordinates for initial clusters   
# Returns:
#   O       DataFrame       dataOut     Each row is a cluster. 
# Notes:
#   None
def calculate_kMeans(dPoints, k, randClust):
    for xPoint, yPoint in dPoints:
        distances = []
        for xClust, yClust in randClust:
            distance = np.sqrt(pow((xClust - xPoint), 2) \
                               + pow((yClust - yPoint), 2))
            distances.append((round(distance, 3), (xClust, yClust)))
        
        print("(", xPoint, ",", yPoint, ")", ":", distances, "\n")
        
        
######################### sklearn_kMeans #####################################
# Purpose:
#   Use Sklearn cluster.KMeans to get clusters
#   Use various parameter values and plot the clusters. 
# Parameters:
#   None
# Returns:
#   None
# Notes:
#   None
def sklearn_kMeans():
    data = pd.read_csv('hwk08.csv')
    X = data[['A', 'B', 'C']]
    y = data['D']
    
    # plot 1 - 2 clusters
    kMeans2_yPred = KMeans(n_clusters = 2, random_state = 0).fit_predict(X)
    
    plt.figure(figsize = (12, 12))
    plt.scatter(X['A'], X['B'], c = kMeans2_yPred)
    plt.title('K = 2, columns (A, B)')
    plt.show()
    
    # Plot 2 - 2 clusters
    plt.figure(figsize = (12, 12))
    plt.scatter(X['B'], X['C'], c = kMeans2_yPred)
    plt.title('K = 2, columns (B, C)')
    plt.show()
    
    # plot 3 - 3 clusters
    kMeans3_yPred = KMeans(n_clusters = 3, random_state = 0).fit_predict(X)
    
    plt.figure(figsize = (12, 12))
    plt.scatter(X['A'], X['C'], c = kMeans3_yPred)
    plt.title('K = 3, columns (A, B)')
    plt.show()

def main():
    data = [[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]]
    initClusterPoints = [(2, 5), (5, 8), (1, 2)]
    # calculate_kMeans(data, 3, initClusterPoints)
    sklearn_kMeans()    

# Context the file is running in is __main__ 
if __name__ == "__main__":
    main()