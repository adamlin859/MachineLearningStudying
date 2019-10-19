Questions

What it the complexity of k-means clustering?

O(t*k*n*d)
    t: number of iterations 
    k: number of centroids 
    n: number of points
    d: number of features

How to pick the best k value?

We can use the elbow method to determine the number of k. This can be done through running the algorithm many times for different K and take the SSE of each of the points. 

