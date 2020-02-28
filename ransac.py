import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time
from scipy import spatial
from random import randrange


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points------------> PONTOS DO LASER
      B: Nxm numpy array of corresponding points -----------> PONTOS QUE VEM DO LASER QUE CORRESPONDEM AOS DO LASER
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    # get number of dimensions
    m = A.shape[1] # DIMENSAO : À PARTIDA SERÁ 2

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t



def nearest_neighbor(laser, mapa):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(mapa)
    distances, indices = neigh.kneighbors(laser, return_distance=True) # INDICES DO MAPA; DISTANCIAS PARA CADA PONTO DO LASER. TAMANHO = LASER
    ind = []
    dist = []

    dist,ind = spatial.KDTree(mapa).query(laser)


    return dist, ind




def icp(A, B, init_pose=None, max_iterations=40, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points ----------------------> A TEM QUE SER O LASER
        B: Nxm numpy array of destination mD point -------------------> B TEM QUE SER O MAPA
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    #assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))

    mapa = np.ones((m+1,B.shape[0]))
    
    src[:m,:] = np.copy(A.T)
    
    mapa[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, mapa[:m,:].T) ## PORQUE?? src[:m,:].T = A # RESOLVIDO


        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, mapa[:m,indices].T)  # mapa[:m,indices].T PONTOS NO MAPA QUE ESTAO MAIS PERTO DOS PONTOS DO LASER

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i

def rotation_matrix(axis, theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])


def RANSAC(A, B, it = 50, threshold = 0.1, d = 50, icp_tolerance = 0.000001, init_pose=None, max_icp_iterations=40):
    # A -------tem que ser nuvem do laser
    # B -------tem que ser mapa
    i = 0
    best = 0
    best_error = 10000
    while(i < it):
        x = randrange(A.shape[0])
        y = randrange(A.shape[0])
        z = randrange(A.shape[0])
        idx = np.sort([x,y,z])
        maybeInliers =  np.array([A[idx[0]],A[idx[1]],A[idx[2]]])
        T, distances, iterations = icp(maybeInliers, B, init_pose, max_icp_iterations, icp_tolerance) # MAYBE MODEL
        alsoInliers = []
        subset = A # Elimino pontos que usei para determinar o modelo
        C = np.ones((subset.shape[0], 3))
        C[:,0:2] = np.copy(subset) # transformo todos os pontos para a transformacao que determinei
        C = np.dot(T, C.T).T
        C = np.delete(C, 2 , axis=1)
        # Processo para determinar closest points..
        m = A.shape[1] #=2
        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1,C.shape[0]))

        mapa = np.ones((m+1,B.shape[0]))
    
        src[:m,:] = np.copy(C.T)
    
        mapa[:m,:] = np.copy(B.T)

        distances2, indices2 = nearest_neighbor(src[:m,:].T, mapa[:m,:].T)
        
        if(sum(distances2<threshold)>d and sum(distances2<threshold)>best): # d é parametro ajustavel
            best = sum(distances2<threshold)
            T, distances, iterations = icp(A[distances2<threshold], B, init_pose, max_icp_iterations, icp_tolerance)
            # check error
            mean_error = np.mean(distances)
            if (np.abs(mean_error) < np.abs(best_error)):
                best_error = mean_error
                bestT = T
        i += 1
    try:
        return bestT
    except:
        return T




def test_icp():
    N = 200
    dim = 2
    noise_sigma = 0.01
    A = np.random.rand(N, dim)
    translation = 0.5  
    rotation = 0 
    best = 0
    best_error = 10000

    B = np.copy(A)

    # Translate
    t = np.random.rand(dim)*translation
    t = np.array([0.5, 0])
    B += t

    # Rotate
    R = rotation_matrix(0,  rotation)
    B = np.dot(R, B.T).T
    
    B += np.random.randn(N, dim) * noise_sigma

    # Run ICP
    
    iterarations = 100
    i = 0

    start = time.time()
    B = np.delete(B, [1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17], axis=0)
    #print(B.shape)
    #print(A.shape)
    bestT = RANSAC(B, A, 50, 0.1, 50)
    end = time.time()

    #print(bestT)
    # Make C a homogeneous representation of B
    C = np.ones((B.shape[0], 3))
    C[:,0:2] = np.copy(B)

    # Transform C
    C = np.dot(bestT, C.T).T
    #print(bestT)
    plt.figure(1)
    plt.scatter(A[:,0],A[:,1])
    plt.scatter(B[:,0],B[:,1])
    plt.scatter(C[:,0],C[:,1])
    plt.show()
    print(end-start)
    print()
    return


if __name__ == "__main__":

    # Constants
    N = 10                                    # number of random points in the dataset
    num_tests = 10                            # number of test iterations
    dim = 2                                     # number of dimensions of the points
    noise_sigma = 0                          # standard deviation error to be added
    translation = 0.1                          # max translation of the test set
    rotation = 0                            # max rotation (radians) of the test set   

    #test_best_fit()
    test_icp()