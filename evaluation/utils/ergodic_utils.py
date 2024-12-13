import jax.numpy as np
import numpy as onp

#######################
### Basis functions ###
#######################

# One-D
def gk_1D(k,x,L):
    if k == 0:
        hk = 1
    else:
        hk = np.sqrt(2)
    return hk*np.cos((k + 0) * (x) * np.pi / L) # Discrete cosine transform II
    # return onp.cos(k * x * onp.pi / L) # Fourier transform (cosine part only)

# Two-D
def gk_2D(ks, xs, Ls):
    prod = 1
    for k, x, L in zip(ks, xs, Ls):
        prod *= gk_1D(k,x,L)
    return prod

# n-D: probably never going to get here, but maybe?
def gk_nD(ks, xs, Ls):
    assert(len(ks) == len(xs))
    assert(len(ks) == len(Ls))
    prod =1
    for k, x, L in zip(ks, xs, Ls):
        prod *= gk_1D(k,x,L)
    return prod

########################################
###### Discrete Cosine Transforms ######
########################################

def get_phiks_1D(X, mu, L, kmax):
    '''
    One-dimensional discrete cosine transform
    Inputs:
        res: resolution
    Outputs:
        phiks: array of length res, with coefficients
    '''
    phiks = onp.zeros(kmax)
    for i,k in enumerate(range(1,kmax+1)):
        for _x, _mu in  zip(X, mu):
            phiks[i] += gk_1D(k,_x,L) * _mu
    return phiks / len(mu)

def get_cks_1D(traj, L, kmax):
    '''
    One-dimensional transform
    Inputs:
        traj: y coordinates (time-independent)
        res: resolution
    Outputs:
        c_ks: array of length res, with coefficients
    '''
    c_ks = onp.zeros(kmax)
    for i,k in enumerate(range(1,kmax+1)):
        for _y in  traj:
            c_ks[i] += gk_1D(k,_y,L)
    return c_ks / len(traj)

def get_cks_2d(Xs,Ys,Ls,kmax):
    '''
    2-Dimensional discrete cosine transform
    Inputs:
        Xs: list of coordinates [[x1_1,x2_1], [x1_2,x2_2], ... [x1_n,x2_n]]
        Ys: list of values [y1, y_2, ... y_n]
    '''
    c_ks = onp.zeros(kmax + np.array([1,1]))
    for k1 in range(kmax[0]+1):
        for k2 in range(kmax[1]+1):
            for i in range(len(Xs)):
                c_ks[k1][k2] += gk_nD([k1,k2],Xs[i],Ls) * Ys[i]
    return c_ks

###################################
############# Inverses ############
###################################

### One Dimension

def inverse_cks_function_1D(cks,L):
    '''
    returns a reconstruction of the original function from the dct
    '''
    def inverse_func(x):
        result = 0
        for i in range(len(cks)):
            k = i + 1
            result += gk_1D(k,x,L) * cks[i]
        return result + 1
    return inverse_func

def inverse_cks_1D(cks,X, L):
    '''
    Applies the inverse function to the range X 
    (in a more efficient manner than looping through with inverce_dct_function_1D)
    '''
    result = X * 0 + 1
    for i in range(len(cks)):
        k = i + 1
        result = result + onp.array([gk_1D(k,_x,L) for _x in X])*cks[i]
    return result


### Two Dimensions

def inverse_dct_2D_func(Xs,cks,Ls):
    '''
    returns a reconstruction of the original function from the dct_2d
    '''
    multiplier = Ls[0]*Ls[1]/2
    def inverse_func(x):
        y = 0
        for k1 in range(cks.shape[0]):
            for k2 in range(cks.shape[1]):
                y += gk_2D([k1,k2],x,Ls) * cks[k1,k2] / multiplier
        return y
    return inverse_func

def inverse_dct_2D(Xs,cks,Ls):
    '''
    Applies the inverse function to the range Xs
    (in a more efficient manner than looping through with inverce_dct_function_2D)
    Inputs:
        Xs: list of coordinates [[x1_1,x2_1], [x1_2,x2_2], ... [x1_n,x2_n]]
        cks: output from dct_2D
        Ls: lengths for each dimension [L1, L2]
    Outputs:
        ys: [y1, y2, ... yn] corresponding to the points in Xs
    '''
    ys = onp.zeros([len(Xs)])
    multiplier = len(Xs) * len(Xs[0])
    for k1 in range(len(cks)):
        for k2 in range(len(cks[0])):
            ys = ys + onp.array([gk_2D([k1,k2],_xs,Ls) for _xs in Xs])*cks[k1][k2] / multiplier
    return ys


if __name__ == "__main__":
    def f(x):
        return 2.849594931 * (onp.exp(-((x-0.5)/0.1)**2)+onp.exp(-((x-0.75)/0.1)**2))
    x = onp.linspace(0,1,1000)
    y = f(x)
    phiks = get_phiks_1D(X=x, mu=y, L=1,res=11)
    print(phiks[1:])