from scipy import zeros,ones,eye,argmin,argmax,dot,sqrt,c_,r_,array,floor
from numpy.linalg import solve, pinv, inv,norm
from dlasso2 import larsen
import numpy as np

def lasso(X, y, stop=0, storepath=True, verbose=False):

    # LARS variable setup
    n, p = X.shape

    Gram = zeros((0,0))
    # if n is approximately a factor 10 bigger than p it is faster to use a
    # precomputed Gram matrix rather than Cholesky factorization when solving
    # the partial OLS problem. Make sure the resulting Gram matrix is not
    # prohibitively large.

    if (n/p) > 10 and p < 1000:
        Gram = dot(X.T,X) # precomputed Gram matrix
        print('Gram matrix: {}'.format(Gram.shape))
        #print('Determinant of X: {}'.format(np.linalg.det(X)))

    # Run the LARS algorithm
    b, steps = larsen(X, y, 0, stop, Gram, storepath, verbose)

    print('b: {}'.format(b.shape))
    print('steps: {}'.format(steps))

    class structtype():
        pass

    q=steps+1
    info = [structtype() for j in range(6)]

    info[0].steps = steps-1
    info[1].df = []
    info[2].Cp = []
    info[3].AIC= []
    info[4].BIC= []
    info[5].s  = []

    nargout=2.
    # Compute auxilliary measures
    if nargout == 2: # only compute if asked for
        st = steps
        b0 = pinv(X).dot(y) # regression coefficients of low-bias model
        penalty0 = sum(abs(b0)) # L1 constraint size of low-bias
        indices = np.arange(0,p)

        if storepath: # for entire path
            q = info[0].steps+1;
            sigma2e = sum((y - X.dot(b0))**2)/n
            lmbda = []
        for step in range(0,steps+2):
            idx = indices[b[:,step]!=0] # active set
            #idx=np.argwhere(b[:,step] != 0)[:,0]
            print('indices: {}, step:{}'.format(idx,step))
            if len(idx)==0:
                r=y
            else:
                #print('X:{}, b:{}'.format( X[:,idx].shape, b[idx,step].reshape(-1,1).shape))
                r=y-X[:,idx].dot(b[idx,step].reshape(-1,1))
            rss=sum(np.float_power(r,2))

            info[1].df.append(step)
            info[2].Cp.append(rss/sigma2e - n + 2*step)
            info[3].AIC.append(rss + 2*sigma2e*step)
            info[4].BIC.append(rss + np.log1p(n)*sigma2e*step)

            # compute L1 penalty constraints s and lambda
            info[5].s.append(sum(abs(b[idx,step]))/penalty0)
            if step == 0:
                lmbda.append( max(2*abs(dot(X.T,y))) )
            else:
                lmbda.append( np.median(2*abs(dot(X[:,idx].T,r))) )

    else: # for single solution
        idx = indices[b!=0] # active set
        info[5].s.append(sum(abs(b))/penalty0)
        info[1].df.append(info[0].steps)
        if len(idx)==0:
            lmbda.append( max(2*abs(X.T.dot(y))) )
        else:
            lmbda.append( np.median(2*abs(dot(X[:,idx].T,(y - dot(X[:,idx],b[idx]))))) )

    return b, info
