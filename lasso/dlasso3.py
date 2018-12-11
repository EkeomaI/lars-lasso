from scipy import zeros,ones,eye,argmin,argmax,dot,sqrt,c_,r_,array,floor
from numpy.linalg import solve, pinv, inv,norm
import numpy as np
import matplotlib.pyplot as plt

def lar(X, y, stop=0, storepath=True, verbose=True, nargout=2):

    n, p = X.shape
    max_variables = min(n-1,p)

    useGram = False
    # if n is approximately a factor 10 bigger than p it is faster to use a
    # precomputed Gram matrix rather than Cholesky factorization when solving
    # the partial OLS problem. Make sure the resulting Gram matrix is not
    # prohibitively large.
    if (n/p) > 10 and p < 1000:
        useGram = True
        Gram = X.T.dot(X)
        #print('Determinant:{}'.format(np.linalg.det(Gram)))
        #print("\n\n")
        #plt.matshow(Gram)
        #plt.show()

    # set up the LAR coefficient vector
    if storepath:
        b = zeros((p, p+1))
    else:
        b = zeros((p, 1))
        b_prev = b

    mu=zeros((n,1))

    I=list(range(0,p))
    A=[]
    C=[]

    if not useGram:
        R=np.zeros((0,0))
    stopCond=0
    step=0

    if verbose:
        print('Step\t\tAdded\t\tActive set size')
        print('-----------------------------------------------\n')

    while len(A) < max_variables and not stopCond:

        r=y-mu
        c=X[:,I].T.dot(r)
        cidx=argmax(abs(c))
        #cmax=c[cidx]
        cmax=max(abs(c))
        #C.append(max(abs(c)))
        C.append(cmax)

        if not useGram:
            R=cholinsert(R,X[:,I[cidx]],X[:,A])
        if verbose:
            print('{}\t\t{}\t\t{}\n'.format(step, I[cidx]+1, len(A)+1))

        A.append(I[cidx])
        #I.remove(cidx)
        I.remove(I[cidx])
        c=np.delete(c,cidx, 0)


        if useGram:
            Gram  = X[:,A].T.dot(X[:,A])
            b_OLS = solve(Gram,(X[:,A].T.dot(y)))
        else:
            b_OLS = solve(R,solve(R.T,X[:,A].T.dot(y)))

        d=X[:,A].dot(b_OLS)-mu

        if len(I)==0:
            gamma = 1
        else:
            cd=X[:,I].T.dot(d)
            gamma=r_[(c-cmax)/(cd-cmax),(c+cmax)/(cd+cmax)]
            gamma=min(gamma[gamma > 0])
            #print('gamma: {}'.format(gamma))

        if storepath:
            b[A,step+1]=b[A,step] + gamma*(b_OLS.reshape(-1)-b[A,step])
        else:
            b_prev = b
            b[A] = b[A] + gamma*(b_OLS - b[A]);

        mu = mu + gamma*d
        step = step + 1

        # Early stopping at specified bound on L1 norm of beta
        if stop > 0:
            if storepath:
                t2 = sum(abs(b[:,step]))
                if t2 >= stop:
                    t1 = sum(abs(b[:,step-1]))
                    s = (stop - t1)/(t2 - t1); # interpolation factor 0 < s < 1
                    b[:,step+1] = b[:,step] + s*(b[:,step] - b[:,step])
                    stopCond = 1

            else:
                t2 = sum(abs(b))
                if t2.any() >= stop:
                    t1 = sum(abs(b_prev))
                    s = (stop - t1)/(t2 - t1) # interpolation factor 0 < s < 1
                    b = b_prev + s*(b - b_prev)
                    stopCond = 1

        # Early stopping at specified number of variables
        if stop < 0:
            stopCond = len(A) >= -stop

    # trim data
    if storepath and b.shape[1] > step:
        b = np.delete(b, np.s_[step+1:], 1)

    # return number of iterations
    steps = step-1

    class structtype():
        pass

    q=steps+1
    info = [structtype() for j in range(6)]

    info[0].steps = step-1
    info[1].df = []
    info[2].Cp = []
    info[3].AIC= []
    info[4].BIC= []
    info[5].s  = []

    nargout=2

    if nargout==2:
        b0=pinv(X).dot(y)
        #print('b0: {}'.format(b0))
        penalty0=sum(abs(b0))
        indices=np.arange(0,p)

        #print('Active set: {}'.format(indices))

        if storepath:
            q=info[0].steps+1

            sigma2e = sum(np.float_power((y - X.dot(b0)),2))/n

            for step in range(0,steps+2):
                idx = indices[b[:,step]!=0] # active set
                #print('indices: {}, step:{}'.format(idx,step))
                if len(idx)==0:
                    r=y
                else:
                    #print('X:{}, b:{}'.format( X[:,idx].shape, b[idx,step].reshape(-1,1).shape))
                    r=y-X[:,idx].dot(b[idx,step].reshape(-1,1))
                rss=sum(np.float_power(r,2))

                #print('rss:{}'.format(rss))

                info[1].df.append(step)
                info[2].Cp.append(rss/sigma2e - n + 2*step)
                info[3].AIC.append(rss + 2*sigma2e*step)
                info[4].BIC.append(rss + np.log1p(n)*sigma2e*step)
                info[5].s.append(sum(abs(b[idx,step]))/penalty0)

        else:
            info[5].s.append(sum(abs(b))/penalty0)
            info[1].df.append(info[0].steps)

    return b, info, C
