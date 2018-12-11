from scipy import zeros,ones,eye,argmin,argmax,dot,sqrt,c_,r_,array,floor
from numpy.linalg import solve, pinv, inv,norm
import numpy as np

def larsen(X, y, delta, stop, Gram, storepath, verbose):

    # algorithm setup
    n, p = X.shape
    eps=2.22e-16
    # Determine maximum number of active variables
    if delta < eps:
        maxVariables = min(n,p) #LASSO
        #print('maxVariables: {}'.format(maxVariables))
    else:
        maxVariables = p # Elastic net

    maxSteps = 8*maxVariables # Maximum number of algorithm steps

    # set up the LASSO coefficient vector
    if storepath:
        b = zeros((p, 2*p))
    else:
        b = zeros((p, 1))
        b_prev = b


    # current "position" as LARS travels towards lsq solution
    mu = np.zeros((n,1))

    # Is a precomputed Gram matrix supplied?
    useGram = not Gram.shape==(0,0)

    I = list(range(0,p)) # inactive set
    A = []               # active set

    if not useGram:
        R = np.zeros((0,0)) # Cholesky factorization R'R = X'X where R is upper triangular

    # correction of stopping criterion to fit naive Elastic Net
    if delta > 0 and stop > 0:
        stop = stop/(1 + delta)

    lassoCond = 0 # LASSO condition boolean
    stopCond = 0  # Early stopping condition boolean
    step = 0      # step count

    if verbose:
        print('Step\tAdded\tDropped\t\tActive set size')
        print('-----------------------------------------------\n')

    ## LARS main loop
    #  while not at OLS solution, early stopping criterion is met, or too many
    #  steps have passed
    while len(A) < maxVariables and not stopCond and (step < maxSteps):
        r = y - mu
        # find max correlation
        c=X[:,I].T.dot(r)
        #print('shape of c: {}'.format(c.shape))
        #print('shape of r: {}'.format(r.shape))

        cidx=argmax(abs(c))   # index of next active variable
        cmax=max(abs(c))

        if  not lassoCond:
        # add variable
            if not useGram:
                R = cholinsert(R, X[:,I[cidx]], X[:,A])
            if verbose:
                print('{}\t\t{}\t\t\t\t\t{}\n'.format(step,I[cidx]+1,len(A)+1))
            A.append(I[cidx]) # add to active set
            I.remove(I[cidx]) # ...and drop from inactive set
            c=np.delete(c,cidx,0)
        else:
        # if a variable has been dropped, do one step with this
        # configuration (don't add new one right away)
            lassoCond = 0
        # partial OLS solution and direction from current position to the OLS
        # solution of X_A
        if useGram:
            Gram=X[:,A].T.dot(X[:,A])
            b_OLS = solve(Gram,dot(X[:,A].T,y)) # same as X(:,A)\y, but faster
        else:
            b_OLS = solve(R,solve(R.T,X[:,A].T.dot(y)))
        d=X[:,A].dot(b_OLS)-mu

        # compute length of walk along equiangular direction
        if storepath:
            if b_OLS.size==1:
                gamma_tilde = b[A[0],step] / (b[A[0],step] - b_OLS[-1])
                #print('gamma_tilde: {}'.format(gamma_tilde))
            else:
                gamma_tilde = b[A[:-1],step] / (b[A[:-1],step] - b_OLS[:-1].reshape(-1))
        else:
            gamma_tilde = b[A[:-1]]/(b[A[:-1]] - b_OLS[:-1].reshape(-1))

        #print('gamma_tilde: {}'.format(gamma_tilde))

        gamma_tilde[gamma_tilde <= 0] = np.inf
        dropIdx = argmin(gamma_tilde)
        gamma_tilde=gamma_tilde[dropIdx]

        if len(I)==0:
        # if all variables active, go all the way to the OLS solution
            gamma = 1;
        else:
            cd=X[:,I].T.dot(d)
            temp=r_[(c-cmax)/(cd-cmax),(c+cmax)/(cd+cmax)]
            temp=np.sort(temp[temp > 0]) #min(temp[temp > 0])

            if temp.shape==0:
                ValueError('SpaSM:larsen', 'Could not find a positive direction towards the next event.');
            gamma = temp[0]

        # check if variable should be dropped
        if gamma_tilde < gamma:
            lassoCond = 1
            gamma = gamma_tilde

        # update beta
        if storepath:
        # check if beta must grow
            #if b.shape[0] < step:
                #b = c_[b, zeros((b.shape[0],))]
            b[A,step+1]=b[A,step] + gamma*(b_OLS.reshape(-1)-b[A,step]) # update beta
        else:
            b_prev = b
            b[A] = b[A] + gamma*(b_OLS - b[A]) # update beta for single step

        # update position
        mu = mu + gamma*d

        # increment step counter
        step = step+1

        # Early stopping at specified bound on L1 norm of beta
        if stop > 0:
            if storepath:
                t2 = sum(abs(b[:,step]))
                if t2 >= stop:
                    t1 = sum(abs(b[:,step-1]))
                    s = (stop - t1)/(t2 - t1) # interpolation factor 0 < s < 1
                    b[:,step+1] = b[:,step] + s*(b[:,step] - b[:,step])
                    stopCond = 1

            else:
                t2 = sum(abs(b))
                if t2.any() >= stop:
                    t1 = sum(abs(b_prev));
                    s = (stop - t1)/(t2 - t1); # interpolation factor 0 < s < 1
                    b = b_prev + s*(b - b_prev)
                    stopCond = 1

            # If LASSO condition satisfied, drop variable from active set
            if lassoCond:
                if verbose:
                    print('{}\t\t\t\t{}\t\t\t{}\n'.format(step, A[dropIdx],len(A)-1));

                if not useGram:
                    R = choldelete(R, dropIdx)

                I.append(A[dropIdx]) # add dropped variable to inactive set
                A.remove(dropIdx)    # ...and remove from active set


            # Early stopping at specified number of variables
            if stop < 0:
                stopCond = len(A) >= -stop

    print('number of cols: {}'.format(b.shape[1]))
    print('number of steps: {}'.format(step))

    # trim beta
    if storepath and b.shape[1] > step:
        b = np.delete(b, np.s_[step+1:], 1)

    # return number of iterations
    steps = step-1

    # issue warning if algorithm did not converge
    if step == maxSteps:
        ValueError('SpaSM:larsen', 'Forced exit. Maximum number of steps reached.');

    return b, steps
