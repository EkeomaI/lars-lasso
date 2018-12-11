def planerot(x):
    if x[1] != 0:
        r = np.norm(x)
        G = r_[x.T, [-x[1],x[0]]]/r
        x = r_[r, 0]
    else:
        G = np.eye(2,dtype=float)
    return G,x


def chodelete(R, j):
    R = np.delete(R, j, axis=1) # remove column j
    n = R.shape[1]
    for k in range(j,n+1):
        p = range(k,k+2)
        G, R[p,k] = planerot(R[p,k])                 # remove extra element in column
        if k < n:
            R[p,k+1:n] = G*R[p,k+1:n]                # adjust rest of row

    R = np.delete(R, np.s_[-1], axis=0)              # remove zero'ed out row

    return R

def cholinsert(R, x, X):
    diag_k = x.T.dot(x)
    if R.shape == (0,0):
        R = np.array([[sqrt(diag_k)]])

    else:
        col_k = x.T.dot(X)

        R_k = solve(R.T,col_k.T)
        R_kk = sqrt(diag_k - R_k.T.dot(R_k))
        #R = r_[c_[R,R_k],c_[zeros((1,R.shape[0])),R_kk]]
        R = r_[c_[R,R_k],c_[zeros((1,R.shape[1])),R_kk]]

    return R

def center(X):
    n = X.shape[0]
    mu= np.mean(X,axis=0)
    X = X - ones((n,1))*mu
    return X,mu

def normalize(X):
    n = X.shape[0]
    X, mu = center(X)
    d = sqrt(sum(X**2))
    d[d == 0]=1
    X = X/(ones((n,1))*d)
    return X
