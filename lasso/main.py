from util.preproc import *
from dlasso import lasso
from dlasso3 import lar
import matplotlib.pyplot as plt
from scipy import zeros,ones,eye,argmin,argmax,dot,sqrt,c_,r_,array,floor
from numpy.linalg import solve, pinv, inv,norm
import numpy as np
from math import inf,log
import pandas as pd

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# load datasets
df=pd.read_excel('./datasets/diabetes/diabetes.xls')
X,y=df.drop('Y',axis=1),df.iloc[:,df.columns=='Y']
X,y=X.values,y.values
cols=df.columns.values

b,info=lasso(X, y)
#b,info,C=lar(X, y)

#print(b.shape, len(info[5].s))

def lplot(beta,info,cols=range(X.shape[1])):
    # Find best fitting model
    bestAIC, bestIdx = min(info[3].AIC),argmin(info[3].AIC)
    best_s = info[5].s[bestIdx];
    best_beta=beta[:,bestIdx]

    xx=np.array([info[5].s[i][0]   for i in range(len(info[5].s))])
    x = xx.reshape(len(info[5].s),1)
    beta=beta.T

    print('-----------------------')
    print('Feature importance')
    print('-----------------------')
    cols=cols
    for col, coef in zip(cols,best_beta.tolist()):
        print('{}: {}'.format(col,coef))

    # Plot results
    f,ax=plt.subplots(figsize=(6,4))
    ax.plot(x,beta, '.-')
    plt.xlabel(r"$s$",fontsize=18)
    plt.ylabel(r"$\beta$",fontsize=18,rotation=90)
    plt.xticks(color='k',fontsize=18)
    plt.yticks(color='k',fontsize=18)
    ax.legend(list(range(1,len(beta)+1)))
    plt.axvline(best_s,-6, 14, linewidth=0.25, color='r',linestyle=':')
    #plt.show()
    plt.savefig('larsplot')


if __name__ == '__main__':
    print("------------------------------------\n\n")
    print('starting process...')
    lplot(b,info,cols)
    #print("\n")
    #print(info[5].s)
