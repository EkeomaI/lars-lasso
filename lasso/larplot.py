import matplotlib.pyplot as plt
from scipy import argmin,argmax
import numpy as np

def lar_plot(X,y,stop=0,storepath=True,verbose=True,measure=2,cols=range(X.shape[1])):

    if stop > 0:

        beta,info,C=lar(X, y,stop, storepath,verbose,measure)

        # Find best fitting model
        bestAIC, bestIdx = min(info[3].AIC),argmin(info[3].AIC)
        best_s = info[5].s[bestIdx];

        # Find best fitting model
        bestAIC, bestIdx = min(info[3].AIC),argmin(info[3].AIC)
        best_s = info[5].s[bestIdx];

        xx=np.array([info[5].s[i][0]   for i in range(len(info[5].s))])
        x = xx.reshape(len(info[5].s),1)
        beta=beta.T
        best_beta=beta[:,bestIdx]

        #
        #print('-----------------------')
        #print('Feature importance')
        #print('-----------------------')
        #cols=cols
        #for col, coef in zip(cols,best_beta.tolist()):
            #print('{}: {}'.format(col,coef))

        # Plot results
        f,ax=plt.subplots(figsize=(10,8))
        ax.plot(x,beta, '.-');
        plt.xlabel(r"$s$",fontsize=18)
        plt.ylabel(r"$\beta$",fontsize=18,rotation=90)
        plt.xticks(color='k',fontsize=18)
        plt.yticks(color='k',fontsize=18)
        ax.legend(list(range(1,len(beta)+1)))
        plt.axvline(best_s,-6, 14, linewidth=0.25, color='r',linestyle=':');
        plt.savefig('larplot_path')
        plt.show()
    else:
        beta,info,C=lar(X, y,stop, storepath,verbose,measure)

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
        f,ax=plt.subplots(figsize=(10,8))
        ax.plot(x,beta, '.-');
        plt.xlabel(r"$s$",fontsize=18)
        plt.ylabel(r"$\beta$",fontsize=18,rotation=90)
        plt.xticks(color='k',fontsize=18)
        plt.yticks(color='k',fontsize=18)
        ax.legend(list(range(1,len(beta)+1)))
        plt.axvline(best_s,-6, 14, linewidth=0.25, color='r',linestyle=':');
        plt.savefig('larplot')
        plt.show()

    return beta,info
