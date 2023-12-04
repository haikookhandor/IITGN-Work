from tree.base import DecisionTree
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.extmath import weighted_mode
from time import perf_counter


class BaggingClassifier():
    def __init__(self, base_estimator=DecisionTree, n_estimators=5,
                 max_depth=100, criterion="information_gain", n_jobs = 1):
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.trees = []
        self.datas = []
        self.n_jobs = n_jobs

    def fit(self, X, y):
        def fit_tree(X, y):
            A = []
            X_sub = X.sample(frac=1, axis='rows', replace=True)
            y_sub = y[X_sub.index]
            X_sub = X_sub.reset_index(drop=True)
            y_sub = y_sub.reset_index(drop=True)

            # Using sampled data to learn tree
            tree = self.base_estimator(criterion=self.criterion)
            # print(X_sub, y_sub)
            tree.fit(X_sub, y_sub)

            # Appeding tree
            A.append(tree)
            A.append([X_sub, y_sub])
            return A
        if self.n_jobs == 1:
            t1_start = perf_counter()
            for n in tqdm(range(self.n_estimators)):
                # Data sampling
                X_sub = X.sample(frac=1, axis='rows', replace=True)
                y_sub = y[X_sub.index]
                X_sub = X_sub.reset_index(drop=True)
                y_sub = y_sub.reset_index(drop=True)

                # Using sampled data to learn tree
                tree = self.base_estimator(criterion=self.criterion)
                tree.fit(X_sub, y_sub)

                # Appending the trees
                self.trees.append(tree)
                self.datas.append([X_sub, y_sub])
            t1_end = perf_counter()
            print("n_jobs = 1, elapsed time:", t1_end-t1_start, "seconds")
            
        else:
            t1_start_m = perf_counter()
            result = Parallel(n_jobs=self.n_jobs, prefer = "threads")(
                delayed(fit_tree)(X, y)
                for i in range(self.n_estimators))
            t1_end_m = perf_counter()
            print(f"n_jobs = {self.n_jobs}, elapsed time:", t1_end_m-t1_start_m, "seconds")
            # print(result)
            for j in result:
                self.trees.append(j[0]) 
                self.datas.append(j[1])
                


    def predict(self, X):
        y_hat_total = None
        for i, tree in enumerate(self.trees):
            if y_hat_total is None:
                y_hat_total = pd.Series(tree.predict(X)).to_frame()
            else:
                y_hat_total[i] = tree.predict(X)
        return y_hat_total.mode(axis=1)[0]
   


    def plot(self, X, y):
        color = ["r", "b", "g"]
        Zs = []
        fig1, ax1 = plt.subplots(
            1, len(self.trees), figsize=(5*len(self.trees), 4))

        x_min, x_max = X[0].min(), X[0].max()
        y_min, y_max = X[1].min(), X[1].max()
        x_range = x_max-x_min
        y_range = y_max-y_min

        for i, tree in enumerate(self.trees):
            X_tree, y_tree = self.datas[i]

            xx, yy = np.meshgrid(np.arange(x_min-0.2, x_max+0.2, (x_range)/50),
                                 np.arange(y_min-0.2, y_max+0.2, (y_range)/50))

            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            Zs.append(Z)
            cs = ax1[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            fig1.colorbar(cs, ax=ax1[i], shrink=0.9)

            for y_label in y.unique():
                idx = y_tree == y_label
                id = list(y_tree.cat.categories).index(y_tree[idx].iloc[0])
                ax1[i].scatter(X_tree.loc[idx, 0], X_tree.loc[idx, 1], c=color[id],
                               cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                               label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()
        
        # if(self.n_jobs==1):
        #     plt.savefig('figures/BaggingClassifier_jobs1.png')
        # else:
        #     plt.savefig('figures/BaggingClassifier_manyjobs.png')


        # For Common surface
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        Zs = np.array(Zs)
        com_surface, _ = weighted_mode(Zs, np.ones(Zs.shape))
        cs = ax2.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        for y_label in y.unique():
            idx = y == y_label
            id = list(y.cat.categories).index(y[idx].iloc[0])
            ax2.scatter(X.loc[idx, 0], X.loc[idx, 1], c=color[id],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                        label="Class: "+str(y_label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend()
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(cs, ax=ax2, shrink=0.9)
        # if(self.n_jobs==1):
        #     plt.savefig('figures/Common_Decision_Surface_jobs1.png')
        # else:
        #     plt.savefig('figures/Common_Decision_Surface_manyjobs.png')
        return fig1, fig2