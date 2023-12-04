from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree as sktree


class AdaBoostClassifier():
    def __init__(self, base_estimator=DecisionTreeClassifier, n_estimators=5,
                 max_depth=1, criterion="entropy"):
        self.base_estimator = base_estimator
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.trees = []
        self.alphas = []

    def fit(self, X, y):
        weights = np.ones(len(y))/len(y)
        for n in tqdm(range(self.n_estimators)):
            # Predicting and Learning the tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y, sample_weight=weights)
            y_hat = pd.Series(tree.predict(X))

            # Error and calculation of alpha
            mis_idx = y_hat != y
            err_m = np.sum(weights[mis_idx])/np.sum(weights)
            alpha_m = 0.5*np.log((1-err_m)/err_m)

            # Updation of weights
            weights[mis_idx] *= np.exp(alpha_m)
            weights[~mis_idx] *= np.exp(-alpha_m)
            
            # Weights are normalized
            weights /= np.sum(weights)

            self.trees.append(tree)
            self.alphas.append(alpha_m)

    def predict(self, X):
        final = np.zeros(X.shape[0])


        for i, (alpha_m, tree) in enumerate(zip(self.alphas, self.trees)):
            final += pd.Series(tree.predict(X))*alpha_m
        return np.sign(final)

    def plot(self, X, y):
        color = ["r", "b", "g"]
        Zs = None
        fig1, ax1 = plt.subplots(
            1, len(self.trees), figsize=(5*len(self.trees), 4))

        x_min, x_max = X.iloc[:, 0].min(), X.iloc[:, 0].max()
        y_min, y_max = X.iloc[:, 1].min(), X.iloc[:, 1].max()
        x_range = x_max-x_min
        y_range = y_max-y_min

        for i, (alpha_m, tree) in enumerate(zip(self.alphas, self.trees)):
            print("-----------------------------")
            print("Tree Number: {}".format(i+1))
            print("-----------------------------")
            print(sktree.export_text(tree))
            xx, yy = np.meshgrid(np.arange(x_min-0.2, x_max+0.2, (x_range)/50),
                                 np.arange(y_min-0.2, y_max+0.2, (y_range)/50))

            ax1[i].set_ylabel("X2")
            ax1[i].set_xlabel("X1")
            Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            if Zs is None:
                Zs = alpha_m*Z
            else:
                Zs += alpha_m*Z
            cs = ax1[i].contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
            fig1.colorbar(cs, ax=ax1[i], shrink=0.9)
            for y_label in y.unique():
                idx = (y == y_label)
                ax1[i].scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                               label="Class: "+str(y_label))
            ax1[i].set_title("Decision Surface Tree: " + str(i+1))
            ax1[i].legend()
        fig1.tight_layout()
        plt.savefig('figures/AdaBoostClassifier_estimators.png')
        
        
        # For Common surface
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 4))
        com_surface = np.sign(Zs)
        cs = ax2.contourf(xx, yy, com_surface, cmap=plt.cm.RdYlBu)
        for y_label in y.unique():
            idx = y == y_label
            ax2.scatter(X[idx].iloc[:, 0], X[idx].iloc[:, 1], 
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=30,
                        label="Class: "+str(y_label))
        ax2.set_ylabel("X2")
        ax2.set_xlabel("X1")
        ax2.legend(loc="lower right")
        ax2.set_title("Common Decision Surface")
        fig2.colorbar(cs, ax=ax2, shrink=0.9)
        plt.savefig('figures/AdaBoostClassifier_Commonsurface.png')        
        return fig1, fig2