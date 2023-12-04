# AdaBoost

Plots for n estimators
<p align="left">
  <img width="400" src="figures\AdaBoostClassifier_estimators.png">
</p>

Common Decision Boundary
<p align="left">
  <img width="400" src="figures\AdaBoostClassifier_commonsurface.png">
</p>

Decision Tree Stump:(Max_depth = 1)

    Accuracy:  0.52
    Precision for  0 = 0.8125
    Recall for  0 = 0.08441558441558442
    Precision for  1 = 0.5035211267605634
    Recall for  1 = 0.9794520547945206

N_estimators: 

    Accuracy: 0.5566666666666666
    Precision for 0  :  0.92
    Recall for  0 :  0.14935064935064934
    Precision for 1  :  0.5236363636363637
    Recall for  1 :  0.9863013698630136

N_estimators has a higher accuracy although marginal but that is partly because of the data which is quite shuffled. But the grasp concept from this is that N_estimators performs better than Decision Tree Stump.

