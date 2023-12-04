import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 3


def data_gen(N,M,j):
    if j==1:                                            ## Data generation for Discrete input Discrete output
        X = pd.DataFrame({i:pd.Series(np.random.randint(N, size = M), dtype="category") for i in range(5)})     
        y = pd.Series(np.random.randint(N, size = M),  dtype="category")    

    elif j==2: 
        X = pd.DataFrame(np.random.randn(M, N))           ## Data generation for Real input Discrete output
        y = pd.Series(np.random.randint(N, size = M), dtype="category")

    elif j==3:                                            ## Data generation for Discrete input Real output
        X = pd.DataFrame({i:pd.Series(np.random.randint(N, size = M), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randn(M))

    else:                                              
        X = pd.DataFrame(np.random.randn(M, N))           ## Data generation for Real input Real output
        y = pd.Series(np.random.randn(M))
        
    return X, y


              ################                Case 1: Real Input Real Output      #####################

fit_inner_time = []
fit_time_mean = []
fit_time_std = []

predict_time_mean = []
predict_time_std = []
predict_inner_time = []

samples = []

for N in range(2,5):
    for M in range(10,50,10):
        X,y = data_gen(N,M,4)
        for num_time in range(num_average_time):
            clf = DecisionTree(criterion="information_gain")
            start_time = time.time()
            clf.fit(X,y)
            end_time = time.time()

            # np.append(fit_inner_time,end_time-start_time)
            fit_inner_time.append(end_time-start_time)

            start_time_p = time.time()
            y_pred = clf.predict(X)
            end_time_p = time.time()

            # np.append(predict_inner_time,end_time_p-start_time_p)
            predict_inner_time.append(end_time_p-start_time_p)

        # np.append(samples,N)
        samples.append([N,M])

        # print(predict_inner_time)

        fit_inner_time = np.array(fit_inner_time)
        predict_inner_time = np.array(predict_inner_time)

        # np.append(predict_time_mean,np.mean(predict_inner_time))
        predict_time_mean.append(predict_inner_time.mean())

        # np.append(predict_time_std,np.std(predict_inner_time))
        predict_time_std.append(predict_inner_time.std())

        # np.append(fit_time_mean,np.mean(fit_inner_time))
        fit_time_mean.append(fit_inner_time.mean())
        
        # np.append(fit_time_std,np.std(fit_inner_time))
        fit_time_std.append(fit_inner_time.std())

        fit_inner_time = []
        predict_inner_time = []



for i in range(len(samples)):
    print(f"sample: {samples[i][0]} x {samples[i][1]}, Mean fit time: {fit_time_mean[i]}, std: {fit_time_std[i]}")

for i in range(len(samples)):
    print(f"sample: {samples[i][0]} x {samples[i][1]}, Mean predict time: {predict_time_mean[i]}, the standard deviation is {predict_time_std[i]}")


plt.plot(list(fit_time_mean),label="Mean Fit time")
plt.title("Real Input Real Output")
plt.plot(list(fit_time_std),label="Std Fit time")
# for i in range(len(samples)):
#   plt.annotate(samples[i],(i,fit_time_mean[i]))

# for i in range(len(samples)):
#   plt.annotate(samples[i],(i,fit_time_std[i]))

plt.xlabel("Sequential Values of pair (N,M)")
plt.ylabel("Fit Time", fontsize=10)
plt.legend()
# plt.show()
plt.savefig("plots/fit_time_riro.png")
plt.clf()

plt.plot(list(predict_time_mean),label="Mean Predict time")
plt.plot(list(predict_time_std),label="Std Predict time")
# for i in range(len(samples)):
#   plt.annotate(samples[i],(i,predict_time_mean[i]))

# for i in range(len(samples)):
#   plt.annotate(samples[i],(i,predict_time_std[i]))

plt.xlabel("Sequential Values of pair (N,M)")
plt.title("Real Input Real Output")
plt.ylabel('"Predict time', fontsize=10)
plt.legend()
plt.savefig("plots/predict_time_riro.png")
plt.clf()
print("Finished RIRO!")
# plt.show()



              ################                Case 2: Real Input Discrete Output      #####################
print("Started RIDO!")
fit_inner_time_2 = []
fit_time_mean_2 = []
fit_time_std_2 = []

predict_time_mean_2 = []
predict_time_std_2 = []
predict_inner_time_2 = []

samples_2 = []


for N in range(2,5):
    for M in range(10,50,10):
        X,y = data_gen(N,M,2)
        print(N,M)
        for num_time in range(num_average_time):
            clf = DecisionTree(criterion="information_gain")
            start_time = time.time()
            clf.fit(X,y)
            end_time = time.time()

            # np.append(fit_inner_time_2,end_time-start_time)
            fit_inner_time_2.append(end_time-start_time)

            start_time_p = time.time()
            y_pred = clf.predict(X)
            end_time_p = time.time()

            # np.append(predict_inner_time_2,end_time_p-start_time_p)
            predict_inner_time_2.append(end_time_p-start_time_p)

        # np.append(samples_2,N)
        samples_2.append([N,M])

        # print(predict_inner_time_2)

        fit_inner_time_2 = np.array(fit_inner_time_2)
        predict_inner_time_2 = np.array(predict_inner_time_2)

        # np.append(predict_time_mean_2,np.mean(predict_inner_time_2))
        predict_time_mean_2.append(predict_inner_time_2.mean())

        # np.append(predict_time_std_2,np.std(predict_inner_time_2))
        predict_time_std_2.append(predict_inner_time_2.std())

        # np.append(fit_time_mean_2,np.mean(fit_inner_time_2))
        fit_time_mean_2.append(fit_inner_time_2.mean())
        
        # np.append(fit_time_std_2,np.std(fit_inner_time_2))
        fit_time_std_2.append(fit_inner_time_2.std())

        fit_inner_time_2 = []
        predict_inner_time_2 = []

print('\n')
for i in range(len(samples_2)):
    print(f"sample: {samples_2[i][0]} x {samples_2[i][1]}, case - 2, Mean fit time: {fit_time_mean_2[i]}, std: {fit_time_std_2[i]}")
print('\n')
for i in range(len(samples_2)):
    print(f"sample: {samples_2[i][0]} x {samples_2[i][1]}, case - 2, Mean predict time: {predict_time_mean_2[i]}, std: {predict_time_std_2[i]}")


plt.plot(list(fit_time_mean_2),label="Mean Fit time")
plt.plot(list(fit_time_std_2),label="Std Fit time")
# for i in range(len(samples_2)):
#   plt.annotate(samples_2[i],(i,fit_time_mean_2[i]))

# for i in range(len(samples_2)):
#   plt.annotate(samples_2[i],(i,fit_time_std_2[i]))

plt.xlabel("Sequential Values of pair (N,M)")
plt.title("Real Input Discrete Output")
plt.ylabel("Fit Time", fontsize=10)
plt.legend()
# plt.show()
plt.savefig("plots/fit_time_rido.png")
plt.clf()

plt.plot(list(predict_time_mean_2),label="Mean Predict time")
plt.plot(list(predict_time_std_2),label="Std Predict time")
# for i in range(len(samples_2)):
#   plt.annotate(samples_2[i],(i,predict_time_mean_2[i]))

# for i in range(len(samples_2)):
#   plt.annotate(samples_2[i],(i,predict_time_std_2[i]))

plt.xlabel("Sequential Values of pair (N,M)")
plt.title("Real Input Discrete Output")
plt.ylabel('"Predict time', fontsize=10)
plt.legend()
plt.savefig("plots/predict_time_rido.png")
plt.clf()
print("Finsihed RIDO!")



              ################                Case 3: Discrete Input Real Output      #####################
print("Started DIRO!")
fit_inner_time_3 = []
fit_time_mean_3 = []
fit_time_std_3 = []

predict_time_mean_3 = []
predict_time_std_3 = []
predict_inner_time_3 = []

samples_3 = []

for N in range(2,5):
    for M in range(10,50,10):
        X,y = data_gen(N,M,3)
        print(N,M)
        for num_time in range(num_average_time):
            clf = DecisionTree(criterion="gini_index")
            start_time = time.time()
            clf.fit(X,y)
            end_time = time.time()

            # np.append(fit_inner_time_3,end_time-start_time)
            fit_inner_time_3.append(end_time-start_time)

            start_time_p = time.time()
            y_pred = clf.predict(X)
            end_time_p = time.time()

            # np.append(predict_inner_time_3,end_time_p-start_time_p)
            predict_inner_time_3.append(end_time_p-start_time_p)

        # np.append(samples_3,N)
        samples_3.append([N,M])

        # print(predict_inner_time_3)

        fit_inner_time_3 = np.array(fit_inner_time_3)
        predict_inner_time_3 = np.array(predict_inner_time_3)

        # np.append(predict_time_mean_3,np.mean(predict_inner_time_3))
        predict_time_mean_3.append(predict_inner_time_3.mean())

        # np.append(predict_time_std_3,np.std(predict_inner_time_3))
        predict_time_std_3.append(predict_inner_time_3.std())

        # np.append(fit_time_mean_3,np.mean(fit_inner_time_3))
        fit_time_mean_3.append(fit_inner_time_3.mean())
        
        # np.append(fit_time_std_3,np.std(fit_inner_time_3))
        fit_time_std_3.append(fit_inner_time_3.std())

        fit_inner_time_3 = []
        predict_inner_time_3 = []


print('\n')
for i in range(len(samples_3)):
    print(f"sample: {samples_3[i][0]} x {samples_3[i][1]}, case - 3, Mean fit time: {fit_time_mean_3[i]}, std: {fit_time_std_3[i]}")


print('\n')
for i in range(len(samples_3)):
    print(f"sample: {samples_3[i][0]} x {samples_3[i][1]}, case -3, Mean predict time: {predict_time_mean_3[i]}, std: {predict_time_std_3[i]}")


plt.plot(list(fit_time_mean_3),label="Mean Fit time")
plt.plot(list(fit_time_std_3),label="Std Fit time")
# for i in range(len(samples_3)):
#   plt.annotate(samples_3[i],(i,fit_time_mean_3[i]))

# for i in range(len(samples_3)):
#   plt.annotate(samples_3[i],(i,fit_time_std_3[i]))

plt.xlabel("Sequential Values of pair (N,M)")
plt.title("Discrete Input Real Output")
plt.ylabel("Fit Time", fontsize=10)
plt.legend()
# plt.show()
plt.savefig("plots/fit_time_diro.png")
plt.clf()

plt.plot(list(predict_time_mean_3),label="Mean Predict time")
plt.plot(list(predict_time_std_3),label="Std Predict time")
# for i in range(len(samples_3)):
#   plt.annotate(samples_3[i],(i,predict_time_mean_3[i]))

# for i in range(len(samples_3)):
#   plt.annotate(samples_3[i],(i,predict_time_std_3[i]))

plt.xlabel("Sequential Values of pair (N,M)")
plt.title("Discrete Input Real Output")
plt.ylabel('"Predict time', fontsize=10)
plt.legend()
plt.savefig("plots/predict_time_diro.png")
plt.clf()
print("Finsihed DIRO!")



              ################                Case 4: Discrete Input Discrete Output      #####################
fit_inner_time_4 = []
fit_time_mean_4 = []
fit_time_std_4 = []
print("Starting DIDO!")
predict_time_mean_4 = []
predict_time_std_4 = []
predict_inner_time_4 = []

samples_4 = []

for N in range(2,5):
    for M in range(10,50,10):
        X,y = data_gen(N,M,1)
        print(N,M)
        for num_time in range(num_average_time):
            clf = DecisionTree(criterion="gini_index")
            start_time = time.time()
            clf.fit(X,y)
            end_time = time.time()

            # np.append(fit_inner_time_4,end_time-start_time)
            fit_inner_time_4.append(end_time-start_time)

            start_time_p = time.time()
            y_pred = clf.predict(X)
            end_time_p = time.time()

            # np.append(predict_inner_time_4,end_time_p-start_time_p)
            predict_inner_time_4.append(end_time_p-start_time_p)

        # np.append(samples_4,N)
        samples_4.append([N,M])

        # print(predict_inner_time_4)

        fit_inner_time_4 = np.array(fit_inner_time_4)
        predict_inner_time_4 = np.array(predict_inner_time_4)

        # np.append(predict_time_mean_4,np.mean(predict_inner_time_4))
        predict_time_mean_4.append(predict_inner_time_4.mean())

        # np.append(predict_time_std_4,np.std(predict_inner_time_4))
        predict_time_std_4.append(predict_inner_time_4.std())

        # np.append(fit_time_mean_4,np.mean(fit_inner_time_4))
        fit_time_mean_4.append(fit_inner_time_4.mean())
        
        # np.append(fit_time_std_4,np.std(fit_inner_time_4))
        fit_time_std_4.append(fit_inner_time_4.std())

        fit_inner_time_4 = []
        predict_inner_time_4 = []

for i in range(len(samples_4)):
    print(f"sample: {samples_4[i][0]} x {samples_4[i][1]}, case - 3, Mean predict time: {fit_time_mean_4[i]}, std: {fit_time_std_4[i]}")

print('\n')

for i in range(len(samples_4)):
    print(f"sample: {samples_4[i][0]} x {samples_4[i][1]}, case - 3, Mean predict time: {predict_time_mean_4[i]}, std: {predict_time_std_4[i]}")


plt.plot(list(fit_time_mean_4),label="Mean Fit time")
plt.plot(list(fit_time_std_4),label="Std Fit time")
# for i in range(len(samples_4)):
#   plt.annotate(samples_4[i],(i,fit_time_mean_4[i]))

# for i in range(len(samples_4)):
#   plt.annotate(samples_4[i],(i,fit_time_std_4[i]))

plt.xlabel("Sequential Values of pair (N,M)")
plt.title("Discrete Input Discrete Output")
plt.ylabel("Fit Time", fontsize=10)
plt.legend()
# plt.show()
plt.savefig("plots/fit_time_dido.png")
plt.clf()

plt.plot(list(predict_time_mean_4),label="Mean Predict time")
plt.plot(list(predict_time_std_4),label="Std Predict time")
# for i in range(len(samples_4)):
#   plt.annotate(samples_4[i],(i,predict_time_mean_4[i]))

# for i in range(len(samples_4)):
#   plt.annotate(samples_4[i],(i,predict_time_std_4[i]))

plt.xlabel("Sequential Values of pair (N,M)")
plt.title("Discrete Input Discrete Output")
plt.ylabel('"Predict time', fontsize=10)
plt.legend()
plt.savefig("plots/predict_time_dido.png")
plt.clf()
print("YAAY! Finsihed DIDO!")