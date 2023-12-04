import timeit
import random as rand

d = { 1:32, 2:64, 3:128, 4:256, 5:512 }

a = int(input())

number = d[a]

arr1 = [[rand.randint(0,100) for i in range(number)] for j in range(number)]
arr2 = [[rand.randint(0,100) for t in range(number)] for l in range(number)]
final_array = [[0 for m in range(number)] for p in range(number)]

start = timeit.default_timer()
for i in range(number):
    for j in range(number):
        for k in range(number):
            final_array[i][j] += arr1[i][k]*arr2[k][j]

stop = timeit.default_timer()

print("Time : ", stop-start)