import random as r

a=int(input())
arr1 = [[r.randint(-10,10) for i in range(a)] for j in range(a)]
arr2 = [[r.randint(-10,10) for i in range(a)] for j in range(a)]
final_array = [[0 for i in range(a)] for j in range(a)]

for i in range(a):
    for j in range(a):
        final_array[i][j]=0
        for k in range(a):
            final_array[i][j]+=arr1[i][k]*arr2[k][j]
            
for i in range(a):
    for j in range(a):
        print(final_array[i][j],end=" ")
    print("\n")

