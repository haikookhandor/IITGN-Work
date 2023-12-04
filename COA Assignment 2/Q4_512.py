import random
import time
N = 512

#initializing A, B
#initialize C with all 0's
A=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
B=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
C= [[0 for i in range(N)] for j in range(N)]

# ijk
start1=time.time()
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j]+=(A[i][k]*B[k][j])
end1=time.time()

#initializing A, B
#initialize C with all 0's
A=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
B=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
C= [[0 for i in range(N)] for j in range(N)]

# ikj
start2=time.time()
for i in range(N):
    for k in range(N):
        for j in range(N):
            C[i][j]+=(A[i][k]*B[k][j])
end2=time.time()

#initializing A, B
#initialize C with all 0's
A=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
B=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
C= [[0 for i in range(N)] for j in range(N)]

# jik
start3=time.time()
for j in range(N):
    for i in range(N):
        for k in range(N):
            C[i][j]+=(A[i][k]*B[k][j])
end3=time.time()

#initializing A, B
#initialize C with all 0's
A=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
B=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
C= [[0 for i in range(N)] for j in range(N)]

# jki
start4=time.time()
for i in range(N):
    for j in range(N):
        for k in range(N):
            C[i][j]+=(A[i][k]*B[k][j])
end4=time.time()

#initializing A, B
#initialize C with all 0's
A=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
B=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
C= [[0 for i in range(N)] for j in range(N)]

# kij
start5=time.time()
for k in range(N):
    for i in range(N):
        for j in range(N):
            C[i][j]+=(A[i][k]*B[k][j])
end5=time.time()

#initializing A, B
#initialize C with all 0's
A=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
B=[[random.randint(-1000000,1000000)*0.3 for i in range(N)] for j in range(N)]
C= [[0 for i in range(N)] for j in range(N)]

# kji
start6=time.time()
for k in range(N):
    for j in range(N):
        for i in range(N):
            C[i][j]+=(A[i][k]*B[k][j])
end6=time.time()


print("Execution time for order ijk is",end1-start1)
print("Execution time for order ikj is",end2-start2)
print("Execution time for order jik is",end3-start3)
print("Execution time for order jki is",end4-start4)
print("Execution time for order kij is",end5-start5)
print("Execution time for order kji is",end6-start6)
