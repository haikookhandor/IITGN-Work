#include<stdio.h>
#include<time.h>
#include<stdlib.h>

int main(){
    const int N = 256;
    double A[N][N];
    double B[N][N];
    double C[N][N];

    //initializing A, B
    //initialize C with all 0's
    for(int i = 0;i < N;i ++){
        for(int j = 0;j < N;j ++){
            A[i][j] = rand()/6.6;
            B[i][j] = rand()/2.31;
            C[i][j] = 0;
        }
    }

    struct timespec s1;
    struct timespec f1;
    char start1[100];
    char end1[100];
    timespec_get(&s1,TIME_UTC);

    //ijk
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            for(int k = 0; k < N; k++) {
                C[i][j]+=(A[i][k]*B[k][j]);
            }
        }
    }

    timespec_get(&f1,TIME_UTC);
    strftime(start1, sizeof start1, "%D %T", gmtime(&s1.tv_sec));    
    printf("%s.%09ld\n",start1,s1.tv_nsec);
    strftime(end1, sizeof end1, "%D %T", gmtime(&f1.tv_sec));    
    printf("%s.%09ld\n",end1,f1.tv_nsec);

    //initializing A, B
    //initialize C with all 0's

    for(int i = 0; i < N;i ++){
        for(int j = 0; j < N;j ++){
            A[i][j] = rand()/6.6;
            B[i][j] = rand()/2.31;
            C[i][j] = 0;
        }
    }

    struct timespec s2;
    struct timespec f2;
    char start2[100];
    char end2[100];
    timespec_get(&s2,TIME_UTC);

    //ikj
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < N; k++) {
            for(int j = 0; j < N; j++) {
                C[i][j]+=(A[i][k]*B[k][j]);
            }
        }
    }

    timespec_get(&f2,TIME_UTC);
    strftime(start2, sizeof start2, "%D %T", gmtime(&s2.tv_sec));    
    printf("%s.%09ld\n",start2,s2.tv_nsec);
    strftime(end2, sizeof end2, "%D %T", gmtime(&f2.tv_sec));    
    printf("%s.%09ld\n",end2,f2.tv_nsec);

    //initializing A, B
    //initialize C with all 0's

    for(int i = 0; i < N;i ++){
        for(int j = 0; j < N;j ++){
            A[i][j] = rand()/6.6;
            B[i][j] = rand()/2.31;
            C[i][j] = 0;
        }
    }

    struct timespec s3;
    struct timespec f3;
    char start3[100];
    char end3[100];
    timespec_get(&s3,TIME_UTC);

    //jik
    for(int j = 0; j < N; j++) {
        for(int i = 0; i < N; i++) {
            for(int k = 0; k < N; k++) {
                C[i][j]+=(A[i][k]*B[k][j]);
            }
        }
    }

    timespec_get(&f3,TIME_UTC);
    strftime(start3, sizeof start3, "%D %T", gmtime(&s3.tv_sec));    
    printf("%s.%09ld\n",start3,s3.tv_nsec);
    strftime(end3, sizeof end3, "%D %T", gmtime(&f3.tv_sec));    
    printf("%s.%09ld\n",end3,f3.tv_nsec);

    //initializing A, B
    //initialize C with all 0's

    for(int i = 0; i < N;i ++){
        for(int j = 0; j < N;j ++){
            A[i][j] = rand()/6.6;
            B[i][j] = rand()/2.31;
            C[i][j] = 0;
        }
    }

    struct timespec s4;
    struct timespec f4;
    char start4[100];
    char end4[100];
    timespec_get(&s4,TIME_UTC);

    //jki
    for(int j = 0; j < N; j++) {
        for(int k = 0; k < N; k++) {
            for(int i = 0; i < N; i++) {
                C[i][j]+=(A[i][k]*B[k][j]);
            }
        }
    }

    timespec_get(&f4,TIME_UTC);
    strftime(start4, sizeof start4, "%D %T", gmtime(&s4.tv_sec));    
    printf("%s.%09ld\n",start4,s4.tv_nsec);
    strftime(end4, sizeof end4, "%D %T", gmtime(&f4.tv_sec));    
    printf("%s.%09ld\n",end4,f4.tv_nsec);

    //initializing A, B
    //initialize C with all 0's

    for(int i = 0; i < N;i ++){
        for(int j = 0; j < N;j ++){
            A[i][j] = rand()/6.6;
            B[i][j] = rand()/2.31;
            C[i][j] = 0;
        }
    }

    struct timespec s5;
    struct timespec f5;
    char start5[100];
    char end5[100];
    timespec_get(&s5,TIME_UTC);

    //kij
    for(int k = 0; k < N; k++) {
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                C[i][j]+=(A[i][k]*B[k][j]);
            }
        }
    }

    timespec_get(&f5,TIME_UTC);
    strftime(start5, sizeof start5, "%D %T", gmtime(&s5.tv_sec));    
    printf("%s.%09ld\n",start5,s5.tv_nsec);
    strftime(end5, sizeof end5, "%D %T", gmtime(&f5.tv_sec));    
    printf("%s.%09ld\n",end5,f5.tv_nsec);

    //initializing A, B
    //initialize C with all 0's

    for(int i = 0; i < N;i ++){
        for(int j = 0; j < N;j ++){
            A[i][j] = rand()/6.6;
            B[i][j] = rand()/2.31;
            C[i][j] = 0;
        }
    }

    struct timespec s6;
    struct timespec f6;
    char start6[100];
    char end6[100];
    timespec_get(&s6,TIME_UTC);

    //kji
    for(int k = 0; k < N; k++) {
        for(int j = 0; j < N; j++) {
            for(int i = 0; i < N; i++) {
                C[i][j]+=(A[i][k]*B[k][j]);
            }
        }
    }

    timespec_get(&f6,TIME_UTC);
    strftime(start6, sizeof start6, "%D %T", gmtime(&s6.tv_sec));    
    printf("%s.%09ld\n",start6,s6.tv_nsec);
    strftime(end6, sizeof end6, "%D %T", gmtime(&f6.tv_sec));    
    printf("%s.%09ld\n",end6,f6.tv_nsec);

    return 0;
    
}