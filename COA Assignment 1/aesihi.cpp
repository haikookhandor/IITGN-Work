#include <iostream>
#include <time.h>
using namespace std;

int main()
{

    struct timespec start, end;

    timespec_get(&start, TIME_UTC);
    int time_taken;
    double long t1 = 0, t2 = 1, t3, i;
    cout << t1 << " " << t2 << " ";
    for (i = 2; i < 100; ++i)
    {
        t3 = t1 + t2;
        cout << t3 << " ";
        t1 = t2;
        t2 = t3;
    }
    timespec_get(&end, TIME_UTC);

    time_taken = (end.tv_sec - start.tv_sec) * 1e9;
    time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;

    cout << "The time taken is:" << time_taken;

    return 0;
}