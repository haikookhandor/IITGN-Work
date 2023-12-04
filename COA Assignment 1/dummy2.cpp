#include <iostream>
#include <ctime>
#define N 100

using namespace std;

const __int128 NIL = -1;
__int128 lookup_table[N];

void init()
{
    for (__int128 i = 0; i < N; i++)
        lookup_table[i] = NIL;
}

__int128 fib_mem(int n)
{
    if (lookup_table[n] == NIL)
    {
        if (n <= 1)
            lookup_table[n] = n;
        else
            lookup_table[n] = fib_mem(n - 1) + fib_mem(n - 2);
    }
    return lookup_table[n];
}

std::ostream &
operator<<(std::ostream &dest, __int128_t value)
{
    std::ostream::sentry s(dest);
    if (s)
    {
        __uint128_t tmp = value < 0 ? -value : value;
        char buffer[256];
        char *d = std::end(buffer);
        do
        {
            --d;
            *d = "0123456789"[tmp % 10];
            tmp /= 10;
        } while (tmp != 0);
        if (value < 0)
        {
            --d;
            *d = '-';
        }
        int len = std::end(buffer) - d;
        if (dest.rdbuf()->sputn(d, len) != len)
        {
            dest.setstate(std::ios_base::badbit);
        }
    }
    return dest;
}
int main()
{
    init();
    timespec start, end;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    int x;
    cout << "Enter the number of terms of series : ";
    cin >> x;
    cout << "\nFibonnaci Series : " << fib_mem(x) << endl;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    double elapsed = seconds + nanoseconds * 1e-9;
    cout << "\nTime taken: " << elapsed << " seconds\n";
}