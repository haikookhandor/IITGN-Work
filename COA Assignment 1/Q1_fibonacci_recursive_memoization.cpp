#include <bits/stdc++.h>
#include <sys/time.h>

using namespace std;

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

struct timespec;

time_t clk_1;
time_t clk_2;

__int128 arr[1000];

__int128 fibonacci_recursion_memoization(int n)
{
    if (n == 1)
    {
        return 0;
    }
    if (n == 2)
    {
        return 1;
    }
    if (arr[n] != 0)
    {
        return arr[n];
    }
    else
    {
        arr[n] = fibonacci_recursion_memoization(n - 1) + fibonacci_recursion_memoization(n - 2);
        return arr[n];
    }
}

int main()
{
    clk_1 = clock();

    for (int i = 1; i < 101; i++)
    {
        cout << fibonacci_recursion_memoization(i) << " ";
    }
    clk_2 = clock() - clk_1;
    float CPU = (float)clk_2 / CLOCKS_PER_SEC;
    cout << "The time taken by the program on CPU using timespec is:"
         << " ";
    cout << CPU << endl;
    float CPU_baseline = (float)clk_2 / CLOCKS_PER_SEC;
    return 0;
}
