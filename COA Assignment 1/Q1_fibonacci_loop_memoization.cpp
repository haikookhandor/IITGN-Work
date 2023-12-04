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

int main()
{
    clk_1 = clock();
    __int128 array[101];
    array[0] = 0;
    array[1] = 1;
    for (int i = 2; i < 100; i++)
    {
        array[i] = array[i - 1] + array[i - 2];
    }
    for (int j = 0; j < 100; j++)
    {
        cout << array[j] << " ";
    }
    clk_2 = clock() - clk_1;
    float CPU = (float)clk_2 / CLOCKS_PER_SEC;
    cout << "The time taken by the program on CPU using timespec is:"
         << " ";
    cout << CPU << endl;
    float CPU_baseline = (float)clk_2 / CLOCKS_PER_SEC;

    return 0;
}
