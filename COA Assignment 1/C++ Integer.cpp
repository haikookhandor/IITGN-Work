#include <bits/stdc++.h>
#include <sys/time.h>

using namespace std;

struct timespec;

time_t clock_1;
time_t clock_2;

int main()
{
    clock_1 = clock();

    srand(time(0));
    int a;
    cin >> a;
    int arr1[a][a];
    int arr2[a][a];
    int final_array[a][a];

    // we have divided by 10 so that the values of the matrix are from 0 to 9

    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < a; j++)
        {
            arr1[i][j] = (rand() % (10));
        }
    }

    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < a; j++)
        {
            arr2[i][j] = (rand() % (10));
        }
    }

    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < a; j++)
        {
            final_array[i][j] = 0;
            for (int k = 0; k < a; k++)
            {
                final_array[i][j] += arr1[i][k] * arr2[k][j];
            }
        }
    }

    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < a; j++)
        {
            cout << final_array[i][j] << "\t";
        }
        cout << "\n";
    }

    clock_2 = clock() - clock_1;
    float time = (float)clock_2 / CLOCKS_PER_SEC;
    cout << "The time taken is:"
         << " " << time << "\n";

    return 0;
}