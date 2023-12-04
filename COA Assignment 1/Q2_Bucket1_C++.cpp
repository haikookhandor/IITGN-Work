#include <iostream>
#include <stdlib.h>
#include <ctime>
using namespace std;

int main()
{
    srand(time(0));
    int a;
    cin >> a;
    int arr1[a][a];
    int arr2[a][a];
    int final_array[a][a];
    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < a; j++)
        {
            arr1[i][j] = (rand() % (11));
        }
    }
    for (int i = 0; i < a; i++)
    {
        for (int j = 0; j < a; j++)
        {
            arr2[i][j] = (rand() % (11));
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

    return 0;
}
