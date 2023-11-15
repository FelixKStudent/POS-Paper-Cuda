#include <iostream>
#include <vector>
#include <chrono>

void vecAdd(std::vector<float>& a, std::vector<float>& b, std::vector<float>& c, int n)
{
    for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
}

void fillVecs(std::vector<float>& a, std::vector<float>& b, int n)
{
    for (int i = 0; i < n; i++) {
        a.push_back(static_cast<float>(i));
        // example: fill a with values 0, 1, 2, ..., n-1
        b.push_back(static_cast<float>(2 * i));
        // example: fill b with values 0, 2, 4, ..., 2n-2
    }
}

int main()
{
    const int N = 100000000;

    std::vector<float> A, B, C;
    A.reserve(N);
    B.reserve(N);
    C.resize(N);

    fillVecs(A, B, N);

    auto start = std::chrono::high_resolution_clock::now();
    vecAdd(A, B, C, N);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    return 0;
}