#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "../include/AlignedMemory.hpp"
#include "../include/SafeFFT.hpp"
#include "Timer.hpp"

void testSafeFFT() {
    safefft::SafeFFT myFFT;
    const int workNumber = 10000000;
    // a list of FFTs to run
    std::vector<safefft::PlanFFT> work(workNumber);
    int size[6] = {256, 512, 1024, 384, 128, 768}; // a large data set

    double error = 0;

#pragma omp parallel for reduction(max : error)
    for (int i = 0; i < workNumber; i++) {
        // create forward and backward plan
        safefft::PlanFFT forward, backward;
        forward.n0 = size[(i + 1) % 6];
        forward.nThreads = 1; // 1-4 threads, to test nested OMP
        forward.sign = 1;

        backward.n0 = forward.n0;
        backward.nThreads = forward.nThreads;
        backward.sign = -1;

        // get in out buffer
        safefft::ComplexPtrT in, out;
        myFFT.fitBuffer(forward, in, out);

        for (int i = 0; i < forward.n0; i++) {
            in[i][0] = i + 1;
            in[i][1] = i * 2;
        }

        // forward
        myFFT.runFFT(forward, in, out);

        // backward
        myFFT.runFFT(backward, out, in);

        // compare error of real part
        error = 0;
        for (int k = 0; k < forward.n0; k++) {
            error = std::max(error, fabs(in[k][0] / forward.n0 - (double)(k + 1)));
        }
#ifndef NDEBUG
#pragma omp critical
        { printf("size %u, error %g, %g, %g\n", forward.n0, error, (double)in[1][0], (double)in[1][1]); }
#endif
    }

    printf("max error %g \n", error);
}

int main() {
    safefft::SafeFFT::init();
    testSafeFFT();
    safefft::SafeFFT::finalize();
    return 0;
}