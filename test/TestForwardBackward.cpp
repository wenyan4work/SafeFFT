#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "../include/AlignedMemory.hpp"
#include "../include/SafeFFT.hpp"
#include "Timer.hpp"

void testSafeFFT() {
    using namespace safefft;

    // a list of FFTs to run
    const int workNumber = 10000000;
    std::vector<safefft::PlanFFT> work(workNumber);
    int size[6] = {256, 512, 1024, 384, 128, 768}; // a large data set

    double error = 0;

#pragma omp parallel for reduction(max : error)
    for (int i = 0; i < workNumber; i++) {
        // create forward and backward plan
        safefft::PlanFFT forward, backward;
        forward.n0 = size[(i + 1) % 6];
        forward.nThreads = 1; // 1-4 threads, to test nested OMP
        forward.sign = FFTW_FORWARD;

        backward.n0 = forward.n0;
        backward.nThreads = forward.nThreads;
        backward.sign = FFTW_BACKWARD;

        // get in out buffer
        safefft::ComplexT *in, *out;
        safefft::SafeFFT::fitBuffer(forward, &in, nullptr, nullptr, &out, nullptr, nullptr);

        for (int i = 0; i < forward.n0; i++) {
            in[i][0] = i;
            in[i][1] = 0;
        }

        // forward
        SafeFFT::runFFT(forward, in, nullptr, nullptr, out, nullptr, nullptr);

        // backward
        SafeFFT::runFFT(backward, out, nullptr, nullptr, in, nullptr, nullptr);

        // compare error of real part
        error = 0;
        for (int k = 0; k < forward.n0; k++) {
            error = std::max(error, fabs(in[k][0] / forward.n0 - (double)(k)));
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