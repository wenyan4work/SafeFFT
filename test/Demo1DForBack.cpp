#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "../include/SafeFFT.hpp"

#include "Timer.hpp"

int main() {
    using namespace safefft;
    SafeFFT::init();
    const int workNumber = 10000000; // 10million small FFTs
#pragma omp parallel for
    for (int i = 0; i < workNumber; i++) {
        PlanFFT forward, backward;
        forward.n0 = pow(2, 5 + (i + 1) % 6);
        forward.sign = FFTW_FORWARD;

        ComplexT *in, *out;
        SafeFFT::fitBuffer(forward, &in, nullptr, nullptr, &out, nullptr,
                           nullptr);

        for (int i = 0; i < forward.n0; i++)
            in[i][0] = in[i][1] = i;

        SafeFFT::runFFT(forward, in, nullptr, nullptr, out, nullptr, nullptr);
    }
    SafeFFT::finalize();
    return 0;
}