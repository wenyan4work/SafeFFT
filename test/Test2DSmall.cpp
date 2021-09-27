#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "../include/AlignedMemory.hpp"
#include "../include/SafeFFT.hpp"
#include "Timer.hpp"

#define TOTALSIZE 10
#define WORKNUMBER 10000

void benchSafeFFT() {
    using namespace safefft;
    const int workNumber = WORKNUMBER;
    // a list of FFTs to run
    // Will be used to obtain a seed for the random number engine
    std::random_device rd;
    // Standard mersenne_twister_engine seeded with rd()
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, TOTALSIZE);

    struct FFT {
        safefft::PlanFFT myPlan;
    };

    std::vector<FFT> work(workNumber);

    // prepare
    for (int i = 0; i < workNumber; i++) {
        const int workSize = 4 * dis(gen); // from 128 to 128*TOTALSIZE
        work[i].myPlan.n0 = workSize;
        work[i].myPlan.n1 = workSize;
        work[i].myPlan.nThreads = dis(gen) % 4 + 1; // 1 to 4 threads, nested omp threads
        work[i].myPlan.sign = 1;
    }

    // run
    Timer mytimer;
    mytimer.start();
#pragma omp parallel for
    for (int i = 0; i < workNumber; i++) {
        int tid = omp_get_thread_num();
        safefft::ComplexT *in = nullptr, *out = nullptr;
        // printf("%u,%u\n", in, out);
        SafeFFT::fitBuffer(work[i].myPlan, &in, nullptr, nullptr, &out, nullptr, nullptr); // contain garbage data
        // printf("%u,%u\n", in, out);

        // run 10 times
        for (int c = 0; c < 10; c++) {
            SafeFFT::runFFT(work[i].myPlan, in, nullptr, nullptr, out, nullptr, nullptr);
        }
    }
    mytimer.stop("FFT finished");
    mytimer.dump();
}

int main(int argc, char **argv) {
    safefft::SafeFFT::init();
    benchSafeFFT();
    safefft::SafeFFT::finalize();
    return 0;
}