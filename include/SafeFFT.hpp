#ifndef SAFEFFT_HPP
#define SAFEFFT_HPP

#include <cassert>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include <fftw3.h>
#include <omp.h>
#ifdef FFTW3_MKL
#include <fftw3_mkl.h>
#endif

#include "AlignedMemory.hpp"

#define DEFAULTALIGN 32  // number of bytes
#define DEFAULTSIZE 1024 // number of ComplexT

#define DEFAULTFLAG (FFTW_MEASURE | FFTW_DESTROY_INPUT)
// with this flag, the input array will be destroyed, and input and output must not overlap

namespace safefft {

using ComplexT = fftw_complex;
using ComplexPtrT = fftw_complex *;

struct PlanFFT {
    int n0 = 1, n1 = 1, n2 = 1;
    int sign = 1;
    unsigned int nThreads = 1;

    bool operator==(const PlanFFT &other) const {
        return (n0 == other.n0 && n1 == other.n1 && n2 == other.n2 && sign == other.sign && nThreads == other.nThreads);
    }
};

class Runner {
  public:
    Runner(size_t size) {
        // allocate aligned memory
        const int nMaxThreads = omp_get_max_threads();
        inbuf = new AlignedMemory<ComplexT, DEFAULTALIGN>[nMaxThreads];
        outbuf = new AlignedMemory<ComplexT, DEFAULTALIGN>[nMaxThreads];

        for (int i = 0; i < nMaxThreads; i++) {
            inbuf[i].resize(size);
            outbuf[i].resize(size);
        }

        readLockPtr = new omp_lock_t[nMaxThreads];
        for (int i = 0; i < nMaxThreads; i++) {
            omp_init_lock(&(readLockPtr[i]));
        }
        omp_init_lock(&writeLock);

        fftw_init_threads();
    }

    ~Runner() {
        const size_t nMaxThreads = omp_get_max_threads();
        // deallocation of buffer handled by destructor of AlignedMemory;
        if (inbuf) {
            delete[] inbuf;
        }
        if (outbuf) {
            delete[] outbuf;
        }
        // free fftw plan
        for (auto &p : planPool) {
            fftw_destroy_plan(p.second);
        }

        for (int i = 0; i < nMaxThreads; i++) {
            omp_destroy_lock(&(readLockPtr[i]));
        }
        omp_destroy_lock(&writeLock);
        delete[] readLockPtr;
    }

    // forbid copy
    Runner(const Runner &) = delete;
    const Runner &operator=(const Runner &) = delete;
    // forbid move
    Runner(Runner &&) = delete;
    const Runner &operator=(Runner &&) = delete;

    void runFFT(const PlanFFT &plan_, ComplexT *inPtr, ComplexT *outPtr) {
        // the user is responsible for aligned ment of inPtr and outPtr
        // must be aligned
        // must be large enough for the FFT specified in plan_

        const int tid = omp_get_num_threads();
        // run fft , number of threads is pre determined by the plan. OMP_NESTED should be true
        fftw_plan plan = getPlan(plan_);
#ifdef FFTW3_MKL
        mkl_set_num_threads_local(plan_.nThreads);
#endif
        fftw_execute_dft(plan, inPtr, outPtr);
    }

    void fitBuffer(const PlanFFT &plan, ComplexPtrT &in, ComplexPtrT &out) {
        const int tid = omp_get_thread_num();
        size_t nT = (plan.n0 + 1) * (plan.n1 + 1) * (plan.n2 + 1); // buffer is a bit larger
        inbuf[tid].resize(nT);
        outbuf[tid].resize(nT);
        in = inbuf[tid].alignedPtr;
        out = outbuf[tid].alignedPtr;
    }

    // allow only 1 thread running this
    inline fftw_plan getPlan(const PlanFFT &plan_) {
        const unsigned flag = DEFAULTFLAG;

        const int tid = omp_get_thread_num();
        const size_t nMaxThreads = omp_get_max_threads();

        const int n0 = plan_.n0;
        const int n1 = plan_.n1;
        const int n2 = plan_.n2;

        fftw_plan plan;

        // multiple reader
        omp_set_lock(&(readLockPtr[tid]));

        auto search = planPool.find(plan_);
        if (search != planPool.end()) {
            plan = search->second;
            omp_unset_lock(&(readLockPtr[tid]));
        } else {
            omp_unset_lock(&(readLockPtr[tid]));
            // actual write
            // create a new plan
            // ensure plan is large enough
            inbuf[tid].resize((n0) * (n1) * (n2));
            outbuf[tid].resize((n0) * (n1) * (n2));

            // use internal buffer to create the plan
            ComplexT *in = inbuf[tid].alignedPtr;
            ComplexT *out = outbuf[tid].alignedPtr;

            // only one writer
            omp_set_lock(&writeLock);
            // lock all readers
            for (int i = 0; i < nMaxThreads; i++) {
                omp_set_lock(&(readLockPtr[i]));
            }

#ifdef FFTW3_MKL // The document here is not accurate. the fftw_plan_with_nthreads() must be called
                 // https://software.intel.com/en-us/articles/how-to-set-number-of-users-threads-while-using-intel-mkl-fftw3-wrappers-3
            fftw3_mkl.number_of_user_threads = plan_.nThreads;
            fftw_plan_with_nthreads(plan_.nThreads);
#else
            fftw_plan_with_nthreads(plan_.nThreads);
#endif
            if (n1 == 1 && n2 == 1) {
                plan = fftw_plan_dft_1d(n0, in, out, plan_.sign, flag);
            } else if (n2 == 1) {
                plan = fftw_plan_dft_2d(n0, n1, in, out, plan_.sign, flag);
            } else {
                plan = fftw_plan_dft_3d(n0, n1, n2, in, out, plan_.sign, flag);
            }
            // put plan to planPool
            planPool[plan_] = plan;

            // release all reader lock, including self
            for (int i = 0; i < nMaxThreads; i++) {
                omp_unset_lock(&(readLockPtr[i]));
            }

            omp_unset_lock(&writeLock);
        }

        // printf("plan pool size %d\n", planPool.size());

        return plan;
    }

  private:
    // custom hash function for PlanFFT
    struct hashPlanFFT {
        std::size_t operator()(const PlanFFT &plan) const {
            using std::hash;
            using std::size_t;
            using std::string;

            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:

            return std::hash<int>{}(plan.sign * (plan.n0 + 1) * (plan.n1 + 1) * (plan.n2 + 1)) ^
                   (std::hash<size_t>{}(plan.nThreads) << 1);
        }
    };

    // storage of fftw_plan
    std::unordered_map<PlanFFT, fftw_plan, hashPlanFFT> planPool; // storage of all plans

    AlignedMemory<ComplexT, DEFAULTALIGN> *inbuf;  // each thread maintains its own buffer
    AlignedMemory<ComplexT, DEFAULTALIGN> *outbuf; // each thread maintains its own buffer
    omp_lock_t *readLockPtr;                       // multi reader, one writer
    omp_lock_t writeLock;
}; // namespace safefft

class SafeFFT {
  public:
    SafeFFT() { assert(runnerPtr); }

    ~SafeFFT() = default;

    // default copy
    SafeFFT(const SafeFFT &) = default;
    SafeFFT(SafeFFT &&) = default;
    SafeFFT &operator=(const SafeFFT &) = default;
    SafeFFT &operator=(SafeFFT &&) = default;

    void runFFT(const PlanFFT &plan, ComplexT *in, ComplexT *out) {
        assert(runnerPtr); // the runner has been allocated.
        runnerPtr->runFFT(plan, in, out);
    }

    // return in/out large enough for the plan
    // operate only on the buffer for this thread
    void fitBuffer(const PlanFFT &plan, ComplexPtrT &in, ComplexPtrT &out) { runnerPtr->fitBuffer(plan, in, out); }

    static void init() { runnerPtr = new Runner(DEFAULTSIZE); }

    static void finalize() { delete runnerPtr; }

  private:
    static Runner *runnerPtr;
};

Runner *SafeFFT::runnerPtr = nullptr;

} // namespace safefft
#endif