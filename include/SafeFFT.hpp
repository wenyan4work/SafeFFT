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

#define DEFAULTALIGN 32   // number of bytes
#define DEFAULTSIZE 1024  // number of ComplexT
#define DEFAULTPLANTIME 2 // 2 seconds of planning time
#define DEFAULTMAXRANK 3  // max 3 dimensional
#define DEFAULTSIGN (-1)  // sign of Fourier transform

#define DEFAULTFLAG (FFTW_MEASURE | FFTW_DESTROY_INPUT)
// with this flag, the input array will be destroyed, and input and output must not overlap

namespace safefft {

using ComplexT = fftw_complex;
using ComplexPtrT = fftw_complex *;

enum COMPLEXSTORAGE { INTERLEAVE, SPLIT };

enum FFTTYPE { C2C, R2C, C2R, R2R };

// simple FFT interface
struct PlanFFT {
    int n0 = 1, n1 = 1, n2 = 1;
    int sign = DEFAULTSIGN;
    unsigned int nThreads = 1;

    bool operator==(const PlanFFT &other) const {
        return (n0 == other.n0 && n1 == other.n1 && n2 == other.n2 && sign == other.sign && nThreads == other.nThreads);
    }
};

// for the guru interface
struct PlanGuruFFT {
    int rank = DEFAULTMAXRANK;
    fftw_iodim iodim[DEFAULTMAXRANK];
    int howmany_rank = DEFAULTMAXRANK;
    fftw_iodim howmany_dims[DEFAULTMAXRANK];
    int sign = DEFAULTSIGN;
    unsigned flags = DEFAULTFLAG;
    COMPLEXSTORAGE complex_storage = COMPLEXSTORAGE::INTERLEAVE;
    FFTTYPE fft_type = FFTTYPE::C2C;
    unsigned int nThreads = 1;

    // DEFAULT CONSTRUCTOR
    // TODO: clear up the meanings of iodim and howmany_dim
    PlanGuruFFT() {
        rank = DEFAULTMAXRANK;
        howmany_rank = DEFAULTMAXRANK;
        sign = DEFAULTSIGN;
        flags = DEFAULTFLAG;
        complex_storage = COMPLEXSTORAGE::INTERLEAVE;
        fft_type = FFTTYPE::C2C;
        for (int i = 0; i < DEFAULTMAXRANK; i++) {
            iodim[i].n = 1;
            iodim[i].is = 0;
            iodim[i].os = 0;
            howmany_dims[i].n = 1;
            howmany_dims[i].is = 0;
            howmany_dims[i].os = 0;
        }
        nThreads = 1;
    }
    // copy constructor
    PlanGuruFFT(const PlanGuruFFT &) = default;
    PlanGuruFFT(PlanGuruFFT &&) = default;
    PlanGuruFFT &operator=(const PlanGuruFFT &) = default;
    PlanGuruFFT &operator=(PlanGuruFFT &&) = default;

    // TODO:copy from SimplePlan
    PlanGuruFFT(const PlanFFT &simplePlan) {
        rank = 3;
        howmany_rank = 3;
        sign = simplePlan.sign;
        flags = DEFAULTFLAG;
        complex_storage = COMPLEXSTORAGE::INTERLEAVE;
        fft_type = FFTTYPE::C2C;
        iodim[0].n = simplePlan.n0;
        iodim[1].n = simplePlan.n1;
        iodim[2].n = simplePlan.n2;

        for (int i = 0; i < 3; i++) {
            iodim[i].is = 0;
            iodim[i].os = 0;
            // TODO: check this
            howmany_dims[i].n = 1;
            howmany_dims[i].is = 0;
            howmany_dims[i].os = 0;
        }
        nThreads = simplePlan.nThreads;
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
        fftw_set_timelimit(DEFAULTPLANTIME);
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
        clearPool();

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

    void clearPool() {
        for (auto &p : planPool) {
            fftw_destroy_plan(p.second);
        }
        planPool.clear();
    }

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
        // TODO: add other FFT types C2R,R2C,R2R,Interleave,split
        fftw_execute_dft(plan, inPtr, outPtr);
    }

    // TODO:run fft
    void runC2CInterleave(const PlanGuruFFT &planGuru, ComplexT **in, ComplexT **out) {
        const int tid = omp_get_thread_num();
        size_t nT = 1;
        for(int i=0;i<DEFAULTMAXRANK;i++){}

        inbuf[tid].resize(nT);
        outbuf[tid].resize(nT);
    }
    void runR2CInterleave(const PlanGuruFFT &planGuru, double **in, ComplexT **out);
    void runC2RInterleave(const PlanGuruFFT &planGuru, ComplexT **in, double **out);
    void runR2R(const PlanGuruFFT &planGuru, double **in, double **out);

    void runC2CSplit(const PlanGuruFFT &planGuru, double **inReal, double **inImag, double **outReal, double **outImag);
    void runR2CSplit(const PlanGuruFFT &planGuru, double **in, double **outReal, double **outImag);
    void runC2RSplit(const PlanGuruFFT &planGuru, double **inReal, double **inImag, double **outImag);

    // TODO: buffer allocating
    void fitBufferC2CInterleave(const PlanGuruFFT &planGuru, ComplexT **in, ComplexT **out);
    void fitBufferR2CInterleave(const PlanGuruFFT &planGuru, double **in, ComplexT **out);
    void fitBufferC2RInterleave(const PlanGuruFFT &planGuru, ComplexT **in, double **out);
    void fitBufferR2R(const PlanGuruFFT &planGuru, double **in, double **out);

    void fitBufferC2CSplit(const PlanGuruFFT &planGuru, double **inReal, double **inImag, double **outReal,
                           double **outImag);
    void fitBufferR2CSplit(const PlanGuruFFT &planGuru, double **in, double **outReal, double **outImag);
    void fitBufferC2RSplit(const PlanGuruFFT &planGuru, double **inReal, double **inImag, double **outImag);

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
            // TODO: plan with guru interface
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
            // TODO: hash for PlanGuruFFT struct using hash::combine
            return std::hash<int>{}(plan.sign * (plan.n0 + 1) * (plan.n1 + 1) * (plan.n2 + 1)) ^
                   (std::hash<size_t>{}(plan.nThreads) << 1);
        }
    };

    // storage of fftw_plan
    // TODO: using PlanGuruFFT as key
    std::unordered_map<PlanFFT, fftw_plan, hashPlanFFT> planPool; // storage of all plans

    AlignedMemory<ComplexT, DEFAULTALIGN> *inbuf;  // each thread maintains its own buffer
    AlignedMemory<ComplexT, DEFAULTALIGN> *outbuf; // each thread maintains its own buffer
    // TODO: allocate and deallocate the double pointers
    AlignedMemory<double, DEFAULTALIGN> *inbuf;  // each thread maintains its own buffer
    AlignedMemory<double, DEFAULTALIGN> *outbuf; // each thread maintains its own buffer

    omp_lock_t *readLockPtr; // multi reader, one writer
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

    // runFFT
    static void runFFT(const PlanFFT &plan, ComplexT *in, ComplexT *out) {
        assert(runnerPtr); // the runner has been allocated.
        runC2CInterleave(plan, in, out);
    }

    static void runC2CInterleave(const PlanGuruFFT &plan, ComplexT *in, ComplexT *out) {}
    static void runR2CInterleave(const PlanGuruFFT &plan, double *in, ComplexT *out);
    static void runC2RInterleave(const PlanGuruFFT &plan, ComplexT *in, double *out);
    static void runR2R(const PlanGuruFFT &plan, double *in, double *out);

    static void runC2CSplit(const PlanGuruFFT &plan, double *inReal, double *inImag, double *outReal, double *outImag);
    static void runR2CSplit(const PlanGuruFFT &plan, double *in, double *outReal, double *outImag);
    static void runC2RSplit(const PlanGuruFFT &plan, double *inReal, double *inImag, double *outImag);

    // return in/out large enough for the plan
    // operate only on the buffer for this thread
    static void fitBuffer(const PlanFFT &plan, ComplexT **in, ComplexT **out) { fitBufferC2CInterleave(plan, in, out); }

    static void fitBufferC2CInterleave(const PlanGuruFFT &plan, ComplexT **in, ComplexT **out) {
        runnerPtr->fitBufferC2CInterleave(plan, in, out);
    }
    static void fitBufferR2CInterleave(const PlanGuruFFT &plan, double **in, ComplexT **out) {
        runnerPtr->fitBufferR2CInterleave(plan, in, out);
    }
    static void fitBufferC2RInterleave(const PlanGuruFFT &plan, ComplexT **in, double **out) {
        runnerPtr->fitBufferC2RInterleave(plan, in, out);
    }
    static void fitBufferR2R(const PlanGuruFFT &plan, double **in, double **out) {
        runnerPtr->fitBufferR2R(plan, in, out);
    }

    static void fitBufferC2CSplit(const PlanGuruFFT &plan, double **inReal, double **inImag, double **outReal,
                                  double **outImag) {
        runnerPtr->fitBufferC2CSplit(plan, inReal, inImag, outReal, outImag);
    }
    static void fitBufferR2CSplit(const PlanGuruFFT &plan, double **in, double **outReal, double **outImag) {

        runnerPtr->fitBufferR2CSplit(plan, in, outReal, outImag);
    }
    static void fitBufferC2RSplit(const PlanGuruFFT &plan, double **inReal, double **inImag, double **out) {

        runnerPtr->fitBufferC2RSplit(plan, inReal, inImag, out);
    }

    // TODO: test clear the hash table
    static void clearPool() { runnerPtr->clearPool(); }

    static void init() { runnerPtr = new Runner(DEFAULTSIZE); }

    static void finalize() { delete runnerPtr; }

  private:
    static Runner *runnerPtr;
};

Runner *SafeFFT::runnerPtr = nullptr;

} // namespace safefft
#endif