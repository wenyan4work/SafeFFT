#ifndef SAFEFFT_HPP
#define SAFEFFT_HPP

#include <cassert>
#include <cstring>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <fftw3.h>
#include <omp.h>
#ifdef FFTW3_MKL
#include <fftw3_mkl.h>
#endif

#include "AlignedMemory.hpp"

#define DEFAULTALIGN 32            // number of bytes
#define DEFAULTSIZE 1024           // number of ComplexT
#define DEFAULTPLANTIME 2          // 2 seconds of planning time
#define DEFAULTMAXRANK 3           // max 3 dimensional
#define DEFAULTSIGN (FFTW_FORWARD) // sign of Fourier transform

#define DEFAULTFLAG (FFTW_MEASURE | FFTW_DESTROY_INPUT)
// with this flag, the input array will be destroyed, and input and output must not overlap

namespace safefft {

using ComplexT = fftw_complex;

// INTER for interleave strorage, SPLIT for split storage
enum FFTTYPE { C2CINTER, C2CSPLIT, R2CINTER, R2CSPLIT, C2RINTER, C2RSPLIT, R2R };

// simple FFT interface
struct PlanFFT {
    int n0 = 1, n1 = 1, n2 = 1;
    int sign = DEFAULTSIGN;
    unsigned int nThreads = 1;
};

// For other types, define a plan struct and define a copy constructor in PlanGuruFFT

// for the guru interface
struct PlanGuruFFT {
    int rank = DEFAULTMAXRANK;
    fftw_iodim dims[DEFAULTMAXRANK];

    int howmany_rank = DEFAULTMAXRANK;
    fftw_iodim howmany_dims[DEFAULTMAXRANK];

    fftw_r2r_kind r2rKind[DEFAULTMAXRANK];

    int sign = DEFAULTSIGN;
    unsigned flags = DEFAULTFLAG;

    FFTTYPE fft_type = FFTTYPE::C2CINTER; // default to C2C interleave

    unsigned int nThreads = 1;

    // DEFAULT CONSTRUCTOR
    // CHECK: clear up the meanings of iodim and howmany_dim
    PlanGuruFFT() {
        rank = DEFAULTMAXRANK;
        howmany_rank = DEFAULTMAXRANK;
        sign = DEFAULTSIGN;
        flags = DEFAULTFLAG;
        fft_type = FFTTYPE::C2CINTER;
        for (int i = 0; i < DEFAULTMAXRANK; i++) {
            dims[i].n = 1;          // dimension in this rank
            dims[i].is = 1;         // input stride
            dims[i].os = 1;         // output strid
            howmany_dims[i].n = 1;  // number of ffts in this rank
            howmany_dims[i].is = 0; // input distance of each fft in this rank
            howmany_dims[i].os = 0; // output distance of each fft in this rank
            r2rKind[i] = FFTW_R2HC; // not used in most cases except for r2r transforms
        }
        nThreads = 1;
    }
    // copy constructor
    PlanGuruFFT(const PlanGuruFFT &) = default;
    PlanGuruFFT(PlanGuruFFT &&) = default;
    PlanGuruFFT &operator=(const PlanGuruFFT &) = default;
    PlanGuruFFT &operator=(PlanGuruFFT &&) = default;

    bool operator==(const PlanGuruFFT &other) const {
        if (rank == other.rank && howmany_rank == other.howmany_rank && sign == other.sign && flags == other.flags &&
            fft_type == other.fft_type && nThreads == other.nThreads) {
            bool result = true;
            for (int i = 0; i < DEFAULTMAXRANK; i++) {
                result = result && (dims[i].n == other.dims[i].n);
                result = result && (dims[i].is == other.dims[i].is);
                result = result && (dims[i].os == other.dims[i].os);
                result = result && (howmany_dims[i].n == other.howmany_dims[i].n);
                result = result && (howmany_dims[i].is == other.howmany_dims[i].is);
                result = result && (howmany_dims[i].os == other.howmany_dims[i].os);
                result = result && (r2rKind[i] == other.r2rKind[i]);
                if (result == false) {
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    }

    // CHECK: check copy from SimplePlan , single 3D C2C
    PlanGuruFFT(const PlanFFT &simplePlan) : PlanGuruFFT() { // first set the default values
        sign = simplePlan.sign;
        flags = DEFAULTFLAG;
        fft_type = FFTTYPE::C2CINTER;
        nThreads = simplePlan.nThreads;

        // FFTW manual page 35: for a single transform, set howmany_rank=0
        howmany_rank = 0;

        rank = 3;
        dims[0].n = simplePlan.n0;
        dims[1].n = simplePlan.n1;
        dims[2].n = simplePlan.n2;


        // set stride for row-major array
        // Ref FFTW manual apge 35
        dims[2].is = 1;
        dims[2].os = 1;
        dims[1].is = dims[2].n * dims[2].is;
        dims[1].os = dims[2].n * dims[2].os;
        dims[0].is = dims[1].n * dims[1].is;
        dims[0].os = dims[1].n * dims[1].os;
    }

    size_t getSingleSize() const {
        // the size large enough to hold the input (or output) data
        // for C2C, this is the actual size needed
        // For R2C/C2R transforms, this is LARGER than the actual size needed
        // accurate size depends on the underlying actual lib used (fftw, mkl)
        // determining the actual accurate size is tediouly complicated
        size_t size = 1;
        for (int r = 0; r < rank; r++) {
            size = size * dims[r].n;
        }
        return size;
    }

    size_t getTotalSize() const {
        size_t size = getSingleSize();
        for (int r = 0; r < howmany_rank; r++) {
            size = size * howmany_dims[r].n;
        }
        return size;
    }
};

class Runner {
  public:
    Runner(size_t size) {
// just a safe guard
#pragma omp single
        {
            // allocate aligned memory
            const int nMaxThreads = omp_get_max_threads();
            inBufC = new AlignedMemory<ComplexT, DEFAULTALIGN>[nMaxThreads];
            outBufC = new AlignedMemory<ComplexT, DEFAULTALIGN>[nMaxThreads];
            inBufDReal = new AlignedMemory<double, DEFAULTALIGN>[nMaxThreads];
            outBufDReal = new AlignedMemory<double, DEFAULTALIGN>[nMaxThreads];
            inBufDImag = new AlignedMemory<double, DEFAULTALIGN>[nMaxThreads];
            outBufDImag = new AlignedMemory<double, DEFAULTALIGN>[nMaxThreads];

            for (int i = 0; i < nMaxThreads; i++) {
                inBufC[i].resize(size);
                outBufC[i].resize(size);
                inBufDReal[i].resize(size);
                outBufDReal[i].resize(size);
                inBufDImag[i].resize(size);
                outBufDImag[i].resize(size);
            }

            readLockPtr = new omp_lock_t[nMaxThreads];
            for (int i = 0; i < nMaxThreads; i++) {
                omp_init_lock(&(readLockPtr[i]));
            }
            omp_init_lock(&writeLock);
            fftw_init_threads();
            fftw_set_timelimit(DEFAULTPLANTIME);
        }
    }

    ~Runner() {
        const size_t nMaxThreads = omp_get_max_threads();
        // deallocation of buffer handled by destructor of AlignedMemory;
        if (inBufC) {
            delete[] inBufC;
        }
        if (outBufC) {
            delete[] outBufC;
        }

        if (inBufDReal) {
            delete[] inBufDReal;
        }
        if (outBufDReal) {
            delete[] outBufDReal;
        }

        if (inBufDImag) {
            delete[] inBufDImag;
        }
        if (outBufDImag) {
            delete[] outBufDImag;
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

    // CHECK:run fft
    void runFFT(const PlanGuruFFT &planGuru, ComplexT *in = nullptr, double *inReal = nullptr, double *inImag = nullptr,
                ComplexT *out = nullptr, double *outReal = nullptr, double *outImag = nullptr) {
        // the user is responsible for aligned ment of inPtr and outPtr
        // must be aligned
        // must be large enough for the FFT specified in plan_

        const int tid = omp_get_num_threads();
        // run fft , number of threads is pre determined by the plan. OMP_NESTED should be true
        fftw_plan plan = getPlan(planGuru);
#ifdef FFTW3_MKL
        mkl_set_num_threads_local(planGuru.nThreads);
#endif
        // execute with new array interface
        switch (planGuru.fft_type) {
        case FFTTYPE::C2CINTER:
            fftw_execute_dft(plan, in, out);
            break;
        case FFTTYPE::C2CSPLIT:
            fftw_execute_split_dft(plan, inReal, inImag, outReal, outImag);
            break;
        case FFTTYPE::R2CINTER:
            fftw_execute_dft_r2c(plan, inReal, out);
            break;
        case FFTTYPE::R2CSPLIT:
            fftw_execute_split_dft_r2c(plan, inReal, outReal, outImag);
            break;
        case FFTTYPE::C2RINTER:
            fftw_execute_dft_c2r(plan, in, outReal);
            break;
        case FFTTYPE::C2RSPLIT:
            fftw_execute_split_dft_c2r(plan, inReal, inImag, outReal);
            break;
        case FFTTYPE::R2R:
            fftw_execute_r2r(plan, inReal, outReal);
            break;
        }
    }

    // CHECK: buffer allocating
    void fitBuffer(const PlanGuruFFT &planGuru, ComplexT **in = nullptr, double **inReal = nullptr,
                   double **inImag = nullptr, ComplexT **out = nullptr, double **outReal = nullptr,
                   double **outImag = nullptr) {
        const int tid = omp_get_thread_num();

        size_t inC = 0;      // numer of elements of type C or D. not size of bytes
        size_t inDReal = 0;  // numer of elements of type C or D. not size of bytes
        size_t inDImag = 0;  // numer of elements of type C or D. not size of bytes
        size_t outC = 0;     // numer of elements of type C or D. not size of bytes
        size_t outDReal = 0; // numer of elements of type C or D. not size of bytes
        size_t outDImag = 0; // numer of elements of type C or D. not size of bytes

        const size_t totalSize = planGuru.getTotalSize();

        switch (planGuru.fft_type) {
        case FFTTYPE::C2CINTER: {
            inC = totalSize;
            outC = totalSize;
        } break;
        case FFTTYPE::C2CSPLIT: {
            inDReal = totalSize;
            inDImag = totalSize;
            outDReal = totalSize;
            outDImag = totalSize;
        } break;
        case FFTTYPE::R2CINTER: {
            inDReal = totalSize;
            outC = totalSize;
        } break;
        case FFTTYPE::R2CSPLIT: {
            inDReal = totalSize;
            outDReal = totalSize;
            outDImag = totalSize;
        } break;
        case FFTTYPE::C2RINTER: {
            inC = totalSize;
            outDReal = totalSize;
        } break;
        case FFTTYPE::C2RSPLIT: {
            inC = totalSize;
            outDReal = totalSize;
            outDImag = totalSize;
        } break;
        case FFTTYPE::R2R: {
            inDReal = totalSize;
            outDReal = totalSize;
        } break;
        }

        inBufC[tid].resize(inC);           // the buffer is never shrinked, passing in 0 is OK
        outBufC[tid].resize(outC);         // the buffer is never shrinked, passing in 0 is OK
        inBufDReal[tid].resize(inDReal);   // the buffer is never shrinked, passing in 0 is OK
        inBufDImag[tid].resize(inDImag);   // the buffer is never shrinked, passing in 0 is OK
        outBufDReal[tid].resize(outDReal); // the buffer is never shrinked, passing in 0 is OK
        outBufDImag[tid].resize(outDImag); // the buffer is never shrinked, passing in 0 is OK

        // return buffer needed
        if (in)
            *in = inBufC[tid].alignedPtr;
        if (out)
            *out = outBufC[tid].alignedPtr;
        if (inReal)
            *inReal = inBufDReal[tid].alignedPtr;
        if (inImag)
            *inImag = inBufDImag[tid].alignedPtr;
        if (outReal)
            *outReal = outBufDReal[tid].alignedPtr;
        if (outImag)
            *outImag = outBufDImag[tid].alignedPtr;

        return;
    }

    // allow only 1 thread running this
    inline fftw_plan getPlan(const PlanGuruFFT &planGuru, ComplexT *in = nullptr, double *inReal = nullptr,
                             double *inImag = nullptr, ComplexT *out = nullptr, double *outReal = nullptr,
                             double *outImag = nullptr) {
        // WARNING: The arrays passed by the pointers are used to create the plan.
        // The arrays returned by the fitBuffer() routine should be fine.

        const unsigned flag = DEFAULTFLAG;
        const int tid = omp_get_thread_num();
        const size_t nMaxThreads = omp_get_max_threads();

        fftw_plan plan;

        // multiple reader
        omp_set_lock(&(readLockPtr[tid]));

        auto search = planPool.find(planGuru);
        if (search != planPool.end()) {
            plan = search->second;
            omp_unset_lock(&(readLockPtr[tid]));
        } else {
            omp_unset_lock(&(readLockPtr[tid]));
            // actual write
            // create a new plan

            // only one writer
            omp_set_lock(&writeLock);
            // lock all readers
            for (int i = 0; i < nMaxThreads; i++) {
                omp_set_lock(&(readLockPtr[i]));
            }

#ifdef FFTW3_MKL // The document here is not accurate. the fftw_plan_with_nthreads() must be called
                 // https://software.intel.com/en-us/articles/how-to-set-number-of-users-threads-while-using-intel-mkl-fftw3-wrappers-3
            fftw3_mkl.number_of_user_threads = planGuru.nThreads;
            fftw_plan_with_nthreads(planGuru.nThreads);
#else
            fftw_plan_with_nthreads(planGuru.nThreads);
#endif
            // execute with new array interface
            auto &rank = planGuru.rank;
            auto &dims = planGuru.dims;
            auto &howmany_rank = planGuru.howmany_rank;
            auto &howmany_dims = planGuru.howmany_dims;
            auto &sign = planGuru.sign;
            auto &flags = planGuru.flags;
            auto &r2rKind = planGuru.r2rKind;

            switch (planGuru.fft_type) {
            case FFTTYPE::C2CINTER:
                plan = fftw_plan_guru_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
                break;
            case FFTTYPE::C2CSPLIT:
                // ref FFTW manual p36, there is no sign parameter in guru_split_dft. always in FFTW_FORWARD mode
                plan = fftw_plan_guru_split_dft(rank, dims, howmany_rank, howmany_dims, inReal, inImag, outReal,
                                                outImag, flags);
                break;
            case FFTTYPE::R2CINTER:
                // p.37 r2c always FORWARD
                plan = fftw_plan_guru_dft_r2c(rank, dims, howmany_rank, howmany_dims, inReal, out, flags);
                break;
            case FFTTYPE::R2CSPLIT:
                // p.37 r2c always FORWARD
                plan = fftw_plan_guru_split_dft_r2c(rank, dims, howmany_rank, howmany_dims, inReal, outReal, outImag,
                                                    flags);
                break;
            case FFTTYPE::C2RINTER:
                // p.37 c2r always BACKWARD
                plan = fftw_plan_guru_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, outReal, flags);
                break;
            case FFTTYPE::C2RSPLIT:
                // p.37 c2r always BACKWARD
                plan = fftw_plan_guru_split_dft_c2r(rank, dims, howmany_rank, howmany_dims, inReal, inImag, outReal,
                                                    flags);
                break;
            case FFTTYPE::R2R:
                // p.37 for r2r there is FORWARD or BACKWARD. directly specify fftw_r2r_kind
                plan = fftw_plan_guru_r2r(rank, dims, howmany_rank, howmany_dims, inReal, outReal, r2rKind, flags);
                break;
            }

            // put plan to planPool
            planPool[planGuru] = plan;

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
    struct hashPlanGuruFFT {
        std::size_t operator()(const PlanGuruFFT &plan) const {
            using std::hash;

            size_t result = hash<int>{}(plan.fft_type);
            hashCombine(result, hash<int>{}(plan.rank));
            hashCombine(result, hash<int>{}(plan.howmany_rank));
            hashCombine(result, hash<int>{}(plan.sign));
            hashCombine(result, hash<unsigned>{}(plan.flags));
            hashCombine(result, hash<unsigned>{}(plan.nThreads));
            for (int i = 0; i < DEFAULTMAXRANK; i++) {
                hashCombine(result, hash<unsigned>{}(plan.dims[i].n));
                hashCombine(result, hash<unsigned>{}(plan.dims[i].is));
                hashCombine(result, hash<unsigned>{}(plan.dims[i].os));
                hashCombine(result, hash<unsigned>{}(plan.howmany_dims[i].n));
                hashCombine(result, hash<unsigned>{}(plan.howmany_dims[i].is));
                hashCombine(result, hash<unsigned>{}(plan.howmany_dims[i].os));
            }
            fftw_iodim howmany_dims[DEFAULTMAXRANK];

            unsigned int nThreads = 1;

            return result;
        }

        // combine hash
        void hashCombine(size_t &lhs, const size_t &rhs) const { lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2); }
    };

    // storage of fftw_plan
    // CHECK: using PlanGuruFFT as key
    std::unordered_map<PlanGuruFFT, fftw_plan, hashPlanGuruFFT> planPool; // storage of all plans

    AlignedMemory<ComplexT, DEFAULTALIGN> *inBufC;  // each thread maintains its own buffer
    AlignedMemory<ComplexT, DEFAULTALIGN> *outBufC; // each thread maintains its own buffer
    // CHECK: allocate and deallocate the double pointers
    AlignedMemory<double, DEFAULTALIGN> *inBufDReal;  // each thread maintains its own buffer
    AlignedMemory<double, DEFAULTALIGN> *outBufDReal; // each thread maintains its own buffer
    AlignedMemory<double, DEFAULTALIGN> *inBufDImag;  // each thread maintains its own buffer
    AlignedMemory<double, DEFAULTALIGN> *outBufDImag; // each thread maintains its own buffer

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

    // CHECK:run fft
    static void runFFT(const PlanGuruFFT planGuru, ComplexT *in = nullptr, double *inReal = nullptr,
                       double *inImag = nullptr, ComplexT *out = nullptr, double *outReal = nullptr,
                       double *outImag = nullptr) {
        // a planGuru is always constructed by copy, depending on the plan passed in
        assert(runnerPtr); // the runner has been allocated.
        runnerPtr->runFFT(planGuru, in, inReal, inImag, out, outReal, outImag);
        return;
    }

    // CHECK: buffer allocating
    static void fitBuffer(const PlanGuruFFT planGuru, ComplexT **in = nullptr, double **inReal = nullptr,
                          double **inImag = nullptr, ComplexT **out = nullptr, double **outReal = nullptr,
                          double **outImag = nullptr) {
        assert(runnerPtr); // the runner has been allocated.
        runnerPtr->fitBuffer(planGuru, in, inReal, inImag, out, outReal, outImag);
        return;
    }

    static void init() { runnerPtr = new Runner(DEFAULTSIZE); }

    static void finalize() { delete runnerPtr; }

  private:
    static Runner *runnerPtr;
}; // namespace safefft

Runner *SafeFFT::runnerPtr = nullptr;

} // namespace safefft
#endif