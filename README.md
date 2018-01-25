# SafeFFT: A thread safe simple c++ wrapper for FFTW & MKL

## Objective
In FFTW3 (or MKL) the only thread safe functions are `fftw_execute_...()`.
This sometimes poses problems on the structure of multithreading code, where each thread may need to perform FFTs with different `fftw_plan`. 
This simple wrapper around FFTW3 aims at making things easier by maintain a global hash table of `fftw_plan`, where each thread may insert new plans and read already allocated plans. 
The hash table is locked such that multiple readers can access it but only one thread can insert new entry.
This allows multiple threads to reuse already allocated plan simultaneously, without allocating a new plan everytime.
I believe this approach has some performance and design advantage because allocating a new plan everytime for every thread requires a mutex lock to allow only one thread to create a plan.

This simple wrapper fits a case where a large number of FFTs must be processed, but the total number of different FFT plans are not that large, and it may also be hard to preallocate all possible plans before running any FFTs. 
In other words, the size of the hash map of plans is expected to be much smaller than the total number of FFTs to run, so that reusing a plan benefits performance.
In general I expect the size of the hash map to be on the order of 10 ~ 1000.

The hash map is implemented with a simple std::unordered_map, and guarded with a customized & naive multi-reader, single writer locking system implemented with `omp_lock`.
It is naively implemented because there is no things like `std::shared_mutex` in openmp, and also mixing openmp threads with facilities from pthreads or `std::thread`, or `boost::thread` is not clear to me whether it is safe to do so.
For the same reason, I am not using a concurrent unordered_map like the one from Intel TBB or other concurrent containers like Junction.

## Usage scenario
1. All threads share one SafeFFT object to process many FFTs in parallel (see `test/TestForwardBackward.cpp`)
2. Every objet can have its own SafeFFT object to process its FFT, and then many such objects are partitioned through `#omp parallel for` to execute FFT simultaneously (see `test/Test1DSmall.cpp` and other)
3. In any cases, the member functions of `SafeFFT` are supposed to be called from the root level openmp thread team. Otherwise, the thread id returned by `omp_get_thread_num()` may not be meaningful, which may cause problems when locating the per thread buffer and internal locks.  

## Implementation Notes
1. It is header only. Put `SafeFFT.hpp` and `AlignedMemory.hpp` anywhere and it should work.
2. All SafeFFT object share one Runner object, which does the real work. Only one Runner instance should exist throughout the program.
3. The safeFFT object stores only one pointer. Its memory footprint is supposed to be small and it could be declared as a member for the user's class. 
4. Each thread maintains its own aligned in/out buffer memory. 
5. Link with either `libfftw3_omp` or `mkl`
6. If using MKL, define the macro FFTW3_MKL for threading control
7. If using MKL, calling `runFFT()` from multiple threads needs careful threading control. The only working setting I figured out is: a. Setting `OMP_NUM_THREADS` to control the number of threads calling `runFFT()`. b. setting `mkl_set_num_threads_local(plan_.nThreads);` in `runFFT()` to ensure the number of threads to execute this plan. c. Setting `OMP_NESTED=true` and `MKL_DYNAMIC=false`. DO NOT set `MKL_NUM_THREADS`. 
8. Nested threading is implemented for flexibility. For small FFTs (even those in Test1DLarge.cpp), sticking to `plan.nThreads=1` gives better performance. On my 12-core Xeon, the program Test1DLarge shows around 95% efficiency with 10 cores. It is the user's duty to think about it and get good performance.
9. `fftw3_threads` uses pthread rather than openmp thread (at least in my understanding). In my tests on a Mac the nested threading control still works well.

## Possible Issues
1. I cannot guarantee the naive implementation of RWLock with omp_lock is optimal or even correct, although it works fine in my tests.
2. The effect of mixing with other thread models is unknown.

## Possible Extensions 
1. The code is probably ugly and I cannot guarentee it is bug free. Comments, forks, and bug reports are welcome.
2. The code currently implements interfaces for simple 1D/2D/3D complex DFTs only. If you need other transforms like r2c,c2r, etc, the code should be fairly simple to extend.
