#ifndef ALIGNEDMEMORY_HPP
#define ALIGNEDMEMORY_HPP

#include <memory>

// alloc a block of uninitialized memory for type T
// alignedPtr is aligned to memory boundary by parameter N
// for non POD type no initialization is done.
// N must be power of 2
template <class T, size_t N>
struct AlignedMemory {
    static_assert(sizeof(T) < N, "size of T must be smaller than N");

    // N is the alignment parameter. 32 for AVX
    T *alignedPtr = nullptr; // ptr to the aligned address of the buffer

    // constructor destructor
    AlignedMemory()
        : alignedPtr(nullptr), numberOfT(0), rawBytePtr(nullptr),
          rawByteSize(0) {}
    explicit AlignedMemory(size_t nT)
        : alignedPtr(nullptr), numberOfT(0), rawBytePtr(nullptr),
          rawByteSize(0) {
        resize(nT);
    }

    ~AlignedMemory() {
        if (rawBytePtr != nullptr) {
            std::free(rawBytePtr);
            rawBytePtr = nullptr;
        }
        alignedPtr = nullptr;
    }

    // forbid copy
    AlignedMemory(const AlignedMemory &) = delete;
    AlignedMemory(AlignedMemory &&) = delete;
    AlignedMemory &operator=(const AlignedMemory &) = delete;
    AlignedMemory &operator=(AlignedMemory &&) = delete;

    // no guarantee of the data already in the allocated memory
    void resize(size_t nT) {
        if (nT <= numberOfT) {
            return;
        }
        // get a large enough raw memory block
        rawByteSize = nT * sizeof(T) + N;
        if (rawBytePtr == nullptr) {
            rawBytePtr = std::malloc(
                rawByteSize); // std::malloc is required to be thread safe
        } else {
            rawBytePtr = std::realloc(rawBytePtr, rawByteSize);
            if (rawBytePtr == nullptr) {
                std::free(rawBytePtr);
                rawBytePtr = std::malloc(rawByteSize);
            }
        }
        if (rawBytePtr == nullptr) {
            printf("allocation fail\n");
            exit(1);
        }
        // return the aligned part
        // copy from Eigen/Memory/h
        alignedPtr =
            reinterpret_cast<T *>((reinterpret_cast<std::size_t>(rawBytePtr) &
                                   ~(std::size_t(N - 1))) +
                                  N);
        numberOfT = nT;
    }

  private:
    void *rawBytePtr = nullptr; // ptr to the whole buffer
    size_t rawByteSize = 0;     // number of Bytes
    size_t numberOfT = 0;
};

//
/* since c++11: http://en.cppreference.com/w/cpp/memory/c/aligned_alloc
The following functions are required to be thread-safe:

The library versions of operator new and operator delete
User replacement versions of global operator new and operator delete
std::calloc, std::malloc, std::realloc, std::aligned_alloc (since C++17)
Calls to these functions that allocate or deallocate a particular unit of storage occur in a single total order, and
each such deallocation call happens-before the next allocation (if any) in this order.

*/

#endif