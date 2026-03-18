#pragma once

#include <cuda_runtime.h>
#include <map>
#include <iostream>

// ==========================================================================
//  CachedAllocator — reusable device memory pool for thrust algorithms
// ==========================================================================
//  Thrust algorithms (sort, reduce, copy_if, unique, etc.) internally
//  allocate temporary device memory via cudaMalloc/cudaFree on every call.
//  On Jetson Orin this is especially expensive due to the unified memory
//  allocator's higher latency, causing 10–40ms spikes.
//
//  This allocator caches freed blocks in a multimap keyed by size.
//  When thrust requests memory, we return a cached block if available,
//  otherwise fall back to cudaMalloc.  Blocks are only truly freed in
//  the destructor.
//
//  Usage with thrust:
//    CachedAllocator alloc;
//    thrust::sort(thrust::cuda::par(alloc).on(stream), ...);
// ==========================================================================
class CachedAllocator
{
public:
    using value_type = char;

    CachedAllocator() = default;

    ~CachedAllocator()
    {
        free_all();
    }

    // thrust calls this to allocate temporary memory
    char* allocate(std::ptrdiff_t num_bytes)
    {
        char* ptr = nullptr;

        // look for a cached block of at least num_bytes
        auto it = free_blocks.lower_bound(num_bytes);
        if (it != free_blocks.end()) {
            ptr = it->second;
            free_blocks.erase(it);
        } else {
            // no cached block — allocate new
            cudaError_t err = cudaMalloc(&ptr, num_bytes);
            if (err != cudaSuccess) {
                // try freeing cached blocks and retry
                free_all();
                err = cudaMalloc(&ptr, num_bytes);
                if (err != cudaSuccess) {
                    throw std::runtime_error("CachedAllocator: cudaMalloc failed");
                }
            }
        }

        // track the allocation size for later caching
        allocated_blocks[ptr] = num_bytes;
        return ptr;
    }

    // thrust calls this to free temporary memory — we cache it instead
    void deallocate(char* ptr, std::ptrdiff_t /*num_bytes*/)
    {
        auto it = allocated_blocks.find(ptr);
        if (it != allocated_blocks.end()) {
            std::ptrdiff_t size = it->second;
            allocated_blocks.erase(it);
            free_blocks.insert(std::make_pair(size, ptr));
        } else {
            // unknown block — just free it
            cudaFree(ptr);
        }
    }

private:
    // free all cached blocks (called in destructor or when OOM)
    void free_all()
    {
        for (auto& kv : free_blocks) {
            cudaFree(kv.second);
        }
        free_blocks.clear();
    }

    // size → device pointer (free list, sorted by size for lower_bound)
    std::multimap<std::ptrdiff_t, char*> free_blocks;

    // device pointer → allocated size (active allocations)
    std::map<char*, std::ptrdiff_t> allocated_blocks;
};
