/**
 * @file allocator.c
 * @brief GPU memory allocator with OOM (Out-Of-Memory) checking
 * 
 * Implements the memory allocation tracking and OOM prevention system.
 * All CUDA memory allocations go through this module, which:
 * - Checks if allocation would exceed the configured memory limit
 * - Tracks allocations in shared memory for multi-process coordination
 * - Maintains lists of allocated chunks for proper cleanup
 * 
 * SLURM-specific: Memory limits are enforced across all processes in a job
 * via shared memory coordination.
 * 
 * @authors Rahim Khoja, Karim Ali
 * @organization Research Computing, University of Alberta
 * @note This is a HAMi-core fork with SLURM-specific changes for Alliance clusters
 */

#include "allocator.h"
#include "include/log_utils.h"
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"

// Allocation size constants
size_t BITSIZE = 512;
size_t IPCSIZE = 2097152;
size_t OVERSIZE = 134217728;
//int pidfound;

region_list *r_list;
allocated_list *device_overallocated;
allocated_list *device_allocasync;

#define ALIGN       2097152
#define MULTI_PARAM 1

#define CHUNK_SIZE  (OVERSIZE/BITSIZE)
#define __CHUNK_SIZE__  CHUNK_SIZE

extern size_t initial_offset;
extern CUresult
    cuMemoryAllocate(CUdeviceptr* dptr, size_t bytesize, void* data);
extern CUresult cuMemoryFree(CUdeviceptr dptr);

pthread_once_t allocator_allocate_flag = PTHREAD_ONCE_INIT;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * @brief Round up a size to the nearest multiple of unit
 * 
 * Used for memory alignment. If size is not already a multiple of unit,
 * rounds up to the next multiple.
 * 
 * @param size Size to round up
 * @param unit Alignment unit (must be power of 2)
 * @return Rounded up size
 */
size_t round_up(size_t size, size_t unit) {
    if (size & (unit-1))
        return ((size / unit) + 1 ) * unit;
    return size;
}

/**
 * @brief Check if allocating additional memory would exceed the limit
 * 
 * Performs out-of-memory (OOM) check before allocation. This is the core
 * enforcement mechanism that prevents processes from exceeding their GPU
 * memory slice. The check:
 * - Gets current total memory usage across ALL processes in the job (via shared memory)
 * - Adds the requested allocation size
 * - Compares against the configured limit
 * - Returns 1 if limit would be exceeded, 0 otherwise
 * 
 * SLURM-specific: Aggregates memory usage across all processes in the same
 * SLURM job via the shared memory region, ensuring the total doesn't exceed
 * the job's allocated slice.
 * 
 * @param dev CUDA device index (-1 to use current context device)
 * @param addon Size of memory to allocate (in bytes)
 * @return 1 if allocation would exceed limit, 0 if OK
 */
int oom_check(const int dev, size_t addon) {
    // Ensure shared memory region is initialized before checking OOM
    ensure_initialized();
    
    int count1=0;
    CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetCount,&count1);
    CUdevice d;
    if (dev==-1)
        cuCtxGetDevice(&d);
    else
        d=dev;
    // Convert CUDA device index to NVML device index for memory tracking functions
    // Memory tracking uses NVML device indices (see add_gpu_device_memory_usage)
    // Default mapping is identity (CUDA 0 -> NVML 0) until map_cuda_visible_devices() is called
    int nvml_dev = cuda_to_nvml_map(d);
    if (nvml_dev < 0 || nvml_dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("oom_check: Invalid NVML device index %d from CUDA device %d (pid=%d)", nvml_dev, d, getpid());
        // Fallback to identity mapping for safety
        nvml_dev = (d >= 0 && d < CUDA_DEVICE_MAX_COUNT) ? d : 0;
    }
    uint64_t limit = get_current_device_memory_limit(nvml_dev);
    size_t _usage = get_gpu_memory_usage(nvml_dev);

    if (limit == 0) {
        return 0;
    }

    size_t new_allocated = _usage + addon;
    // Log OOM check details
    LOG_INFO("oom_check: pid=%d CUDA_dev=%d NVML_dev=%d usage=%lu limit=%lu addon=%lu new_total=%lu", 
             getpid(), d, nvml_dev, _usage, limit, addon, new_allocated);
    LOG_DEBUG("oom_check: Current usage breakdown: _usage=%lu (from get_gpu_memory_usage), addon=%lu, limit=%lu", 
              _usage, addon, limit);
    if (new_allocated > limit) {
        LOG_ERROR("oom_check: Device %d (NVML %d) OOM! %lu / %lu (pid=%d, exceeded by %lu bytes)", 
                  d, nvml_dev, new_allocated, limit, getpid(), new_allocated - limit);

        if (clear_proc_slot_nolock(1) > 0)
            return oom_check(dev,addon);
        return 1;
    }
    return 0;
}

/**
 * @brief Debug function to view current allocator state
 * 
 * Logs information about overallocated memory chunks and current device
 * memory usage. Useful for debugging memory limit issues.
 * 
 * @return CUDA_SUCCESS
 */
CUresult view_vgpu_allocator() {
    allocated_list_entry *al;
    size_t total;
    total=0;
    LOG_INFO("[view1]:overallocated:");
    for (al=device_overallocated->head;al!=NULL;al=al->next){
        LOG_INFO("(%p %lu)\t",(void *)al->entry->address,al->entry->length);
        total+=al->entry->length;
    }
    LOG_INFO("total=%lu",total);
    size_t t = get_current_device_memory_usage(0);
    LOG_INFO("current_device_memory_usage:%lu",t);
    return 0;
}

/**
 * @brief Calculate total size of all chunks in an allocation list
 * 
 * Iterates through the linked list of allocated chunks and sums their sizes.
 * Used for tracking total allocated memory.
 * 
 * @param al Allocation list to measure
 * @param size Output parameter for total size
 * @return CUDA_SUCCESS
 */
CUresult get_listsize(allocated_list *al, size_t *size) {
    if (al->length == 0){
        *size = 0;
        return CUDA_SUCCESS;
    }
    size_t count=0;
    allocated_list_entry *val;
    for (val=al->head;val!=NULL;val=val->next){
        count+=val->entry->length;
    }
    *size = count;
    return CUDA_SUCCESS;
}

/**
 * @brief Initialize the memory allocator
 * 
 * Sets up the data structures for tracking GPU memory allocations:
 * - device_overallocated: List of chunks allocated via cuMemAlloc (synchronous)
 * - device_allocasync: List of chunks allocated via cuMemAllocAsync (asynchronous)
 * - Initializes the mutex for thread-safe operations
 * 
 * Called once during postInit() after CUDA initialization.
 */
void allocator_init() {
    LOG_DEBUG("Allocator_init\n");
    
    device_overallocated = malloc(sizeof(allocated_list));
    LIST_INIT(device_overallocated);
    device_allocasync=malloc(sizeof(allocated_list));
    LIST_INIT(device_allocasync);

    pthread_mutex_init(&mutex,NULL);
}

/**
 * @brief Allocate a new GPU memory chunk and add it to tracking
 * 
 * Performs OOM check, allocates memory via CUDA, and tracks it in the
 * overallocated list. For small allocations (<= IPCSIZE), uses standard
 * cuMemAlloc. For larger allocations, uses custom allocation logic.
 * 
 * SLURM-specific: Updates shared memory with the new allocation so other
 * processes in the job can see the total memory usage.
 * 
 * @param address Output parameter for allocated device pointer
 * @param size Size to allocate in bytes
 * @return 0 on success, CUDA_ERROR_OUT_OF_MEMORY if limit exceeded
 */
int add_chunk(CUdeviceptr *address, size_t size) {
    size_t addr=0;
    size_t allocsize;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev,size))
        return CUDA_ERROR_OUT_OF_MEMORY;
    
    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    if (size <= IPCSIZE)
        res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAlloc_v2,&e->entry->address,size);
    else{
        e->entry->length = size;
        res = cuMemoryAllocate(&e->entry->address, size, e->entry->allocHandle);
    }
    if (res!=CUDA_SUCCESS){
        LOG_ERROR("cuMemoryAllocate failed res=%d",res);
        return res;
    }
    LIST_ADD(device_overallocated,e);
    //uint64_t t_size;
    *address = e->entry->address;
    allocsize = size;
    cuCtxGetDevice(&dev);
    add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
    return 0;
}

/**
 * @brief Add an already-allocated chunk to tracking (no actual allocation)
 * 
 * Used when memory is allocated outside our allocator but we still need
 * to track it for OOM checking. Just adds the chunk to the tracking list
 * and updates shared memory usage.
 * 
 * @param address Device pointer of already-allocated memory
 * @param size Size of the allocation
 * @return 0 on success, -1 on failure
 */
int add_chunk_only(CUdeviceptr address, size_t size) {
    pthread_mutex_lock(&mutex);
    size_t addr=0;
    size_t allocsize;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev,size)){
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    LIST_ADD(device_overallocated,e);
    //uint64_t t_size;
    e->entry->address=address;
    allocsize = size;
    cuCtxGetDevice(&dev);
    add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
    pthread_mutex_unlock(&mutex);
    return 0;
}

/**
 * @brief Check if an address belongs to tracked device memory
 * 
 * Searches through the overallocated list to see if the address falls
 * within any tracked allocation. Used to distinguish device vs host memory.
 * 
 * @param address Memory address to check
 * @return CU_MEMORYTYPE_DEVICE if found in tracked allocations, CU_MEMORYTYPE_HOST otherwise
 */
int check_memory_type(CUdeviceptr address) {
    allocated_list_entry *cursor;
    cursor = device_overallocated->head;
    for (cursor=device_overallocated->head;cursor!=NULL;cursor=cursor->next){
        if ((cursor->entry->address <= address) && (cursor->entry->address+cursor->entry->length>=address))
            return CU_MEMORYTYPE_DEVICE;
    }
    return CU_MEMORYTYPE_HOST;
}

/**
 * @brief Remove a chunk from tracking and free the memory
 * 
 * Finds the chunk in the list, frees it via CUDA, removes it from tracking,
 * and updates shared memory to reflect the freed memory.
 * 
 * @param a_list Allocation list to remove from
 * @param dptr Device pointer to free
 * @return 0 on success, -1 if chunk not found
 */
int remove_chunk(allocated_list *a_list, CUdeviceptr dptr) {
    size_t t_size;
    if (a_list->length==0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val=a_list->head;val!=NULL;val=val->next){
        if (val->entry->address == dptr) {
            t_size=val->entry->length;
            cuMemoryFree(dptr);
            LIST_REMOVE(a_list,val);
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(), dev, t_size, 2);
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Remove a chunk from tracking without freeing memory
 * 
 * Used when memory is freed outside our allocator but we need to stop
 * tracking it. Removes from list and updates shared memory usage.
 * 
 * @param dptr Device pointer to remove from tracking
 * @return 0 on success, -1 if chunk not found
 */
int remove_chunk_only(CUdeviceptr dptr) {
    allocated_list *a_list = device_overallocated;
    size_t t_size;
    if (a_list->length == 0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            t_size = val->entry->length;
            LIST_REMOVE(a_list, val);
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(), dev, t_size, 2);
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Thread-safe wrapper for add_chunk
 * 
 * Locks the allocator mutex, calls add_chunk, then unlocks.
 * Used by CUDA memory allocation hooks.
 * 
 * @param dptr Output parameter for allocated device pointer
 * @param size Size to allocate
 * @return 0 on success, error code on failure
 */
int allocate_raw(CUdeviceptr *dptr, size_t size) {
    int tmp;
    pthread_mutex_lock(&mutex);
    tmp = add_chunk(dptr, size);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

/**
 * @brief Thread-safe wrapper for remove_chunk
 * 
 * Locks the allocator mutex, calls remove_chunk, then unlocks.
 * Used by CUDA memory free hooks.
 * 
 * @param dptr Device pointer to free
 * @return 0 on success, -1 on failure
 */
int free_raw(CUdeviceptr dptr) {
    pthread_mutex_lock(&mutex);
    unsigned int tmp = remove_chunk(device_overallocated, dptr);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

/**
 * @brief Remove an async allocation chunk and free it
 * 
 * Similar to remove_chunk but for asynchronous allocations. Frees via
 * cuMemFreeAsync and updates the async allocation list limit.
 * 
 * @param a_list Async allocation list
 * @param dptr Device pointer to free
 * @param hStream CUDA stream for async operation
 * @return 0 on success, -1 if chunk not found
 */
int remove_chunk_async(
    allocated_list *a_list, CUdeviceptr dptr, CUstream hStream) {
    size_t t_size;
    if (a_list->length == 0) {
        return -1;
    }
    allocated_list_entry *val;
    for (val = a_list->head; val != NULL; val = val->next) {
        if (val->entry->address == dptr) {
            t_size=val->entry->length;
            CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemFreeAsync,dptr,hStream);
            LIST_REMOVE(a_list,val);
            a_list->limit-=t_size;
            CUdevice dev;
            cuCtxGetDevice(&dev);
            rm_gpu_device_memory_usage(getpid(),dev,t_size,2);
            return 0;
        }
    }
    return -1;
}

/**
 * @brief Thread-safe wrapper for remove_chunk_async
 * 
 * @param dptr Device pointer to free
 * @param hStream CUDA stream for async operation
 * @return 0 on success, -1 on failure
 */
int free_raw_async(CUdeviceptr dptr, CUstream hStream) {
    pthread_mutex_lock(&mutex);
    unsigned int tmp = remove_chunk_async(device_allocasync, dptr, hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}

/**
 * @brief Allocate memory asynchronously and track it
 * 
 * Performs OOM check, allocates via cuMemAllocAsync, and handles memory
 * pool limits. Tracks the allocation in the async list.
 * 
 * @param address Output parameter for allocated device pointer
 * @param size Size to allocate
 * @param hStream CUDA stream for async operation
 * @return 0 on success, -1 if limit exceeded or allocation failed
 */
int add_chunk_async(CUdeviceptr *address, size_t size, CUstream hStream) {
    size_t addr=0;
    size_t allocsize;
    CUresult res = CUDA_SUCCESS;
    CUdevice dev;
    cuCtxGetDevice(&dev);
    if (oom_check(dev,size))
        return -1;

    allocated_list_entry *e;
    INIT_ALLOCATED_LIST_ENTRY(e,addr,size);
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemAllocAsync,&e->entry->address,size,hStream);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemoryAllocate failed res=%d",res);
        return res;
    }
    *address = e->entry->address;
    CUmemoryPool pool;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetMemPool,&pool,dev);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuDeviceGetMemPool failed res=%d",res);
        return res;
    }
    size_t poollimit;
    res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuMemPoolGetAttribute,pool,CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,&poollimit);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuMemPoolGetAttribute failed res=%d",res);
        return res;
    }
    if (poollimit != 0) {
        if (poollimit> device_allocasync->limit) {
            allocsize = (poollimit-device_allocasync->limit < size)? poollimit-device_allocasync->limit : size;
            cuCtxGetDevice(&dev);
            add_gpu_device_memory_usage(getpid(), dev, allocsize, 2);
            device_allocasync->limit=device_allocasync->limit+allocsize;
            e->entry->length=allocsize;
        }else{
            e->entry->length=0;
        } 
    }
    LIST_ADD(device_allocasync,e);
    return 0;
}

/**
 * @brief Thread-safe wrapper for add_chunk_async
 * 
 * @param dptr Output parameter for allocated device pointer
 * @param size Size to allocate
 * @param hStream CUDA stream for async operation
 * @return 0 on success, error code on failure
 */
int allocate_async_raw(CUdeviceptr *dptr, size_t size, CUstream hStream) {
    int tmp;
    pthread_mutex_lock(&mutex);
    tmp = add_chunk_async(dptr,size,hStream);
    pthread_mutex_unlock(&mutex);
    return tmp;
}
