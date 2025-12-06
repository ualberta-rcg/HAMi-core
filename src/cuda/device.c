#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"
#include "include/nvml_prefix.h"
#include "include/libnvml_hook.h"

#include "allocator/allocator.h"
#include "include/memory_limit.h"

CUresult CUDAAPI cuDeviceGetAttribute ( int* pi, CUdevice_attribute attrib, CUdevice dev ) {
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetAttribute,pi,attrib,dev);
    //LOG_DEBUG("[%d]cuDeviceGetAttribute dev=%d attrib=%d %d",res,dev,(int)attrib,*pi);
    return res;
}

CUresult cuDeviceGet(CUdevice *device,int ordinal){
    LOG_DEBUG("into cuDeviceGet ordinal=%d\n",ordinal);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGet,device,ordinal);
    return res;
}

CUresult cuDeviceGetCount( int* count ) {
    LOG_DEBUG("into cuDeviceGetCount");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetCount,count);
    LOG_DEBUG("cuDeviceGetCount res=%d count=%d",res,*count);
    return res;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    LOG_DEBUG("into cuDeviceGetName");
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetName, name, len, dev);
    return res;
}

CUresult cuDeviceCanAccessPeer( int* canAccessPeer, CUdevice dev, CUdevice peerDev ) {
    LOG_INFO("into cuDeviceCanAccessPeer %d %d",dev,peerDev);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceCanAccessPeer,canAccessPeer,dev,peerDev);
}

CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib,
                                 CUdevice srcDevice, CUdevice dstDevice) {
    LOG_DEBUG("into cuDeviceGetP2PAttribute\n");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetP2PAttribute, value,
                         attrib, srcDevice, dstDevice);
}

CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetByPCIBusId, dev,
                         pciBusId);
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    LOG_INFO("into cuDeviceGetPCIBusId dev=%d len=%d",dev,len);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetPCIBusId, pciBusId, len,
                        dev);
    return res;
}

CUresult cuDeviceGetUuid(CUuuid* uuid,CUdevice dev) {
    LOG_DEBUG("into cuDeviceGetUuid dev=%d",dev);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetUuid,uuid,dev);
    return res;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
    LOG_DEBUG("cuDeviceGetDefaultMemPool");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetDefaultMemPool,
                         pool_out, dev);
}

CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev){
    LOG_DEBUG("cuDeviceGetMemPool");
    return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetMemPool, pool, dev);
}

CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask,
                         CUdevice dev) {
  LOG_DEBUG("cuDeviceGetLuid");
  return CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceGetLuid, luid,
                         deviceNodeMask, dev);
}

CUresult cuDeviceTotalMem_v2 ( size_t* bytes, CUdevice dev ) {
    LOG_DEBUG("into cuDeviceTotalMem dev=%d (pid=%d)", dev, getpid());
    
    // Ensure initialization (but don't block if it fails - might be called before cuInit)
    ensure_initialized();
    
    // Get real physical memory first (as fallback)
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry, cuDeviceTotalMem_v2, bytes, dev);
    if (res != CUDA_SUCCESS) {
        LOG_ERROR("cuDeviceTotalMem_v2: Failed to get real memory (dev=%d, pid=%d)", dev, getpid());
        return res;
    }
    size_t real_memory = *bytes;
    
    // Check if softmig is enabled and limits are configured
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        // SoftMig not enabled or not initialized yet - return real memory
        LOG_DEBUG("cuDeviceTotalMem_v2: SoftMig disabled or not initialized - returning real memory: %.2f GB (dev=%d, pid=%d)", 
                 real_memory / (1024.0 * 1024.0 * 1024.0), dev, getpid());
        return CUDA_SUCCESS;
    }
    
    // Convert CUDA device index to NVML device index for memory limit lookup
    // Use identity mapping as fallback if mapping not initialized yet
    int nvml_dev = (dev >= 0 && dev < CUDA_DEVICE_MAX_COUNT) ? cuda_to_nvml_map(dev) : dev;
    if (nvml_dev < 0 || nvml_dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_WARN("cuDeviceTotalMem_v2: Invalid NVML device index %d from CUDA device %d - using identity mapping (pid=%d)", 
                 nvml_dev, dev, getpid());
        nvml_dev = dev;
    }
    
    size_t limit = get_current_device_memory_limit(nvml_dev);
    
    if (limit == 0) {
        // No limit configured - return real physical memory
        LOG_INFO("cuDeviceTotalMem_v2: No limit - returning real physical memory: %.2f GB (dev=%d, nvml_dev=%d, pid=%d)", 
                 real_memory / (1024.0 * 1024.0 * 1024.0), dev, nvml_dev, getpid());
        return CUDA_SUCCESS;
    } else {
        // Return the imposed limit instead of physical memory
        // This "lies" to the application about total GPU memory to prevent over-allocation
        *bytes = limit;
        LOG_INFO("cuDeviceTotalMem_v2: Returning limit instead of physical memory: %.2f GB (real=%.2f GB, dev=%d, nvml_dev=%d, pid=%d) - application will see this as total GPU memory",
                 limit / (1024.0 * 1024.0 * 1024.0), real_memory / (1024.0 * 1024.0 * 1024.0), dev, nvml_dev, getpid());
        return CUDA_SUCCESS;
    }
}

CUresult cuDriverGetVersion(int *driverVersion) {
    LOG_DEBUG("into cuDriverGetVersion__");
    
    //stub dlsym to prelaod cuda functions
    dlsym(RTLD_DEFAULT,"cuDriverGetVersion");

    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuDriverGetVersion,driverVersion);
    //*driverVersion=11030;
    if ((res==CUDA_SUCCESS) && (driverVersion!=NULL)) {
        LOG_INFO("driver version=%d",*driverVersion);
    }
    return res;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements, CUarray_format format, unsigned numChannels, CUdevice dev){
    LOG_DEBUG("cuDeviceGetTexture1DLinearMaxWidth");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceGetTexture1DLinearMaxWidth,maxWidthInElements,format,numChannels,dev);
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
    LOG_DEBUG("cuDeviceSetMemPool");
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuDeviceSetMemPool,dev,pool);
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target, CUflushGPUDirectRDMAWritesScope scope) {
   LOG_DEBUG("cuFlushGPUDirectRDMAWrites");
   return CUDA_OVERRIDE_CALL(cuda_library_entry,cuFlushGPUDirectRDMAWrites,target,scope);
}
