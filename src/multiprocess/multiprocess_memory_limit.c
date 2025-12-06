/**
 * @file multiprocess_memory_limit.c
 * @brief Multi-process shared memory coordination for GPU memory limits
 * 
 * This is the core of SoftMig's multi-process coordination system. It manages
 * a shared memory region (mmap) that allows all processes within a SLURM job
 * to coordinate GPU memory usage. Key features:
 * - Shared memory region with process slots for tracking per-process usage
 * - Memory limit enforcement across all processes in a job
 * - SM (Streaming Multiprocessor) utilization tracking
 * - Process registration and cleanup
 * 
 * SLURM-specific: Uses SLURM_JOB_ID to create job-specific shared memory files
 * in SLURM_TMPDIR, ensuring processes in the same job share the same region
 * while different jobs are isolated.
 * 
 * @authors Rahim Khoja, Karim Ali
 * @organization Research Computing, University of Alberta
 * @note This is a HAMi-core fork with SLURM-specific changes for Alliance clusters
 */

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stddef.h>
#include <semaphore.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>

#include <assert.h>
#include <cuda.h>
#include "include/nvml_prefix.h"
#include <nvml.h>

#include "include/process_utils.h"
#include "include/memory_limit.h"
#include "multiprocess/multiprocess_memory_limit.h"


#ifndef SEM_WAIT_TIME
#define SEM_WAIT_TIME 10
#endif

#ifndef SEM_WAIT_TIME_ON_EXIT
#define SEM_WAIT_TIME_ON_EXIT 3
#endif

#ifndef SEM_WAIT_RETRY_TIMES
#define SEM_WAIT_RETRY_TIMES 30
#endif

int pidfound;

int ctx_activate[32];

static shared_region_info_t region_info = {0, -1, PTHREAD_ONCE_INIT, NULL, 0};
//size_t initial_offset=117440512;
int env_utilization_switch;
int enable_active_oom_killer;
size_t context_size;
size_t initial_offset=0;
// Flag to track if softmig is disabled (when env vars are not set)
static int softmig_disabled = -1;  // -1 = not checked yet, 0 = enabled, 1 = disabled

/**
 * @brief Check if SoftMig is enabled (passive mode detection)
 * 
 * SoftMig operates in "passive mode" if no limits are configured. This allows
 * the library to be preloaded without breaking applications that don't need
 * GPU slicing. Checks if CUDA_DEVICE_MEMORY_LIMIT or CUDA_DEVICE_SM_LIMIT
 * are set (via config file or environment).
 * 
 * SLURM-specific: Limits are typically set via config files created by the
 * prolog script at /var/run/softmig/{jobid}.conf
 * 
 * @return 1 if enabled (limits configured), 0 if disabled (passive mode)
 */
static int is_softmig_enabled(void) {
    if (softmig_disabled == -1) {
        // First time check - see if environment variables are configured
        if (!is_softmig_configured()) {
            softmig_disabled = 1;
            LOG_DEBUG("softmig: CUDA_DEVICE_MEMORY_LIMIT and CUDA_DEVICE_SM_LIMIT not set - softmig disabled (passive mode)");
            return 0;
        }
        softmig_disabled = 0;
    }
    return (softmig_disabled == 0);
}
//lock for record kernel time
pthread_mutex_t _kernel_mutex;
int _record_kernel_interval = 1;

// forwards

void do_init_device_memory_limits(uint64_t*, int);
void exit_withlock(int exitcode);

// External function from config_file.c - reads from config file or env
extern size_t get_limit_from_config_or_env(const char* env_name);
// Forward declaration - defined in config_file.c
int is_softmig_configured(void);

/**
 * @brief Set the GPU status for the current process
 * 
 * Updates the status field in the shared memory region for this process.
 * Status values: 1 = normal, 2 = swapped/suspended. Used for process
 * coordination and suspension/resumption.
 * 
 * @param status Status value to set
 */
void set_current_gpu_status(int status){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++)
        if (getpid()==region_info.shared_region->procs[i].pid){
            region_info.shared_region->procs[i].status = status;
            return;
        }
}

/**
 * @brief Signal handler for SIGUSR1 - restore/resume process
 * 
 * Sets process status to 1 (normal/active) when resumed.
 */
void sig_restore_stub(int signo){
    set_current_gpu_status(1);
}

/**
 * @brief Signal handler for SIGUSR2 - swap/suspend process
 * 
 * Sets process status to 2 (swapped/suspended) when suspended.
 */
void sig_swap_stub(int signo){
    set_current_gpu_status(2);
}

/**
 * @brief Get memory limit from config file or environment variable
 * 
 * Wrapper that prioritizes config file (SLURM-specific) over environment
 * variables. Config files are created by the SLURM prolog script.
 * 
 * @param env_name Environment variable name (e.g., "CUDA_DEVICE_MEMORY_LIMIT")
 * @return Limit in bytes, or 0 if not set
 */
size_t get_limit_from_env(const char* env_name) {
    return get_limit_from_config_or_env(env_name);
}

/**
 * @brief Initialize device information in shared memory
 * 
 * Queries NVML to get the number of GPUs and their UUIDs, storing them
 * in the shared memory region. This allows all processes to have consistent
 * device identification.
 * 
 * @return 0 on success
 */
int init_device_info() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    unsigned int i,nvmlDevicesCount;
    CHECK_NVML_API(nvmlDeviceGetCount_v2(&nvmlDevicesCount));
    region_info.shared_region->device_num=nvmlDevicesCount;
    nvmlDevice_t dev;
    for(i=0;i<nvmlDevicesCount;i++){
        CHECK_NVML_API(nvmlDeviceGetHandleByIndex(i, &dev));
        CHECK_NVML_API(nvmlDeviceGetUUID(dev,region_info.shared_region->uuids[i],NVML_DEVICE_UUID_V2_BUFFER_SIZE));
    }
    LOG_INFO("put_device_info finished %d",nvmlDevicesCount);
    return 0;
}


/**
 * @brief Load environment variables from a config file
 * 
 * Reads a config file (typically created by SLURM prolog) and sets
 * environment variables. Format: KEY=VALUE, one per line.
 * 
 * SLURM-specific: Used to load limits from /var/run/softmig/{jobid}.conf
 * 
 * @param filename Path to config file
 * @return 0 on success
 */
int load_env_from_file(char *filename) {
    FILE *f=fopen(filename,"r");
    if (f==NULL)
        return 0;
    char tmp[10000];
    int cursor=0;
    while (!feof(f)){
        fgets(tmp,10000,f);
        if (strstr(tmp,"=")==NULL)
            break;
        if (tmp[strlen(tmp)-1]=='\n')
            tmp[strlen(tmp)-1]='\0';
        for (cursor=0;cursor<strlen(tmp);cursor++){
            if (tmp[cursor]=='=') {
                tmp[cursor]='\0';
                setenv(tmp,tmp+cursor+1,1);
                LOG_INFO("SET %s to %s",tmp,tmp+cursor+1);
                break;
            }
        }
    }
    return 0;
}

/**
 * @brief Initialize memory limits for all devices
 * 
 * Reads memory limits from config/environment for each device. Supports
 * per-device limits (CUDA_DEVICE_MEMORY_LIMIT_0, _1, etc.) or a global
 * limit (CUDA_DEVICE_MEMORY_LIMIT).
 * 
 * @param arr Array to fill with limits (one per device)
 * @param len Number of devices
 */
void do_init_device_memory_limits(uint64_t* arr, int len) {
    size_t fallback_limit = get_limit_from_env(CUDA_DEVICE_MEMORY_LIMIT);
    int i;
    for (i = 0; i < len; ++i) {
        char env_name[CUDA_DEVICE_MEMORY_LIMIT_KEY_LENGTH] = CUDA_DEVICE_MEMORY_LIMIT;
        char index_name[16];  // Increased from 8 to handle large device indices
        snprintf(index_name, sizeof(index_name), "_%d", i);
        strcat(env_name, index_name);
        size_t cur_limit = get_limit_from_env(env_name);
        if (cur_limit > 0) {
            arr[i] = cur_limit;
        } else if (fallback_limit > 0) {
            arr[i] = fallback_limit;
        } else {
            arr[i] = 0;
        }
    }
}

/**
 * @brief Initialize SM (Streaming Multiprocessor) limits for all devices
 * 
 * Reads SM utilization limits from config/environment. SM limits control
 * the percentage of GPU compute resources available. Defaults to 100% if
 * not specified.
 * 
 * @param arr Array to fill with SM limits (one per device, as percentage)
 * @param len Number of devices
 */
void do_init_device_sm_limits(uint64_t *arr, int len) {
    size_t fallback_limit = get_limit_from_env(CUDA_DEVICE_SM_LIMIT);
    if (fallback_limit == 0) fallback_limit = 100;
    int i;
    for (i = 0; i < len; ++i) {
        char env_name[CUDA_DEVICE_SM_LIMIT_KEY_LENGTH] = CUDA_DEVICE_SM_LIMIT;
        char index_name[16];  // Increased from 8 to handle large device indices
        snprintf(index_name, sizeof(index_name), "_%d", i);
        strcat(env_name, index_name);
        size_t cur_limit = get_limit_from_env(env_name);
        if (cur_limit > 0) {
            arr[i] = cur_limit;
        } else if (fallback_limit > 0) {
            arr[i] = fallback_limit;
        } else {
            arr[i] = 0;
        }
    }
}

/**
 * @brief Kill all processes in the shared memory region (emergency OOM handler)
 * 
 * Emergency function that sends SIGKILL to all registered processes.
 * Used as a last resort when memory limits are exceeded and cleanup
 * fails. Should rarely be needed in normal operation.
 * 
 * @return 0
 */
int active_oom_killer() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        kill(region_info.shared_region->procs[i].pid,9);
    }
    return 0;
}

/**
 * @brief Called before kernel launch for SM utilization tracking
 * 
 * Records the timestamp of kernel launches to enable SM utilization
 * rate limiting. The utilization watcher thread uses this to throttle
 * kernel launches when SM limits are exceeded.
 * 
 * SLURM-specific: SM limits are enforced per-job via shared memory
 * coordination.
 */
void pre_launch_kernel() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    uint64_t now = time(NULL);
    pthread_mutex_lock(&_kernel_mutex);
    if (now - region_info.last_kernel_time < _record_kernel_interval) {
        pthread_mutex_unlock(&_kernel_mutex);
        return;
    }
    region_info.last_kernel_time = now;
    pthread_mutex_unlock(&_kernel_mutex);
    LOG_INFO("write last kernel time: %ld", now)
    lock_shrreg();
    if (region_info.shared_region->last_kernel_time < now) {
        region_info.shared_region->last_kernel_time = now;
    }
    unlock_shrreg();
}

int shrreg_major_version() {
    return MAJOR_VERSION;
}

int shrreg_minor_version() {
    return MINOR_VERSION;
}


size_t get_gpu_memory_monitor(const int dev) {
    LOG_DEBUG("get_gpu_memory_monitor dev=%d",dev);
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;
    }
    int i=0;
    size_t total=0;
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        LOG_DEBUG("dev=%d i=%lu,%lu\n",dev,region_info.shared_region->procs[i].monitorused[dev],region_info.shared_region->procs[i].used[dev].total);
        total+=region_info.shared_region->procs[i].monitorused[dev];
    }
    unlock_shrreg();
    return total;
}

/**
 * @brief Get total GPU memory usage across all processes in the job
 * 
 * This is the critical function for multi-process memory coordination.
 * It sums memory usage from ALL processes registered in the shared memory
 * region for the specified device. This ensures that when one process
 * checks OOM, it sees the total usage from all processes in the job.
 * 
 * SLURM-specific: All processes in the same SLURM job share the same
 * shared memory file (via SLURM_JOB_ID), so they all see the same
 * aggregated memory usage.
 * 
 * @param dev NVML device index
 * @return Total memory usage in bytes across all processes
 */
size_t get_gpu_memory_usage(const int dev) {
    LOG_INFO("get_gpu_memory_usage dev=%d pid=%d",dev,getpid());
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        LOG_WARN("get_gpu_memory_usage: softmig disabled or shared_region NULL (pid=%d)", getpid());
        return 0;
    }
    int i=0;
    size_t total=0;
    lock_shrreg();
    LOG_INFO("get_gpu_memory_usage: proc_num=%d (pid=%d)", region_info.shared_region->proc_num, getpid());
    for (i=0;i<region_info.shared_region->proc_num;i++){
        size_t proc_usage = region_info.shared_region->procs[i].used[dev].total;
        LOG_INFO("dev=%d pid=%d host_pid=%d usage=%lu (total so far=%lu)", 
                 dev, region_info.shared_region->procs[i].pid, 
                 region_info.shared_region->procs[i].hostpid, proc_usage, total);
        total+=proc_usage;
    }
    total+=initial_offset;
    unlock_shrreg();
    LOG_INFO("get_gpu_memory_usage: total=%lu (pid=%d, dev=%d)", total, getpid(), dev);
    return total;
}

int set_gpu_device_memory_monitor(int32_t pid,int dev,size_t monitor){
    //LOG_WARN("set_gpu_device_memory_monitor:%d %d %lu",pid,dev,monitor);
    int i;
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].hostpid == pid){
            LOG_INFO("set_gpu_device_memory_monitor:%d %d %lu->%lu",pid,dev,region_info.shared_region->procs[i].used[dev].total,monitor);
            region_info.shared_region->procs[i].monitorused[dev] = monitor;
            break;
        }
    }
    unlock_shrreg();
    return 1;
}

int set_gpu_device_sm_utilization(int32_t pid,int dev, unsigned int smUtil){  // new function
    int i;
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].hostpid == pid){
            LOG_INFO("set_gpu_device_sm_utilization:%d %d %lu->%u", pid, dev, region_info.shared_region->procs[i].device_util[dev].sm_util, smUtil);
            region_info.shared_region->procs[i].device_util[dev].sm_util = smUtil;
            break;
        }
    }
    unlock_shrreg();
    return 1;
}

int init_gpu_device_utilization(){
    int i,dev;
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    lock_shrreg();
    for (i=0;i<region_info.shared_region->proc_num;i++){
        for (dev=0;dev<CUDA_DEVICE_MAX_COUNT;dev++){
            region_info.shared_region->procs[i].device_util[dev].sm_util = 0;
            region_info.shared_region->procs[i].monitorused[dev] = 0;
            break;
        }
    }
    unlock_shrreg();
    return 1;
}

uint64_t nvml_get_device_memory_usage(const int dev) {
    nvmlDevice_t ndev;
    nvmlReturn_t ret;
    ret = nvmlDeviceGetHandleByIndex(dev, &ndev);
    if (ret != NVML_SUCCESS) {
        LOG_ERROR("NVML get device %d error, %s", dev, nvmlErrorString(ret));
    }
    unsigned int pcnt = SHARED_REGION_MAX_PROCESS_NUM;
    nvmlProcessInfo_v1_t infos[SHARED_REGION_MAX_PROCESS_NUM];
    LOG_DEBUG("before nvmlDeviceGetComputeRunningProcesses");
    ret = nvmlDeviceGetComputeRunningProcesses(ndev, &pcnt, infos);
    if (ret != NVML_SUCCESS) {
        LOG_ERROR("NVML get process error, %s", nvmlErrorString(ret));
    }
    int i = 0;
    uint64_t usage = 0;
    shared_region_t* region = region_info.shared_region;
    lock_shrreg();
    for (; i < pcnt; i++) {
        int slot = 0;
        for (; slot < region->proc_num; slot++) {
            if (infos[i].pid != region->procs[slot].pid)
                continue;
            usage += infos[i].usedGpuMemory;
        }
    }
    unlock_shrreg();
    LOG_DEBUG("Device %d current memory %lu / %lu", 
            dev, usage, region->limit[dev]);
    return usage;
}

/**
 * @brief Add memory usage for a process to the shared memory region
 * 
 * Updates the shared memory region to track memory allocated by a process.
 * This is called whenever memory is allocated (via CUDA hooks) to keep the
 * shared memory region synchronized. The usage is tracked per-process and
 * per-device, allowing get_gpu_memory_usage() to aggregate across processes.
 * 
 * SLURM-specific: All processes in the same job update the same shared
 * memory file, ensuring coordinated memory tracking.
 * 
 * @param pid Process ID (typically getpid())
 * @param cudadev CUDA device index
 * @param usage Memory size in bytes to add
 * @param type Memory type: 0=context, 1=module, 2=data
 * @return 0 on success, -1 if process not found in shared region
 */
int add_gpu_device_memory_usage(int32_t pid,int cudadev,size_t usage,int type){
    LOG_INFO("add_gpu_device_memory: pid=%d cuda_dev=%d->nvml_dev=%d usage=%lu type=%d", 
             pid, cudadev, cuda_to_nvml_map(cudadev), usage, type);
    int dev = cuda_to_nvml_map(cudadev);
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        LOG_WARN("add_gpu_device_memory: softmig disabled or shared_region NULL (pid=%d)", pid);
        return 0;  // No-op when softmig is disabled
    }
    LOG_DEBUG("add_gpu_device_memory: Acquiring lock (pid=%d, proc_num=%d)", 
              pid, region_info.shared_region->proc_num);
    lock_shrreg();
    int i;
    int found = 0;
    LOG_DEBUG("add_gpu_device_memory: Searching for pid=%d in %d processes", 
              pid, region_info.shared_region->proc_num);
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == pid){
            found = 1;
            size_t old_total = region_info.shared_region->procs[i].used[dev].total;
            region_info.shared_region->procs[i].used[dev].total+=usage;
            LOG_DEBUG("add_gpu_device_memory: Found pid=%d at slot %d, dev=%d: %lu -> %lu", 
                      pid, i, dev, old_total, region_info.shared_region->procs[i].used[dev].total);
            switch (type) {
                case 0:{
                    region_info.shared_region->procs[i].used[dev].context_size += usage;
                    break;
                }
                case 1:{
                    region_info.shared_region->procs[i].used[dev].module_size += usage;
                    break;
                }
                case 2:{
                    region_info.shared_region->procs[i].used[dev].data_size += usage;
                }
            }
            LOG_INFO("add_gpu_device_memory: pid=%d dev=%d old_total=%lu added=%lu new_total=%lu", 
                     pid, dev, old_total, usage, region_info.shared_region->procs[i].used[dev].total);
            break;
        }
    }
    if (!found) {
        LOG_ERROR("add_gpu_device_memory: PID %d not found in shared region! proc_num=%d (memory not tracked)", 
                  pid, region_info.shared_region->proc_num);
        LOG_ERROR("add_gpu_device_memory: Current processes in shared region:");
        for (i=0; i<region_info.shared_region->proc_num; i++) {
            LOG_ERROR("  Slot[%d]: pid=%d, hostpid=%d, dev[%d].total=%lu", 
                      i, region_info.shared_region->procs[i].pid,
                      region_info.shared_region->procs[i].hostpid, dev,
                      region_info.shared_region->procs[i].used[dev].total);
        }
        // Process not registered - this is a serious issue, but don't fail allocation
        // The process should have been registered in init_proc_slot_withlock()
    }
    unlock_shrreg();
    size_t total_usage = get_gpu_memory_usage(dev);
    LOG_INFO("gpu_device_memory_added: pid=%d dev=%d added=%lu total_across_all_procs=%lu limit=%lu", 
             pid, dev, usage, total_usage, get_current_device_memory_limit(dev));
    return 0;
}

int rm_gpu_device_memory_usage(int32_t pid,int cudadev,size_t usage,int type){
    LOG_INFO("rm_gpu_device_memory:%d %d->%d %d:%lu",pid,cudadev,cuda_to_nvml_map(cudadev),type,usage);
    int dev = cuda_to_nvml_map(cudadev);
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    lock_shrreg();
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == pid){
            region_info.shared_region->procs[i].used[dev].total-=usage;
            switch (type) {
                case 0:{
                    region_info.shared_region->procs[i].used[dev].context_size -= usage;
                    break;
                }
                case 1:{
                    region_info.shared_region->procs[i].used[dev].module_size -= usage;
                    break;
                }
                case 2:{
                    region_info.shared_region->procs[i].used[dev].data_size -= usage;
                }
            }
            LOG_INFO("after delete:%lu",region_info.shared_region->procs[i].used[dev].total);
        }
    }
    unlock_shrreg();
    return 0;
}

void get_timespec(int seconds, struct timespec* spec) {
    struct timeval tv;
    gettimeofday(&tv, NULL);  // struggle with clock_gettime version
    spec->tv_sec = tv.tv_sec + seconds;
    spec->tv_nsec = 0;
}

int fix_lock_shrreg() {
    int res = 1;
    if (region_info.fd == -1) {
        // should never happen
        LOG_ERROR("Uninitialized shrreg");
    }
    // upgrade
    if (lockf(region_info.fd, F_LOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to upgraded lock: errno=%d", errno);
    }
    SEQ_POINT_MARK(SEQ_FIX_SHRREG_ACQUIRE_FLOCK_OK);

    shared_region_t* region = region_info.shared_region;
    int32_t current_owner = region->owner_pid;
    if (current_owner != 0) {
        int flag = 0;
        if (current_owner == region_info.pid) {
            LOG_INFO("Detect onwer pid = self pid (%d), "
                "indicates pid loopback or race condition", current_owner);
            flag = 1;
        } else {
            int proc_status = proc_alive(current_owner);
            if (proc_status == PROC_STATE_NONALIVE) {
                LOG_INFO("Kick dead owner proc (%d)", current_owner);
                flag = 1;
            }
        }
        if (flag == 1) {
            LOG_INFO("Take upgraded lock (%d)", region_info.pid);
            region->owner_pid = region_info.pid;
            SEQ_POINT_MARK(SEQ_FIX_SHRREG_UPDATE_OWNER_OK);
            res = 0;     
        }
    }

    if (lockf(region_info.fd, F_ULOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to upgraded unlock: errno=%d", errno);
    }
    SEQ_POINT_MARK(SEQ_FIX_SHRREG_RELEASE_FLOCK_OK);
    return res;
}

void exit_withlock(int exitcode) {
    unlock_shrreg();
    exit(exitcode);
}


// External function from config_file.c - cleanup config file
extern void cleanup_config_file(void);

void exit_handler() {
    if (region_info.init_status == PTHREAD_ONCE_INIT) {
        return;
    }
    shared_region_t* region = region_info.shared_region;
    
    // Check if shared region was never initialized (e.g., program failed to start)
    // This can happen when bash loads the library but the program doesn't exist
    if (region == NULL) {
        // Clean up config file even if shared region wasn't initialized
        cleanup_config_file();
        return;
    }
    
    int slot = 0;
    LOG_INFO("exit_handler: Process exiting (pid=%d, proc_num=%d)", getpid(), region->proc_num);
    
    // Clean up config file (delete it)
    cleanup_config_file();
    
    struct timespec sem_ts;
    get_timespec(SEM_WAIT_TIME_ON_EXIT, &sem_ts);
    LOG_DEBUG("exit_handler: Attempting to acquire lock for cleanup (pid=%d)", getpid());
    int status = sem_timedwait(&region->sem, &sem_ts);
    if (status == 0) {  // just give up on lock failure
        region->owner_pid = region_info.pid;
        LOG_DEBUG("exit_handler: Lock acquired, searching for pid=%d in %d processes", 
                  region_info.pid, region->proc_num);
        while (slot < region->proc_num) {
            if (region->procs[slot].pid == region_info.pid) {
                LOG_INFO("exit_handler: Found process slot %d for pid=%d, removing (pid=%d)", 
                         slot, region_info.pid, getpid());
                memset(region->procs[slot].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
                memset(region->procs[slot].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
                region->proc_num--;
                region->procs[slot] = region->procs[region->proc_num];
                LOG_INFO("exit_handler: Process removed, new proc_num=%d (pid=%d)", 
                         region->proc_num, getpid());
                break;
            }
            slot++;
        }
        if (slot >= region->proc_num) {
            LOG_WARN("exit_handler: PID %d not found in shared region (pid=%d)", 
                     region_info.pid, getpid());
        }
        __sync_synchronize();
        region->owner_pid = 0;
        sem_post(&region->sem);
        LOG_DEBUG("exit_handler: Lock released, cleanup complete (pid=%d)", getpid());
    } else {
        LOG_WARN("exit_handler: Failed to take lock on exit: errno=%d (pid=%d)", errno, getpid());
    }
}


/**
 * @brief Acquire lock on shared memory region
 * 
 * Locks the semaphore protecting the shared memory region. This is critical
 * for thread/process safety when multiple processes access the same shared
 * memory. Uses timed wait to avoid deadlocks.
 * 
 * SLURM-specific: All processes in the same job compete for the same lock,
 * ensuring atomic updates to the shared memory region.
 */
void lock_shrreg() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    LOG_DEBUG("lock_shrreg: Attempting to acquire lock (pid=%d, owner_pid=%ld)", 
              getpid(), region_info.shared_region ? region_info.shared_region->owner_pid : -1);
    struct timespec sem_ts;
    get_timespec(SEM_WAIT_TIME, &sem_ts);
    shared_region_t* region = region_info.shared_region;
    int trials = 0;
    while (1) {
        int status = sem_timedwait(&region->sem, &sem_ts);
        SEQ_POINT_MARK(SEQ_ACQUIRE_SEMLOCK_OK);
        LOG_DEBUG("lock_shrreg: sem_timedwait returned %d (pid=%d, errno=%d)", status, getpid(), errno);

        if (status == 0) {
            // TODO: irregular exit here will hang pending locks
            region->owner_pid = region_info.pid;
            __sync_synchronize();
            SEQ_POINT_MARK(SEQ_UPDATE_OWNER_OK);
            LOG_DEBUG("lock_shrreg: Lock acquired successfully (pid=%d, owner_pid=%ld)", 
                      getpid(), region->owner_pid);
            trials = 0;
            break;
        } else if (errno == ETIMEDOUT) {
            LOG_WARN("Lock shrreg timeout, try fix (%d:%ld)", region_info.pid,region->owner_pid);
            int32_t current_owner = region->owner_pid;
            if (current_owner != 0 && (current_owner == region_info.pid ||
                    proc_alive(current_owner) == PROC_STATE_NONALIVE)) {
                LOG_WARN("Owner proc dead (%d), try fix", current_owner);
                if (0 == fix_lock_shrreg()) {
                    break;
                }
            } else {
                trials++;
                if (trials > SEM_WAIT_RETRY_TIMES) {
                    LOG_WARN("Fail to lock shrreg in %d seconds",
                        SEM_WAIT_RETRY_TIMES * SEM_WAIT_TIME);
                    if (current_owner == 0) {
                        LOG_WARN("fix current_owner 0>%d",region_info.pid);
                        region->owner_pid = region_info.pid;
                        if (0 == fix_lock_shrreg()) {
                            break;
                        } 
                    }
                }
                continue;  // slow wait path
            }
        } else {
            LOG_ERROR("Failed to lock shrreg: %d", errno);
        }
    }
}

void unlock_shrreg() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    LOG_DEBUG("unlock_shrreg: Releasing lock (pid=%d, owner_pid=%ld)", 
              getpid(), region_info.shared_region->owner_pid);
    SEQ_POINT_MARK(SEQ_BEFORE_UNLOCK_SHRREG);
    shared_region_t* region = region_info.shared_region;

    __sync_synchronize();
    region->owner_pid = 0;
    // TODO: irregular exit here will hang pending locks
    SEQ_POINT_MARK(SEQ_RESET_OWNER_OK);

    sem_post(&region->sem);
    SEQ_POINT_MARK(SEQ_RELEASE_SEMLOCK_OK);
    LOG_DEBUG("unlock_shrreg: Lock released (pid=%d)", getpid());
}


int clear_proc_slot_nolock(int do_clear) {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    int slot = 0;
    int res=0;
    shared_region_t* region = region_info.shared_region;
    while (slot < region->proc_num) {
        int32_t pid = region->procs[slot].pid;
        if (pid != 0) {
            if (do_clear > 0 && proc_alive(pid) == PROC_STATE_NONALIVE) {
                LOG_WARN("Kick dead proc %d", pid);
            } else {
                slot++;
                continue;
            }
            res=1;
            region->proc_num--;
            region->procs[slot] = region->procs[region->proc_num];
            __sync_synchronize();
        }
    }
    return res;
}

/**
 * @brief Register the current process in the shared memory region
 * 
 * Called during initialization to register this process in the shared
 * memory region. Finds an empty slot or reuses an existing slot if the
 * process was already registered (e.g., after fork). Initializes the
 * process's memory tracking structures.
 * 
 * SLURM-specific: All processes in the same job register themselves,
 * allowing coordinated memory tracking.
 */
void init_proc_slot_withlock() {
    int32_t current_pid = getpid();
    LOG_INFO("init_proc_slot_withlock: Registering process (pid=%d, current proc_num=%d)", 
             current_pid, region_info.shared_region ? region_info.shared_region->proc_num : -1);
    lock_shrreg();
    shared_region_t* region = region_info.shared_region;
    if (region->proc_num >= SHARED_REGION_MAX_PROCESS_NUM) {
        LOG_ERROR("init_proc_slot_withlock: Shared region full! proc_num=%d >= max=%d (pid=%d)", 
                  region->proc_num, SHARED_REGION_MAX_PROCESS_NUM, current_pid);
        exit_withlock(-1);
    }
    signal(SIGUSR2,sig_swap_stub);
    signal(SIGUSR1,sig_restore_stub);
    // If, by any means a pid of itself is found in region->proces, then it is probably caused by crashloop
    // we need to reset it.
    int i,found=0;
    for (i=0; i<region->proc_num; i++) {
        if (region->procs[i].pid == current_pid) {
            LOG_INFO("init_proc_slot_withlock: Found existing slot for pid=%d at index %d, resetting (pid=%d)", 
                     current_pid, i, current_pid);
            region->procs[i].status = 1;
            memset(region->procs[i].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
            memset(region->procs[i].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
            found = 1;
            break;
        }
    }
    if (!found) {
        LOG_INFO("init_proc_slot_withlock: Creating new slot for pid=%d at index %d (pid=%d)", 
                 current_pid, region->proc_num, current_pid);
        region->procs[region->proc_num].pid = current_pid;
        region->procs[region->proc_num].status = 1;
        memset(region->procs[region->proc_num].used,0,sizeof(device_memory_t)*CUDA_DEVICE_MAX_COUNT);
        memset(region->procs[region->proc_num].device_util,0,sizeof(device_util_t)*CUDA_DEVICE_MAX_COUNT);
        region->proc_num++;
        LOG_INFO("init_proc_slot_withlock: Process registered successfully (pid=%d, new proc_num=%d)", 
                 current_pid, region->proc_num);
    }

    int cleared = clear_proc_slot_nolock(1);
    if (cleared > 0) {
        LOG_INFO("init_proc_slot_withlock: Cleared %d dead process slot(s) (pid=%d)", cleared, current_pid);
    }
    unlock_shrreg();
    LOG_INFO("init_proc_slot_withlock: Registration complete (pid=%d, final proc_num=%d)", 
             current_pid, region->proc_num);
}

void print_all() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        LOG_INFO("softmig is disabled - no process information available");
        return;
    }
    int i;
    LOG_INFO("Total process: %d",region_info.shared_region->proc_num);
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        for (int dev=0;dev<CUDA_DEVICE_MAX_COUNT;dev++){
            LOG_INFO("Process %d hostPid: %d, sm: %lu, memory: %lu, record: %lu",
                region_info.shared_region->procs[i].pid,
                region_info.shared_region->procs[i].hostpid, 
                region_info.shared_region->procs[i].device_util[dev].sm_util, 
                region_info.shared_region->procs[i].monitorused[dev], 
                region_info.shared_region->procs[i].used[dev].total);
        }
    }
}

void child_reinit_flag() {
    LOG_DEBUG("Detect child pid: %d -> %d", region_info.pid, getpid());   
    region_info.init_status = PTHREAD_ONCE_INIT;
}

int set_active_oom_killer() {
    char *oom_killer_env;
    oom_killer_env = getenv("ACTIVE_OOM_KILLER");
    if (oom_killer_env!=NULL){
        if (strcmp(oom_killer_env,"false") == 0)
            return 0;
        if (strcmp(oom_killer_env,"true") == 0)
            return 1;
        if (strcmp(oom_killer_env,"0")==0)
            return 0;
        if (strcmp(oom_killer_env,"1")==0)
            return 1;
    }
    return 1;
}

int set_env_utilization_switch() {
    char *utilization_env;
    utilization_env = getenv("GPU_CORE_UTILIZATION_POLICY");
    if (utilization_env!=NULL){
        if ((strcmp(utilization_env,"FORCE") ==0 ) || (strcmp(utilization_env,"force") ==0))
            return 1;
        if ((strcmp(utilization_env,"DISABLE") ==0 ) || (strcmp(utilization_env,"disable") ==0 ))
            return 2;
    }
    return 0;
}

/**
 * @brief Create or attach to the shared memory region
 * 
 * This is the core function that sets up multi-process coordination. It:
 * - Determines the shared memory file path based on SLURM_JOB_ID
 * - Creates the file if it doesn't exist (first process in job)
 * - Maps it into memory via mmap
 * - Initializes the region structure if this is the first process
 * - Registers the current process in the region
 * 
 * SLURM-specific: Uses SLURM_TMPDIR and SLURM_JOB_ID to create job-specific
 * shared memory files, ensuring processes in the same job share the region
 * while different jobs are isolated.
 */
void try_create_shrreg() {
    LOG_INFO("try_create_shrreg: Starting (pid=%d)", getpid());
    if (region_info.fd == -1) {
        // use .fd to indicate whether a reinit after fork happen
        // no need to register exit handler after fork
        if (0 != atexit(exit_handler)) {
            LOG_ERROR("Register exit handler failed: %d", errno);
        }
    }

    enable_active_oom_killer = set_active_oom_killer();
    env_utilization_switch = set_env_utilization_switch();
    pthread_atfork(NULL, NULL, child_reinit_flag);

    region_info.pid = getpid();
    region_info.fd = -1;
    region_info.last_kernel_time = time(NULL);

    umask(0);

    char* shr_reg_file = getenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV);
    if (shr_reg_file == NULL) {
        // Compute Canada optimized: Use SLURM_TMPDIR with job ID for isolation
        // Only use SLURM_TMPDIR (not regular /tmp) for proper job isolation
        static char cache_path[512] = {0};
        char* tmpdir = getenv("SLURM_TMPDIR");
        if (tmpdir == NULL) {
            // No SLURM_TMPDIR - this should only happen outside SLURM jobs
            // For local testing, use /tmp with job ID if available
            char* job_id = getenv("SLURM_JOB_ID");
            if (job_id != NULL) {
                // We're in a SLURM job but SLURM_TMPDIR not set - use /tmp with job ID
                tmpdir = "/tmp";
            } else {
                // Not in SLURM job - use /tmp (for local testing only)
                tmpdir = "/tmp";
            }
        }
        
        // Include job ID for proper isolation (per-job cache)
        // For oversubscription, each job gets its own cache but they coordinate via shared memory
        char* job_id = getenv("SLURM_JOB_ID");
        char* array_id = getenv("SLURM_ARRAY_TASK_ID");
        
        if (job_id != NULL) {
            if (array_id != NULL) {
                snprintf(cache_path, sizeof(cache_path), "%s/cudevshr.cache.%s.%s", tmpdir, job_id, array_id);
            } else {
                snprintf(cache_path, sizeof(cache_path), "%s/cudevshr.cache.%s", tmpdir, job_id);
            }
        } else {
            // Fallback: use user ID and PID
            uid_t uid = getuid();
            pid_t pid = getpid();
            snprintf(cache_path, sizeof(cache_path), "%s/cudevshr.cache.uid%d.pid%d", tmpdir, uid, pid);
        }
        shr_reg_file = cache_path;
        LOG_INFO("try_create_shrreg: Using shared memory file: %s (pid=%d, job_id=%s, tmpdir=%s)", 
                 shr_reg_file, getpid(), job_id ? job_id : "NULL", tmpdir ? tmpdir : "NULL");
    } else {
        LOG_INFO("try_create_shrreg: Using custom shared memory file from env: %s (pid=%d)", 
                 shr_reg_file, getpid());
    }
    // Initialize NVML BEFORE!! open it
    //nvmlInit();

    /* If you need sm modification, do it here */
    /* ... set_sm_scale */

    LOG_INFO("try_create_shrreg: Opening shared memory file: %s (pid=%d)", shr_reg_file, getpid());
    int fd = open(shr_reg_file, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        LOG_ERROR("Fail to open shrreg %s: errno=%d", shr_reg_file, errno);
    } else {
        LOG_INFO("try_create_shrreg: Successfully opened file descriptor %d (pid=%d)", fd, getpid());
    }
    region_info.fd = fd;
    size_t offset = lseek(fd, SHARED_REGION_SIZE_MAGIC, SEEK_SET);
    if (offset != SHARED_REGION_SIZE_MAGIC) {
        LOG_ERROR("Fail to init shrreg %s: errno=%d", shr_reg_file, errno);
    }
    size_t check_bytes = write(fd, "\0", 1);
    if (check_bytes != 1) {
        LOG_ERROR("Fail to write shrreg %s: errno=%d", shr_reg_file, errno);
    }
    if (lseek(fd, 0, SEEK_SET) != 0) {
        LOG_ERROR("Fail to reseek shrreg %s: errno=%d", shr_reg_file, errno);
    }
    LOG_INFO("try_create_shrreg: Mapping shared memory (size=%zu, pid=%d)", SHARED_REGION_SIZE_MAGIC, getpid());
    region_info.shared_region = (shared_region_t*) mmap(
        NULL, SHARED_REGION_SIZE_MAGIC, 
        PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
    shared_region_t* region = region_info.shared_region;
    if (region == NULL) {
        LOG_ERROR("Fail to map shrreg %s: errno=%d", shr_reg_file, errno);
    } else {
        LOG_INFO("try_create_shrreg: Successfully mapped shared memory at %p (pid=%d)", (void*)region, getpid());
    }
    if (lockf(fd, F_LOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to lock shrreg %s: errno=%d", shr_reg_file, errno);
    }
    //put_device_info();
    if (region->initialized_flag != 
          MULTIPROCESS_SHARED_REGION_MAGIC_FLAG) {
        LOG_INFO("try_create_shrreg: Initializing new shared region (pid=%d, proc_num=%d)", getpid(), region->proc_num);
        region->major_version = MAJOR_VERSION;
        region->minor_version = MINOR_VERSION;
        do_init_device_memory_limits(
            region->limit, CUDA_DEVICE_MAX_COUNT);
        do_init_device_sm_limits(
            region->sm_limit,CUDA_DEVICE_MAX_COUNT);
        LOG_INFO("try_create_shrreg: Initialized limits - device 0: memory=%lu bytes, sm=%lu%% (pid=%d)", 
                 region->limit[0], region->sm_limit[0], getpid());
        if (sem_init(&region->sem, 1, 1) != 0) {
            LOG_ERROR("Fail to init sem %s: errno=%d", shr_reg_file, errno);
        }
        __sync_synchronize();
        region->sm_init_flag = 0;
        region->utilization_switch = 1;
        region->recent_kernel = 2;
        region->priority = 1;
        if (getenv(CUDA_TASK_PRIORITY_ENV)!=NULL)
            region->priority = atoi(getenv(CUDA_TASK_PRIORITY_ENV));
        region->initialized_flag = MULTIPROCESS_SHARED_REGION_MAGIC_FLAG;
        LOG_INFO("try_create_shrreg: Shared region initialized (pid=%d)", getpid());
    } else {
        LOG_INFO("try_create_shrreg: Attaching to existing shared region (pid=%d, proc_num=%d, version=%d.%d)", 
                 getpid(), region->proc_num, region->major_version, region->minor_version);
        if (region->major_version != MAJOR_VERSION || 
                region->minor_version != MINOR_VERSION) {
            LOG_ERROR("The current version number %d.%d"
                    " is different from the file's version number %d.%d",
                    MAJOR_VERSION, MINOR_VERSION,
                    region->major_version, region->minor_version);
        }
        uint64_t local_limits[CUDA_DEVICE_MAX_COUNT];
        do_init_device_memory_limits(local_limits, CUDA_DEVICE_MAX_COUNT);
        int i;
        for (i = 0; i < CUDA_DEVICE_MAX_COUNT; ++i) {
            if (local_limits[i] != region->limit[i]) {
                // Downgrade to DEBUG - this is expected when cache is from different job/limit
                // Recreate cache with correct limits from environment
                LOG_DEBUG("Limit inconsistency detected for %dth device, %lu expected, get %lu - updating cache", 
                    i, local_limits[i], region->limit[i]);
                // Update cache with environment limits (environment is source of truth)
                region->limit[i] = local_limits[i];
            }
        }
        do_init_device_sm_limits(local_limits,CUDA_DEVICE_MAX_COUNT);
        for (i = 0; i < CUDA_DEVICE_MAX_COUNT; ++i) {
            if (local_limits[i] != region->sm_limit[i]) {
                // Update cache with environment limits (environment is source of truth)
                LOG_DEBUG("SM limit inconsistency detected for %dth device, %lu expected, get %lu - updating cache",
                    i, local_limits[i], region->sm_limit[i]);
                region->sm_limit[i] = local_limits[i];
            }
        }
    }
    region->last_kernel_time = region_info.last_kernel_time;
    if (lockf(fd, F_ULOCK, SHARED_REGION_SIZE_MAGIC) != 0) {
        LOG_ERROR("Fail to unlock shrreg %s: errno=%d", shr_reg_file, errno);
    }
    LOG_DEBUG("shrreg created");
}

void initialized() {
    LOG_INFO("initialized: Starting initialization (pid=%d)", getpid());
    // Check if softmig should be active (if env vars are set)
    if (!is_softmig_enabled()) {
        // softmig is disabled - don't initialize anything
        LOG_INFO("initialized: SoftMig disabled (passive mode) - skipping initialization (pid=%d)", getpid());
        return;
    }
    
    LOG_INFO("initialized: SoftMig enabled - proceeding with initialization (pid=%d)", getpid());
    pthread_mutex_init(&_kernel_mutex, NULL);
    char* _record_kernel_interval_env = getenv("RECORD_KERNEL_INTERVAL");
    if (_record_kernel_interval_env) {
        _record_kernel_interval = atoi(_record_kernel_interval_env);
        LOG_INFO("initialized: RECORD_KERNEL_INTERVAL=%d (pid=%d)", _record_kernel_interval, getpid());
    }
    try_create_shrreg();
    init_proc_slot_withlock();
    LOG_INFO("initialized: Initialization complete (pid=%d)", getpid());
}

void ensure_initialized() {
    // Check if softmig should be active before initializing
    if (!is_softmig_enabled()) {
        // softmig is disabled - don't initialize anything
        return;
    }
    
    (void) pthread_once(&region_info.init_status, initialized);
}

int update_host_pid() {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == getpid()){
            if (region_info.shared_region->procs[i].hostpid!=0)
                pidfound=1; 
        }
    }
    return 0;
}

int set_host_pid(int hostpid) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    int i,j,found=0;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid == getpid()){
            LOG_INFO("SET PID= %d",hostpid);
            found=1;
            region_info.shared_region->procs[i].hostpid = hostpid;
            for (j=0;j<CUDA_DEVICE_MAX_COUNT;j++)
                region_info.shared_region->procs[i].monitorused[j]=0;
        }
    }
    if (!found) {
        LOG_ERROR("HOST PID NOT FOUND. %d",hostpid);
        return -1;
    }
    setspec();
    return 0;
}

int set_current_device_sm_limit_scale(int dev, int scale) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    if (region_info.shared_region->sm_init_flag==1) return 0;
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    LOG_INFO("dev %d new sm limit set mul by %d",dev,scale);
    region_info.shared_region->sm_limit[dev]=region_info.shared_region->sm_limit[dev]*scale;
    region_info.shared_region->sm_init_flag = 1;
    return 0;
}

int get_current_device_sm_limit(int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 100;  // No limit (100%) when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    return region_info.shared_region->sm_limit[dev];
}

int set_current_device_memory_limit(const int dev,size_t newlimit) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    LOG_INFO("dev %d new limit set to %ld",dev,newlimit);
    region_info.shared_region->limit[dev]=newlimit;
    return 0; 
}

uint64_t get_current_device_memory_limit(const int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No limit when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    return region_info.shared_region->limit[dev];       
}

uint64_t get_current_device_memory_monitor(const int dev) {
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No monitoring when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    uint64_t result = get_gpu_memory_monitor(dev);
//    result= nvml_get_device_memory_usage(dev);
    return result;
}

uint64_t get_current_device_memory_usage(const int dev) {
    clock_t start,finish;
    uint64_t result;
    start = clock();
    ensure_initialized();
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No usage tracking when softmig is disabled
    }
    if (dev < 0 || dev >= CUDA_DEVICE_MAX_COUNT) {
        LOG_ERROR("Illegal device id: %d", dev);
    }
    result = get_gpu_memory_usage(dev);
//    result= nvml_get_device_memory_usage(dev);
    finish=clock();
    LOG_DEBUG("get_current_device_memory_usage:tick=%lu result=%lu\n",finish-start,result);
    return result;
}

int get_current_priority() {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 1;  // Default priority when softmig is disabled
    }
    return region_info.shared_region->priority;
}

int get_recent_kernel(){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // Default when softmig is disabled
    }
    return region_info.shared_region->recent_kernel;
}

int set_recent_kernel(int value){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // No-op when softmig is disabled
    }
    region_info.shared_region->recent_kernel=value;
    return 0;
}

int get_utilization_switch() {
    if (env_utilization_switch==1)
        return 1;
    if (env_utilization_switch==2)
        return 0;
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 0;  // Default when softmig is disabled
    }
    return region_info.shared_region->utilization_switch; 
}

void suspend_all(){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        LOG_INFO("Sending USR2 to %d",region_info.shared_region->procs[i].pid);
        kill(region_info.shared_region->procs[i].pid,SIGUSR2);
    }
}

void resume_all(){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return;  // No-op when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        LOG_INFO("Sending USR1 to %d",region_info.shared_region->procs[i].pid);
        kill(region_info.shared_region->procs[i].pid,SIGUSR1);
    }
}

int wait_status_self(int status){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 1;  // Always return "ready" when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++){
        if (region_info.shared_region->procs[i].pid==getpid()){
            if (region_info.shared_region->procs[i].status==status)
                return 1;
            else
                return 0;
        }
    }
    return -1;
}

int wait_status_all(int status){
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return 1;  // Always return "ready" when softmig is disabled
    }
    int i;
    int released = 1;
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        LOG_INFO("i=%d pid=%d status=%d",i,region_info.shared_region->procs[i].pid,region_info.shared_region->procs[i].status);
        if ((region_info.shared_region->procs[i].status!=status) && (region_info.shared_region->procs[i].pid!=getpid()))
            released = 0; 
    }
    LOG_INFO("Return released=%d",released);
    return released;
}

shrreg_proc_slot_t *find_proc_by_hostpid(int hostpid) {
    if (!is_softmig_enabled() || region_info.shared_region == NULL) {
        return NULL;  // No process found when softmig is disabled
    }
    int i;
    for (i=0;i<region_info.shared_region->proc_num;i++) {
        if (region_info.shared_region->procs[i].hostpid == hostpid) 
            return &region_info.shared_region->procs[i];
    }
    return NULL;
}


int comparelwr(const char *s1,char *s2){
    if ((s1==NULL) || (s2==NULL))
        return 1;
    if (strlen(s1)!=strlen(s2)) {
        return 1;
    }
    int i;
    for (i=0;i<strlen(s1);i++)
        if (tolower(s1[i])!=tolower(s2[i])){
            return 1;
        }
    return 0;
}
