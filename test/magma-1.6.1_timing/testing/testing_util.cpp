/*
 *  -- MAGMA (version 1.3.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2012
 *
 * @precisions normal z -> c s d
 *
 * Utilities for testing.
 * @author Mark Gates
 **/

#include "testings.h"

// --------------------
// If condition is false, print error message and exit.
// Error message is formatted using printf, using any additional arguments.
extern "C"
void magma_assert( bool condition, const char* msg, ... )
{
    if ( ! condition ) {
        va_list va;
        va_start( va, msg );
        vprintf( msg, va );
        exit(1);
    }
}

//#define __USE_GNU
//#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>

/*
 *  Forces computation to be done on a given CPU.
 *  @param: cpu - core for work to be done on
 *  @return: 0 if successful, -1 if not
 */
int map_cpu(int cpu) {
    int ret, nprocs;
    cpu_set_t cpu_mask;

    nprocs = sysconf(_SC_NPROCESSORS_CONF);//return 32, should be 16, does not matter.
    CPU_ZERO(&cpu_mask);
    CPU_SET(cpu%nprocs, &cpu_mask);

    ret = sched_setaffinity(0, sizeof(cpu_mask), &cpu_mask);
    ////int affinity = sched_getcpu();printf("Running on CPU %d\n", affinity);
    if(ret == -1) {
        perror("sched_setaffinity");
        return -1;
    }
    return 0;
}

#include <string.h>

void SetCPUFreq(long freq){//void SetCPUFreq(char *freq){
    int i;
    for(i = 0; i < 20; i++){
        char command[100], freq_buffer[7], index_buffer[2];
        strcpy(command, "echo ");
        sprintf(freq_buffer, "%ld", freq);//strcat(command, freq);
	strcat(command, freq_buffer);
        strcat(command, " > /sys/devices/system/cpu/cpu");
        sprintf(index_buffer, "%d", i);//itoa(i, buffer, 10);
        strcat(command, index_buffer);
        strcat(command, "/cpufreq/scaling_setspeed");
        //printf("%s\n",command);
        system(command);
    }
}

#include <nvml.h>

// NVIDIA NVML library function wrapper for GPU DVFS.
int SetGPUFreq(unsigned int clock_mem, unsigned int clock_core) {
    nvmlDevice_t device;//int device;
    nvmlReturn_t result;
    result = nvmlInit();
    result = nvmlDeviceGetHandleByIndex(0, &device);//cudaGetDevice(&device);
    result = nvmlDeviceSetApplicationsClocks(device, clock_mem, clock_core);//(nvmlDevice_t)device
    if(result != NVML_SUCCESS)
    {
        printf("Failed to set GPU core and memory frequencies: %s\n", nvmlErrorString(result));
        return 1;
    }
    else
    {
        nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_GRAPHICS, &clock_core);
        nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_MEM, &clock_mem);
        //printf("GPU core frequency is now set to %d MHz; GPU memory frequency is now set to %d MHz", clock_core, clock_mem);
        return 0;
    }
}

#include <sys/time.h>
#include <signal.h>

static struct itimerval itv;

static void signal_handler(int signal) {
    SetGPUFreq(2600, 705);//SetGPUFreq(2600, 758);//758 is not stable, it changes to 705 if temp. is high.
}

static void set_alarm(double s) {
    itv.it_value.tv_sec = (suseconds_t)s;
    itv.it_value.tv_usec = (suseconds_t) ((s-floor(s))*1000000.0);
    setitimer(ITIMER_REAL, &itv, NULL);
}

static void initialize_handler(void) {
    sigset_t sig;
    struct sigaction act;
    sigemptyset(&sig);
    act.sa_handler = signal_handler;
    act.sa_flags = SA_RESTART;
    act.sa_mask = sig;
    sigaction(SIGALRM, &act, NULL);
}
