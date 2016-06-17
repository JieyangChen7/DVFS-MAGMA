 #include <cstdlib>
#include <cmath>
#include "stdio.h"
#include "nvml.h"
#include <sys/time.h>
#include <signal.h>
#include <sched.h>
#include <unistd.h>

static struct itimerval itv;

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
        printf("GPU core frequency is now set to %d MHz; GPU memory frequency is now set to %d MHz", clock_core, clock_mem);
        return 0;

    }
}


void signal_handler_gpu(int signal) {
    SetGPUFreq(2600, 705);//SetGPUFreq(2600, 758);//758 is not stable, it changes to 705 if temp. is high.

    //SetCPUFreq(2500000);
    //SetGPUFreq(324, 324);
}

void signal_handler_cpu(int signal) {
    system("echo 2500000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
}

void set_alarm(double s) {
    s = s / 1000;
    itv.it_value.tv_sec = (suseconds_t)s;
    itv.it_value.tv_usec = (suseconds_t) ((s-floor(s))*1000000.0);
    itv.it_interval.tv_sec = 0;
    itv.it_interval.tv_usec = 0;
    int res = setitimer(ITIMER_REAL, &itv, NULL);
    if (res != 0) {
        printf("setitimer error! \n");
    }
}

void initialize_handler(int type) {
    sigset_t sig;
    struct sigaction act;
    int res = sigemptyset(&sig);
    if (res != 0) {
        printf("sigemptyset error! \n");
    }
    if (type == 0)//GPU
        act.sa_handler = signal_handler_gpu;
    else
        act.sa_handler = signal_handler_cpu;
    act.sa_flags = SA_RESTART;
    act.sa_mask = sig;
    res = sigaction(SIGALRM, &act, NULL);
    if (res != 0) {
        printf("sigaction error! \n");
    }
}


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