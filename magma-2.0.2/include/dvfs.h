
int SetGPUFreq(unsigned int clock_mem, unsigned int clock_core);

void restore_gpu_handler (int signal);
void dvfs_gpu_handler (int signal);

void restore_cpu_handler(int signal);
void dvfs_cpu_handler(int signal);



void set_alarm(double s);
void initialize_handler(int type);

int map_cpu(int cpu);

void dvfs_adjust(double s, char type);
void r2h_adjust(double s1, double s2, char type);

