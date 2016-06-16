
int SetGPUFreq(unsigned int clock_mem, unsigned int clock_core);
void signal_handler_gpu(int signal);
void signal_handler_cpu(int signal);
void set_alarm(double s);
void initialize_handler(int type);
int map_cpu(int cpu);
