/*
    -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2012

       @generated d Wed Nov 14 22:53:05 2012

*/
#include "common_magma.h"
#include "../testing/testing_util.cpp"
//#include "../testing/testing_dgetrf.cpp"

// === Define what BLAS to use ============================================
#define PRECISION_d
#if (GPUSHMEM <= 200) && (defined(PRECISION_s) || defined(PRECISION_d))
  #define magma_dgemm magmablas_dgemm
  #define magma_dtrsm magmablas_dtrsm
#endif
// === End defining what BLAS to use =======================================


// =========================================================================
// definitions of non-GPU-resident multi-GPU subroutines
/* non-gpu-resident interface to multiple GPUs */
extern "C" magma_int_t
magma_dgetrf_m(magma_int_t num_gpus0, magma_int_t m, magma_int_t n, double *a, magma_int_t lda,
               magma_int_t *ipiv, magma_int_t *info);

/* to apply pivoting from the previous big panel on CPU */
extern "C" magma_int_t
magma_dgetrf_piv(magma_int_t num_gpus, magma_int_t m, magma_int_t n, double *a, magma_int_t lda,
                 magma_int_t *ipiv, magma_int_t *info);
// =========================================================================


extern "C" magma_int_t
magma_dgetrf(magma_int_t m, magma_int_t n, double *a, magma_int_t lda, 
             magma_int_t *ipiv, magma_int_t *info)
{
/*  -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2012

    Purpose
    =======
    DGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.  This version does not
    require work space on the GPU passed as input. GPU memory is allocated
    in the routine.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) DOUBLE_PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    =====================================================================    */

#define inAT(i,j) (dAT + (i)*nb*ldda + (j)*nb)

    double *dAT, *dA, *da, *work;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t     iinfo, nb;

    *info = 0;

    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    //nb = magma_get_dgetrf_nb(m);//printf("Wenyi%d",nb);
    nb = 64;//optimal block size

    if ( (nb <= 1) || (nb >= min(m,n)) ) {
        /* Use CPU code. */
        lapackf77_dgetrf(&m, &n, a, &lda, ipiv, info);
    } else {
        /* Use hybrid blocked code. */
        magma_int_t maxm, maxn, ldda, maxdim;
        magma_int_t i, rows, cols, s = min(m, n)/nb;
        
        magma_int_t num_gpus = magma_num_gpus();
        if ( num_gpus > 1 ) {
          /* call multi-GPU non-GPU-resident interface  */
          magma_int_t rval = magma_dgetrf_m(num_gpus, m, n, a, lda, ipiv, info);
          if( *info >= 0 ) magma_dgetrf_piv(num_gpus, m, n, a, lda, ipiv, info);
          return *info;
        }

        maxm = ((m + 31)/32)*32;
        maxn = ((n + 31)/32)*32;
        maxdim = max(maxm, maxn);

        ldda = maxn;
        work = a;

        if (maxdim*maxdim < 2*maxm*maxn)
        {
            if (MAGMA_SUCCESS != magma_dmalloc( &dA, nb*maxm + maxdim*maxdim )) {
                        /* alloc failed so call non-GPU-resident version */ 
                        magma_int_t rval = magma_dgetrf_m(num_gpus, m, n, a, lda, ipiv, info);
                        if( *info >= 0 ) magma_dgetrf_piv(num_gpus, m, n, a, lda, ipiv, info);
                        return *info;
            }
            da = dA + nb*maxm;
            
            ldda = maxdim;
            magma_dsetmatrix( m, n, a, lda, da, ldda );
            
            dAT = da;
            magmablas_dinplace_transpose( dAT, ldda, ldda );
        }
        else
        {
            if (MAGMA_SUCCESS != magma_dmalloc( &dA, (nb + maxn)*maxm )) {
                        /* alloc failed so call non-GPU-resident version */
                        magma_int_t rval = magma_dgetrf_m(num_gpus, m, n, a, lda, ipiv, info);
                        if( *info >= 0 ) magma_dgetrf_piv(num_gpus, m, n, a, lda, ipiv, info);
                        return *info;
            }
            da = dA + nb*maxm;
            
            magma_dsetmatrix( m, n, a, lda, da, maxm );
            
            if (MAGMA_SUCCESS != magma_dmalloc( &dAT, maxm*maxn )) {
                        /* alloc failed so call non-GPU-resident version */
                        magma_free( dA );
                        magma_int_t rval = magma_dgetrf_m(num_gpus, m, n, a, lda, ipiv, info);
                        if( *info >= 0 ) magma_dgetrf_piv(num_gpus, m, n, a, lda, ipiv, info);
                        return *info;
            }

            magmablas_dtranspose2( dAT, ldda, da, maxm, m, n );
        }
        
        lapackf77_dgetrf( &m, &nb, work, &lda, ipiv, &iinfo);

	//real_Double_t cpu_time, gpu_time1, gpu_time2, gpu_time3, download_copy_time, upload_copy_time;
	//real_Double_t total_cpu_time = 0, total_gpu_time = 0, total_copy_time = 0, main_loop_time = 0;
        float cpu_time_cuda_temp, total_cpu_time_cuda = 0;
	float download_copy_time_cuda_temp, upload_copy_time_cuda_temp, total_copy_time_cuda = 0;
	float main_loop_time_cuda = 0;
	float per_iter_time_cuda_temp;
	float gpu_time1_cuda_temp, gpu_time2_cuda_temp, gpu_time3_cuda_temp, total_gpu_time_cuda = 0;

	cudaEvent_t start_main_loop, stop_main_loop;
	cudaEvent_t start_per_iter, stop_per_iter;
	cudaEvent_t start_download_copy, stop_download_copy;
	cudaEvent_t start_gpu1, stop_gpu1;
	cudaEvent_t start_cpu, stop_cpu;
        cudaEvent_t start_upload_copy, stop_upload_copy;
	cudaEvent_t start_gpu2, stop_gpu2;
	cudaEvent_t start_gpu3, stop_gpu3;

	#define TIME_MEASUREMENT1 0
	#define TIME_MEASUREMENT2 0//Only print CPU/GPU time for the first iteration, no data copy time
	#define ALGORITHMIC_SLACK_PREDICTION 0
	#define CPU_SLACK_RECLAMATION_ENABLED 0//CPU DVFS for eliminating CPU slack of each iteration
	#define GPU_SLACK_RECLAMATION_ENABLED 0//GPU DVFS for eliminating GPU slack of each iteration
	#define CPU_AGGR_DVFS_ENABLED 0//does not save energy by down F/V for memory-bound due to T up.
	#define RACE_TO_HALT_ENABLED 0//Embarrassingly saves energy in terms of GFLOPS/W with minor performance loss, if enabled together with TIME_MEASUREMENT for synchronization. Otherwise HUGE performance loss. In terms of GFLOPS/W, no matter if enabled together with TIME_MEASUREMENT, energy can be saved.
	#define COMPARED_CPUSPEED 0//OS level race-to-halt approach

	//SetGPUFreq(324, 324);
	//SetGPUFreq(2600, 614);//system("sudo nvidia-smi -ac 2600,614 > gpu_dvfs.log");
	//if(RACE_TO_HALT_ENABLED) SetGPUFreq(2600, 758);//does not need to, done in testing_dgetrf.cpp
	/*
	system("echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu8/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu9/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu10/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu11/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu12/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu13/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu14/cpufreq/scaling_setspeed");
	system("echo 1200000 > /sys/devices/system/cpu/cpu15/cpufreq/scaling_setspeed");
	*/
	if(RACE_TO_HALT_ENABLED) SetCPUFreq("1200000");
	if(COMPARED_CPUSPEED) system("sudo /home/lchen/CPUSpeed/cpuspeed-1.5/cpuspeed -d");

	int gpu_switched_flag[10] = {0,0,0,0,0,0,0,0,0,0};//int gpu_switched_flag1 = 0, gpu_switched_flag2 = 0, gpu_switched_flag3 = 0, gpu_switched_flag4 = 0, gpu_switched_flag5 = 0;
	int cpu_switched_flag[10] = {0,0,0,0,0,0,0,0,0,0};//int cpu_switched_flag1 = 0, cpu_switched_flag2 = 0, cpu_switched_flag3 = 0, cpu_switched_flag4 = 0, cpu_switched_flag5 = 0, cpu_switched_flag6 = 0, cpu_switched_flag7 = 0, cpu_switched_flag8 = 0;

	int TIME_MEASUREMENT2_flag1 = -1, TIME_MEASUREMENT2_flag2 = 0;
	int test_flag = 0;
	double gpu_time_iter0 = 0, cpu_time_iter0 = 0;
	int gpu_time_iter0_flag = 0, cpu_time_iter0_flag = 0;
	double gpu_time_lowest_freq_iter0 = 0.037763;//iter 1
	double gpu_time_last_iter = 0, cpu_time_last_iter = 0;
	double gpu_time_pred = 0, cpu_time_pred = 0;
	double ratio_slack_pred = 0;
	double diff_total_cpu = 0, diff_total_gpu = 0, diff_total_slack = 0;

	if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
	//main_loop_time = magma_wtime();
        cudaEventCreate(&start_main_loop);
        cudaEventCreate(&stop_main_loop);
        cudaEventRecord(start_main_loop, 0);
	}

        for( i = 0; i < s; i++ )
        {
            if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
            //per_iter_time = magma_wtime();
            cudaEventCreate(&start_per_iter);
            cudaEventCreate(&stop_per_iter);
            cudaEventRecord(start_per_iter, 0);
            }
            if(i==1)//i==0 skipped by the program itself.//Only record the first iteration execution time.
            {
                if(TIME_MEASUREMENT2)TIME_MEASUREMENT2_flag1 = 1;//#define TIME_MEASUREMENT 1//TIME_MEASUREMENT = 1;
            }
            else if(i>1)
            {
                if(TIME_MEASUREMENT2)TIME_MEASUREMENT2_flag1 = 0;//#define TIME_MEASUREMENT 0//TIME_MEASUREMENT = 0;
                //if(gpu_time_iter0 < cpu_time_iter0)
		if(TIME_MEASUREMENT2)if(!TIME_MEASUREMENT2_flag2)
		{
			printf("iter %d: cpu_time_iter0=%f, gpu_time_iter0=%f\n", i-1, cpu_time_iter0, gpu_time_iter0);
			TIME_MEASUREMENT2_flag2 = 1;
		}

		if(ALGORITHMIC_SLACK_PREDICTION)
		{
			ratio_slack_pred = 1.0 - (double)nb/(m-i*nb);
			//printf("iter%d: %f\n", i, ratio);

			cpu_time_pred = cpu_time_pred*ratio_slack_pred;
			//diff_total_cpu += (cpu_time_pred - cpu_time_last_iter)/cpu_time_last_iter;
			gpu_time_pred = gpu_time_pred*ratio_slack_pred*ratio_slack_pred;
			//diff_total_gpu += (gpu_time_pred - gpu_time_last_iter)/gpu_time_last_iter;
			printf("iter %d: cpu_time_pred = %f\n", i, cpu_time_pred);
			printf("iter %d: gpu_time_pred = %f\n", i, gpu_time_pred);
			//diff_total_slack += ((cpu_time_pred - gpu_time_pred) - (cpu_time_last_iter - gpu_time_last_iter))/(cpu_time_last_iter - gpu_time_last_iter);//(slack_pred - slack_measured) / slack_measured
			if(cpu_time_last_iter - gpu_time_last_iter > 0) printf("!!Border found at iter %d\n", i-1);
		}

		/*if(GPU_SLACK_RECLAMATION_ENABLED)
		{
			double ratio_split_freq = (cpu_time_pred - gpu_time_pred)/(gpu_time_pred*(7.21-1));//gpu_time_lowest_freq_iter0 / gpu_time_highest_freq_iter0 = 7.21
			printf("iter %d: ratio_split_freq = %f\n", i, ratio_split_freq);

			gpu_time_lowest_freq_iter0 = gpu_time_lowest_freq_iter0*ratio_slack_pred*ratio_slack_pred;
			double seconds_until_interrupt = gpu_time_lowest_freq_iter0 * ratio_split_freq;
			//double DVFS_overhead_delay_adjustment1 = 0.0015;
			//double DVFS_overhead_delay_adjustment2 = 0.02/(1-ratio_slack_pred*ratio_slack_pred);
			//if(ratio_split_freq < 1) seconds_until_interrupt = seconds_until_interrupt * DVFS_overhead_delay_adjustment2;//seconds_until_interrupt = seconds_until_interrupt - DVFS_overhead_delay_adjustment1;
			initialize_handler();
			SetGPUFreq(324, 324);
			if(ratio_split_freq < 1) set_alarm(seconds_until_interrupt);
			else set_alarm(cpu_time_pred);
			//SetGPUFreq(324, 324);
		}*/
            }

            // download i-th panel
            cols = maxm - i*nb;
            
            if (i>0){
		if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
		//download_copy_time = magma_wtime();
		cudaEventCreate(&start_download_copy);
		cudaEventCreate(&stop_download_copy);
		cudaEventRecord(start_download_copy, 0);
		////kernel<<<grid,threads>>> (d_odata, d_idata, size_x, size_y, NUM_REPS);
		}
                magmablas_dtranspose( dA, cols, inAT(i,i), ldda, nb, cols );
                magma_dgetmatrix( m-i*nb, nb, dA, cols, work, lda );
		if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
		//download_copy_time = magma_wtime() - download_copy_time;
		//total_copy_time += download_copy_time;
		cudaEventRecord(stop_download_copy, 0);
		cudaEventSynchronize(stop_download_copy);
		cudaEventElapsedTime(&download_copy_time_cuda_temp, start_download_copy, stop_download_copy);
		cudaEventDestroy(start_download_copy);
		cudaEventDestroy(stop_download_copy);
		total_copy_time_cuda += download_copy_time_cuda_temp;
		}

                // make sure that gpu queue is empty
                magma_device_sync();

		if(TIME_MEASUREMENT1 || TIME_MEASUREMENT2_flag1){
		//gpu_time1 = magma_wtime();
                cudaEventCreate(&start_gpu1);
                cudaEventCreate(&stop_gpu1);
                cudaEventRecord(start_gpu1, 0);
		}

		// Race-to-halt.
		//if(RACE_TO_HALT_ENABLED) SetGPUFreq(2600, 758);//system("sudo nvidia-smi -ac 2600,758 > gpu_dvfs.log");
		// dtrsm is on the critical path, so cannot be slowed down.
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                             n - (i+1)*nb, nb, 
                             c_one, inAT(i-1,i-1), ldda, 
                                    inAT(i-1,i+1), ldda );

		if(i > 1)
		{
			if(GPU_SLACK_RECLAMATION_ENABLED)
                	{       
                        	double ratio_split_freq = (cpu_time_pred - gpu_time_pred)/(gpu_time_pred*(7.21-1));//gpu_time_lowest_freq_iter0 / gpu_time_highest_freq_iter0 = 7.21 
                        	printf("iter %d: ratio_split_freq = %f\n", i, ratio_split_freq);

                        	gpu_time_lowest_freq_iter0 = gpu_time_lowest_freq_iter0*ratio_slack_pred*ratio_slack_pred;
                        	double seconds_until_interrupt = gpu_time_lowest_freq_iter0 * ratio_split_freq;
                        	double DVFS_overhead_delay_adjustment1 = 0.0026;
                        	//double DVFS_overhead_delay_adjustment2 = 0.02/(1-ratio_slack_pred*ratio_slack_pred);
                        	//double DVFS_overhead_delay_adjustment3 = 0.4;
                        	if(ratio_split_freq < 1) seconds_until_interrupt -= DVFS_overhead_delay_adjustment1;//seconds_until_interrupt *= DVFS_overhead_delay_adjustment3;
                        	initialize_handler();
                        	SetGPUFreq(324, 324);
                        	if(ratio_split_freq < 1) set_alarm(seconds_until_interrupt);
                        	else set_alarm(cpu_time_pred);
                        	//SetGPUFreq(324, 324);
                        }
		}

		// slow down trailing matrix update only.
		if(GPU_SLACK_RECLAMATION_ENABLED)
		{
			if(i < 47 && !gpu_switched_flag[0])//s/2
			{
				//system("echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
				//SetGPUFreq(2600, 614);//system("sudo nvidia-smi -ac 2600,614 > gpu_dvfs.log");
				gpu_switched_flag[0] = 1;
			}
			if(i >= 47 && i < 88 && !gpu_switched_flag[1])//s/2
			{
				//system("echo 1300000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
				//SetGPUFreq(324, 324);//system("sudo nvidia-smi -ac 324,324 > gpu_dvfs.log");
				gpu_switched_flag[1] = 1;
			}
			//else if(i >= 58 && !gpu_switched_flag[2])//s/2
			//{
			//	system("sudo nvidia-smi -ac 324,324 > gpu_dvfs.log");
			//	gpu_switched_flag[2] = 1;
			//}
		}
		////if(GPU_SLACK_RECLAMATION_ENABLED) SetGPUFreq(324, 324);//system("sudo nvidia-smi -ac 324,324 > gpu_dvfs.log");
		////if(GPU_SLACK_RECLAMATION_ENABLED) SetGPUFreq(2600, 614);//system("sudo nvidia-smi -ac 2600,614 > gpu_dvfs.log");
		if(ALGORITHMIC_SLACK_PREDICTION){
                //gpu_time1 = magma_wtime();
                cudaEventCreate(&start_gpu1);
                cudaEventCreate(&stop_gpu1);
                cudaEventRecord(start_gpu1, 0);
                }
                magma_dgemm( MagmaNoTrans, MagmaNoTrans, 
                             n-(i+1)*nb, m-i*nb, nb, 
                             c_neg_one, inAT(i-1,i+1), ldda, 
                                        inAT(i,  i-1), ldda, 
                             c_one,     inAT(i,  i+1), ldda );
		if(TIME_MEASUREMENT1 || TIME_MEASUREMENT2_flag1 || ALGORITHMIC_SLACK_PREDICTION){
                //gpu_time1 = magma_wtime() - gpu_time1;
                //total_gpu_time += gpu_time1;
                cudaEventRecord(stop_gpu1, 0);
                cudaEventSynchronize(stop_gpu1);
                cudaEventElapsedTime(&gpu_time1_cuda_temp, start_gpu1, stop_gpu1);
                cudaEventDestroy(start_gpu1);
                cudaEventDestroy(stop_gpu1);
                total_gpu_time_cuda += gpu_time1_cuda_temp;
                if(!gpu_time_iter0_flag)
                {
                        gpu_time_iter0 = gpu_time1_cuda_temp/1000;
                        gpu_time_pred = gpu_time_iter0;
                        gpu_time_iter0_flag = 1;
                }
                gpu_time_last_iter = gpu_time1_cuda_temp/1000;
                diff_total_gpu += (gpu_time_pred - gpu_time_last_iter)/gpu_time_last_iter;
                }
		if(GPU_SLACK_RECLAMATION_ENABLED)
		{
			if(i >= 88 && !gpu_switched_flag[2])
			{
				//system("echo 2600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
				//SetGPUFreq(2600, 614);//system("sudo nvidia-smi -ac 2600,758 > gpu_dvfs.log");
				gpu_switched_flag[2] = 1;
			}
			if(i >= 229 && !gpu_switched_flag[3])
			{
				//SetGPUFreq(2600, 614);//system("sudo nvidia-smi -ac 2600,758 > gpu_dvfs.log");
				gpu_switched_flag[3] = 1;
			}
			if(i >= 252 && !gpu_switched_flag[4])
			{
				//SetGPUFreq(324, 324);//system("sudo nvidia-smi -ac 324,324 > gpu_dvfs.log");
				gpu_switched_flag[4] = 1;
			}
		}
		// Race-to-halt. Performance degrades a lot, so cannot really save energy.
		//if(RACE_TO_HALT_ENABLED) SetGPUFreq(324, 324);//system("sudo nvidia-smi -ac 324,324 > gpu_dvfs.log");

		if(CPU_AGGR_DVFS_ENABLED){
		// should not place setcpuspeed within the program -> CPU performance degrades a lot.
		/*
		system("/apps/power-bench/setcpuspeed sandy 1200000");//change all core speed.
		system("echo 2600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
		*/
		// perform DVFS switches in the program -> a little CPU perf down + a lot of overhead
		//SetCPUFreq("1200000");
			if(i == 1 && !cpu_switched_flag[0])
			{
				system("echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
				cpu_switched_flag[0] = 1;
			}
		}
		// Race-to-halt. Performance degrades a little, but can save energy in GFLOPS/W.
		if(RACE_TO_HALT_ENABLED) SetGPUFreq(324, 324);
		//if(RACE_TO_HALT_ENABLED) SetCPUFreq("2600000");
		if(RACE_TO_HALT_ENABLED)
		//	if(!test_flag){
			system("echo 2600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
		//	test_flag=1;}
		if(CPU_SLACK_RECLAMATION_ENABLED)
                {
                        if(i < 69 && !cpu_switched_flag[0])//s/2
                        {
                                system("echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
                                //SetGPUFreq(2600, 758);//system("sudo nvidia-smi -ac 2600,614 > gpu_dvfs.log");
                                cpu_switched_flag[0] = 1;
                        }
                        if(i >= 69 && i < 88 && !cpu_switched_flag[1])//s/2
                        {
                                system("echo 1300000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
                                cpu_switched_flag[1] = 1;
                        }
			if(i >= 88 && i < 103 && !cpu_switched_flag[2])//s/2
                        {
                                system("echo 1400000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
                                cpu_switched_flag[2] = 1;
                        }
			if(i >= 103 && i < 121 && !cpu_switched_flag[3])//s/2
                        {
                                system("echo 1500000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
                                cpu_switched_flag[3] = 1;
                        }
			if(i >= 121 && i < 133 && !cpu_switched_flag[4])//s/2
                        {
                                system("echo 1600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
                                cpu_switched_flag[4] = 1;
                        }
		}
		if(TIME_MEASUREMENT1 || TIME_MEASUREMENT2_flag1 || ALGORITHMIC_SLACK_PREDICTION){
                //cpu_time = magma_wtime();
                cudaEventCreate(&start_cpu);
                cudaEventCreate(&stop_cpu);
                cudaEventRecord(start_cpu, 0);
                }
                // do the cpu part
                rows = m - i*nb;
                lapackf77_dgetrf( &rows, &nb, work, &lda, ipiv+i*nb, &iinfo);
		if(TIME_MEASUREMENT1 || TIME_MEASUREMENT2_flag1 || ALGORITHMIC_SLACK_PREDICTION){
                //cpu_time = magma_wtime() - cpu_time;
                //total_cpu_time += cpu_time;
                cudaEventRecord(stop_cpu, 0);
                cudaEventSynchronize(stop_cpu);
                cudaEventElapsedTime(&cpu_time_cuda_temp, start_cpu, stop_cpu);
                cudaEventDestroy(start_cpu);
                cudaEventDestroy(stop_cpu);
                total_cpu_time_cuda += cpu_time_cuda_temp;
                if(!cpu_time_iter0_flag)
                {
                        cpu_time_iter0 = cpu_time_cuda_temp/1000;
                        cpu_time_pred = cpu_time_iter0;
                        cpu_time_iter0_flag = 1;
                }
                cpu_time_last_iter = cpu_time_cuda_temp/1000;
                diff_total_cpu += (cpu_time_pred - cpu_time_last_iter)/cpu_time_last_iter;
                diff_total_slack += ((cpu_time_pred - gpu_time_pred) - (cpu_time_last_iter - gpu_time_last_iter))/(cpu_time_last_iter - gpu_time_last_iter);//(slack_pred - slack_measured) / slack_measured
                }
		if(CPU_SLACK_RECLAMATION_ENABLED)
                {
                        if(i >= 133 && !cpu_switched_flag[5])
                        {
                                system("echo 2600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
                                //SetGPUFreq(2600, 614);//system("sudo nvidia-smi -ac 2600,758 > gpu_dvfs.log");
                                cpu_switched_flag[5] = 1;
                        }
		}
                if(RACE_TO_HALT_ENABLED) SetGPUFreq(2600, 758);
                //if(RACE_TO_HALT_ENABLED) SetCPUFreq("1200000");
                if(RACE_TO_HALT_ENABLED) system("echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
		if(CPU_AGGR_DVFS_ENABLED){
		//system("/apps/power-bench/setcpuspeed sandy 2600000");//change all core speed.
		//SetCPUFreq("2600000");
			if(i == s-1 && !cpu_switched_flag[1])
			{
				system("echo 2600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
				cpu_switched_flag[1] = 1;
			}
                }
            }
            if (*info == 0 && iinfo > 0)
                *info = iinfo + i*nb;
            magmablas_dpermute_long2( ldda, dAT, ldda, ipiv, nb, i*nb );

            if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
            //upload_copy_time = magma_wtime();
            cudaEventCreate(&start_upload_copy);
            cudaEventCreate(&stop_upload_copy);
            cudaEventRecord(start_upload_copy, 0);
            }
            // upload i-th panel
            magma_dsetmatrix( m-i*nb, nb, work, lda, dA, cols );
            magmablas_dtranspose( inAT(i,i), ldda, dA, cols, cols, nb);
            if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
            //upload_copy_time = magma_wtime() - upload_copy_time;
            //total_copy_time += upload_copy_time;
            cudaEventRecord(stop_upload_copy, 0);
            cudaEventSynchronize(stop_upload_copy);
            cudaEventElapsedTime(&upload_copy_time_cuda_temp, start_upload_copy, stop_upload_copy);
            cudaEventDestroy(start_upload_copy);
            cudaEventDestroy(stop_upload_copy);
            total_copy_time_cuda += upload_copy_time_cuda_temp;
            }

            // do the small non-parallel computations
            if (s > (i+1)){
		if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
		//gpu_time2 = magma_wtime();
            	cudaEventCreate(&start_gpu2);
                cudaEventCreate(&stop_gpu2);
                cudaEventRecord(start_gpu2, 0);
		}
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                             nb, nb, 
                             c_one, inAT(i, i  ), ldda,
                                    inAT(i, i+1), ldda);
                magma_dgemm( MagmaNoTrans, MagmaNoTrans, 
                             nb, m-(i+1)*nb, nb, 
                             c_neg_one, inAT(i,   i+1), ldda,
                                        inAT(i+1, i  ), ldda, 
                             c_one,     inAT(i+1, i+1), ldda );
		if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
		//gpu_time2 = magma_wtime() - gpu_time2;
		//total_gpu_time += gpu_time2;
		cudaEventRecord(stop_gpu2, 0);
                cudaEventSynchronize(stop_gpu2);
                cudaEventElapsedTime(&gpu_time2_cuda_temp, start_gpu2, stop_gpu2);
                cudaEventDestroy(start_gpu2);
                cudaEventDestroy(stop_gpu2);
		//total_gpu_time_cuda += gpu_time2_cuda_temp;
		}
            }
            else{
		if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
		//gpu_time3 = magma_wtime();
                cudaEventCreate(&start_gpu3);
                cudaEventCreate(&stop_gpu3);
                cudaEventRecord(start_gpu3, 0);
		}
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                             n-s*nb, nb,
                             c_one, inAT(i, i  ), ldda,
                                    inAT(i, i+1), ldda);
                magma_dgemm( MagmaNoTrans, MagmaNoTrans, 
                             n-(i+1)*nb, m-(i+1)*nb, nb,
                             c_neg_one, inAT(i,   i+1), ldda,
                                        inAT(i+1, i  ), ldda, 
                             c_one,     inAT(i+1, i+1), ldda );
		if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
		//gpu_time3 = magma_wtime() - gpu_time3;
		//total_gpu_time += gpu_time3;
		cudaEventRecord(stop_gpu3, 0);
                cudaEventSynchronize(stop_gpu3);
                cudaEventElapsedTime(&gpu_time3_cuda_temp, start_gpu3, stop_gpu3);
                cudaEventDestroy(start_gpu3);
                cudaEventDestroy(stop_gpu3);
		//total_gpu_time_cuda += gpu_time3_cuda_temp;
		}
            }

            if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
            //per_iter_time = magma_wtime() - per_iter_time;
            cudaEventRecord(stop_per_iter, 0);
            cudaEventSynchronize(stop_per_iter);
            cudaEventElapsedTime(&per_iter_time_cuda_temp, start_per_iter, stop_per_iter);
            cudaEventDestroy(start_per_iter);
            cudaEventDestroy(stop_per_iter);
            }

            if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
            //printf("iter %d: cpu_time = %.6f\n", i, cpu_time);
            //printf("iter %d: gpu_time1 = %.6f\n", i, gpu_time1);
            //printf("iter %d: gpu_time2 = %.6f\n", i, gpu_time2);
            //printf("iter %d: gpu_time3 = %.6f\n", i, gpu_time3);
            //printf("iter %d: download_copy_time = %.6f\n", i, download_copy_time);
            //printf("iter %d: upload_copy_time = %.6f\n", i, upload_copy_time);
            printf("iter %d: cpu_time_cuda = %.6f\n", i, cpu_time_cuda_temp/1000);
            printf("iter %d: gpu_time_cuda = %.6f\n", i, gpu_time1_cuda_temp/1000);
            //printf("iter %d: gpu_time2_cuda = %.6f\n", i, gpu_time2_cuda_temp/1000);
            //printf("iter %d: gpu_time3_cuda = %.6f\n", i, gpu_time3_cuda_temp/1000);
            printf("iter %d: download_copy_time_cuda = %.6f\n", i, download_copy_time_cuda_temp/1000);
            printf("iter %d: upload_copy_time_cuda = %.6f\n", i, upload_copy_time_cuda_temp/1000);
            printf("iter %d: per_iter_time_cuda = %.6f\n\n", i, per_iter_time_cuda_temp/1000);
            }
        }

	if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
	//main_loop_time = magma_wtime() - main_loop_time;
	cudaEventRecord(stop_main_loop, 0);
        cudaEventSynchronize(stop_main_loop);
        cudaEventElapsedTime(&main_loop_time_cuda, start_main_loop, stop_main_loop);
        cudaEventDestroy(start_main_loop);
        cudaEventDestroy(stop_main_loop);
	}

	if(TIME_MEASUREMENT1 || ALGORITHMIC_SLACK_PREDICTION){
	//printf("total_cpu_time = %.6f\n", total_cpu_time);
	//printf("total_gpu_time = %.6f\n", total_gpu_time);
	//printf("total_copy_time = %.6f\n", total_copy_time);
	//printf("main_loop_time = %.6f\n", main_loop_time);
	printf("total_cpu_time_cuda = %.6f\n", total_cpu_time_cuda/1000);
	printf("total_gpu_time_cuda = %.6f\n", total_gpu_time_cuda/1000);
        printf("total_copy_time_cuda = %.6f\n", total_copy_time_cuda/1000);
        printf("main_loop_time_cuda = %.6f\n", main_loop_time_cuda/1000);
	printf("Normalized difference of predicted CPU runtime per iteration is: %.6f\%\n", 100*diff_total_cpu/(m/nb));
	printf("Normalized difference of predicted GPU runtime per iteration is: %.6f\%\n", 100*diff_total_gpu/(m/nb));
	printf("Normalized difference of predicted slack per iteration is: %.6f\%\n", 100*diff_total_slack/(m/nb));
	}

	////system("sudo nvidia-smi -ac 2600,758 > gpu_dvfs.log");
	//if(RACE_TO_HALT_ENABLED) SetGPUFreq(2600, 758);
	/*
	system("echo 2600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu2/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu3/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu4/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu5/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu6/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu7/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu8/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu9/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu10/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu11/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu12/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu13/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu14/cpufreq/scaling_setspeed");
	system("echo 2600000 > /sys/devices/system/cpu/cpu15/cpufreq/scaling_setspeed");
	*/
	if(RACE_TO_HALT_ENABLED) SetCPUFreq("2600000");
	if(COMPARED_CPUSPEED) system("sudo killall -9 cpuspeed");

        magma_int_t nb0 = min(m - s*nb, n - s*nb);
        if ( nb0 > 0 ) {
            rows = m - s*nb;
            cols = maxm - s*nb;
    
            magmablas_dtranspose2( dA, cols, inAT(s,s), ldda, nb0, rows);
            magma_dgetmatrix( rows, nb0, dA, cols, work, lda );
    
            // make sure that gpu queue is empty
            magma_device_sync();
    
            // do the cpu part
            lapackf77_dgetrf( &rows, &nb0, work, &lda, ipiv+s*nb, &iinfo);
            if (*info == 0 && iinfo > 0)
                *info = iinfo + s*nb;
            magmablas_dpermute_long2( ldda, dAT, ldda, ipiv, nb0, s*nb );
    
            magma_dsetmatrix( rows, nb0, work, lda, dA, cols );
            magmablas_dtranspose2( inAT(s,s), ldda, dA, cols, rows, nb0);
    
            magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit, 
                         n-s*nb-nb0, nb0,
                         c_one, inAT(s, s),     ldda, 
                                inAT(s, s)+nb0, ldda);
        }
        
        if (maxdim*maxdim< 2*maxm*maxn){
            magmablas_dinplace_transpose( dAT, ldda, ldda );
            magma_dgetmatrix( m, n, da, ldda, a, lda );
        } else {
            magmablas_dtranspose2( da, maxm, dAT, ldda, n, m );
            magma_dgetmatrix( m, n, da, maxm, a, lda );
            magma_free( dAT );
        }

        magma_free( dA );
    }
    
    return *info;
} /* magma_dgetrf */

#undef inAT
