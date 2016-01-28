/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Stan Tomov
       @generated from zgetrf.cpp normal z -> d, Fri Jan 30 19:00:14 2015
*/
#include "common_magma.h"
#include "../testing/testing_util.cpp"
#include "cuda_profiler_api.h"
#include "papi.h"
#include <iostream>
using namespace std;

/**
    Purpose
    -------
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
    
    If the current stream is NULL, this version replaces it with a new
    stream to overlap computation with communication.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       DOUBLE_PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @ingroup magma_dgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dgetrf(
    magma_int_t m, magma_int_t n, double *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t *info)
{
    #define dAT(i_, j_) (dAT + (i_)*nb*ldda + (j_)*nb)

    double *dAT, *dA, *da, *work;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t     iinfo, nb;

    /* Check arguments */
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

    /* Function Body */
    nb = magma_get_dgetrf_nb(m);
    //cout << "nb="<<nb<<endl; 
    nb = 100;//optimal

    if ( (nb <= 1) || (nb >= min(m,n)) ) {
        /* Use CPU code. */
        lapackf77_dgetrf(&m, &n, A, &lda, ipiv, info);
    } else {
        /* Use hybrid blocked code. */
        magma_int_t maxm, maxn, ldda, maxdim;
        magma_int_t i, j, rows, cols, s = min(m, n)/nb;
        
        maxm = ((m + 31)/32)*32;
        maxn = ((n + 31)/32)*32;
        maxdim = max(maxm, maxn);

        /* set number of GPUs */
        magma_int_t ngpu = magma_num_gpus();
        if ( ngpu > 1 ) {
            /* call multi-GPU non-GPU-resident interface  */
            magma_dgetrf_m(ngpu, m, n, A, lda, ipiv, info);
            return *info;
        }
        //cout << "nb="<<nb<<endl; 
        /* explicitly checking the memory requirement */
        size_t freeMem, totalMem;
        cudaMemGetInfo( &freeMem, &totalMem );
        freeMem /= sizeof(double);
        //cout << "freeMem="<<freeMem<<endl; 
        int h = 1+(2+ngpu), ngpu2 = ngpu;
        //cout << "h="<<h<<endl; 
        
        int NB = (magma_int_t)(0.8*freeMem/maxm-h*nb);
        //cout << "NB="<<NB<<endl; 
        const char* ngr_nb_char = getenv("MAGMA_NGR_NB");
        if ( ngr_nb_char != NULL )
            NB = max( nb, min( NB, atoi(ngr_nb_char) ) );

        if ( ngpu > ceil((double)NB/nb) ) {
            ngpu2 = (int)ceil((double)NB/nb);
            h = 1+(2+ngpu2);
            NB = (magma_int_t)(0.8*freeMem/maxm-h*nb);
        }
        if ( ngpu2*NB < n ) {
            /* require too much memory, so call non-GPU-resident version */
            magma_dgetrf_m(ngpu, m, n, A, lda, ipiv, info);
            return *info;
        }
        //cout << "NB="<<NB<<endl; 
        //cout << "nb="<<nb<<endl; 
        ldda = maxn;
        work = A;
        if (maxdim*maxdim < 2*maxm*maxn) {
            // if close to square, allocate square matrix and transpose in-place
            if (MAGMA_SUCCESS != magma_dmalloc( &dA, nb*maxm + maxdim*maxdim )) {
                /* alloc failed so call non-GPU-resident version */
                magma_dgetrf_m(ngpu, m, n, A, lda, ipiv, info);
                return *info;
            }
            da = dA + nb*maxm;
            
            ldda = maxdim;
            magma_dsetmatrix( m, n, A, lda, da, ldda );
            
            dAT = da;
            magmablas_dtranspose_inplace( ldda, dAT, ldda );
        }
        else {
            // if very rectangular, allocate dA and dAT and transpose out-of-place
            if (MAGMA_SUCCESS != magma_dmalloc( &dA, (nb + maxn)*maxm )) {
                /* alloc failed so call non-GPU-resident version */
                magma_dgetrf_m(ngpu, m, n, A, lda, ipiv, info);
                return *info;
            }
            da = dA + nb*maxm;
            
            magma_dsetmatrix( m, n, A, lda, da, maxm );
            
            if (MAGMA_SUCCESS != magma_dmalloc( &dAT, maxm*maxn )) {
                /* alloc failed so call non-GPU-resident version */
                magma_free( dA );
                magma_dgetrf_m(ngpu, m, n, A, lda, ipiv, info);
                return *info;
            }

            magmablas_dtranspose( m, n, da, maxm, dAT, ldda );
        }
        //cout << "nb="<<nb<<endl; 
        float real_time = 0.0;
		float proc_time = 0.0;
		long long flpins = 0.0;
		float mflops = 0.0;
		
//		//PAPI timing start
//		if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
//			cout << "PAPI ERROR" << endl;
//			return -1;
//		} 
//      
        
        lapackf77_dgetrf( &m, &nb, A, &lda, ipiv, &iinfo);
        
        
      
//        if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
//			cout << "PAPI ERROR" << endl;
//			return -1;
//		} 
//		
//		cout<<"cpu time:"<<real_time<<endl;
//		PAPI_shutdown();
//		
//		real_time = 0.0;
//		proc_time = 0.0;
//		flpins = 0.0;
//		mflops = 0.0;
//        
        

        /* Define user stream if current stream is NULL */
        magma_queue_t stream[2];
        
        magma_queue_t orig_stream;
        magmablasGetKernelStream( &orig_stream );

        magma_queue_create( &stream[0] );
        if (orig_stream == NULL) {
            magma_queue_create( &stream[1] );
            magmablasSetKernelStream(stream[1]);
        }
        else {
            stream[1] = orig_stream;
        }
  //      cout << "nb="<<nb<<endl; 
/***********
 * GreenLA *
 ***********/

        float cpu_time_cuda_temp, total_cpu_time_cuda = 0;
        float download_copy_time_cuda_temp, upload_copy_time_cuda_temp, total_copy_time_cuda = 0;
        float main_loop_time_cuda = 0;
        float per_iter_time_cuda_temp;
        float gpu_time_cuda_temp, total_gpu_time_cuda = 0;

        cudaEvent_t start_main_loop, stop_main_loop;
        cudaEvent_t start_per_iter, stop_per_iter;
        cudaEvent_t start_download_copy, stop_download_copy;
        cudaEvent_t start_gpu, stop_gpu;
        cudaEvent_t start_cpu, stop_cpu;
        cudaEvent_t start_upload_copy, stop_upload_copy;

        double total_slack_overflow = 0;

        double ratio_slack_pred = 0;
        //double ratio_slack_pred_cpu = 0, ratio_slack_pred_gpu = 0;
        double ratio_split_freq = 0;
        double seconds_until_interrupt = 0;
        double diff_total_cpu = 0, diff_total_gpu = 0, diff_total_slack = 0;
        double gpu_time_pred = 0, cpu_time_pred = 0;
        double gpu_time_this_iter = 0, cpu_time_this_iter = 0;
        int gpu_time_iter0_flag = 0, cpu_time_iter0_flag = 0;
        double gpu_time_iter0, cpu_time_iter0;
        
        static double gpu_time_iter0_highest_freq = 0.401762, gpu_time_iter0_lowest_freq = 1.753552;
        static double cpu_time_iter0_highest_freq = 1.043;
        
        double gpu_time_this_iter_lowest_freq = gpu_time_iter0_lowest_freq;
        int cpu_switched_flag1 = 0;

        #define TIME_MEASUREMENT 0
        #define TIME_DIFF_CPU_FREQ 0
		#define TIME_DIFF_GPU_FREQ 0
		#define SIMPLEST_TEST 0
        #define ALGORITHMIC_SLACK_PREDICTION 0

		#define RACE_TO_HALT 0
		#define CPU_SLACK_RECLAMATION 0
        #define GPU_SLACK_RECLAMATION 0//When testing, set GPU and ALGORITHMIC_SLACK_PREDICTION both to 1.
 
//        if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//        {
//            cudaEventCreate(&start_main_loop);
//            cudaEventCreate(&stop_main_loop);
//            cudaEventRecord(start_main_loop, 0);
//        }

        if(TIME_DIFF_CPU_FREQ) SetCPUFreq(1200000);
        if(TIME_DIFF_GPU_FREQ) SetGPUFreq(324, 324);

//		#include <nvml.h>
//
//        nvmlDevice_t device;
//        nvmlInit();
//        nvmlDeviceGetHandleByIndex(0, &device);
        cudaProfilerStart();
        for( j = 0; j < s; j++ )
        {
            // download j-th panel
            cols = maxm - j*nb;
            
            /* START OF PARALLEL */
            if (j > 0)
            {

                magmablas_dtranspose( nb, cols, dAT(j,j), ldda, dA, cols );
                // make sure that gpu queue is empty
                magma_device_sync();

                magma_dgetmatrix_async( m-j*nb, nb, dA, cols, work, lda,
                                                                     stream[0]);
                
                
				
				
                //[NOT TRUE]dtrsm is on the critical path, so cannot be slowed down.
				magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
							 n - (j+1)*nb, nb,
							 c_one, dAT(j-1,j-1), ldda,
									dAT(j-1,j+1), ldda );
//											 
//				if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//				{
//					cudaEventCreate(&start_gpu);
//					cudaEventCreate(&stop_gpu);
//					cudaEventRecord(start_gpu, 0);
//				}
				//[NOT TRUE]Slow down trailing matrix updating only.
				magma_dgemm( MagmaNoTrans, MagmaNoTrans,
							 n-(j+1)*nb, m-j*nb, nb,
							 c_neg_one, dAT(j-1,j+1), ldda,
										dAT(j,  j-1), ldda,
							 c_one,     dAT(j,  j+1), ldda );
                
//				
//				if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//				{
//					cudaEventRecord(stop_gpu, 0);
//					cudaEventSynchronize(stop_gpu);
//					cudaEventElapsedTime(&gpu_time_cuda_temp, start_gpu, stop_gpu);
//					cudaEventDestroy(start_gpu);
//					cudaEventDestroy(stop_gpu);
//                    total_gpu_time_cuda += gpu_time_cuda_temp;
//                    if(ALGORITHMIC_SLACK_PREDICTION)
//                    {
//                        if(!gpu_time_iter0_flag)
//                        {
//                            gpu_time_iter0 = gpu_time_cuda_temp/1000;
//                            gpu_time_pred = gpu_time_iter0;
//                            gpu_time_iter0_flag = 1;
//                        }
//						gpu_time_this_iter = gpu_time_cuda_temp/1000;
//						printf("iter %d: gpu_time_this_iter = %f\n", j, gpu_time_this_iter);
//                        if(j>1) 
//                        {
//                        	diff_total_gpu += (gpu_time_pred - gpu_time_this_iter)/gpu_time_this_iter;
//                        	double diff_gpu = (gpu_time_pred - gpu_time_this_iter)/gpu_time_this_iter;
//                        	printf("iter %d: diff_gpu = %f\n", j, diff_gpu);
//                        }
//                        ////if(!GPU_SLACK_RECLAMATION)gpu_time_pred = gpu_time_this_iter;//Prediction without this line is worse.
//                    }
	

				
                if (j == 1) {
                	cpu_time_pred = cpu_time_iter0_highest_freq;
					gpu_time_pred = gpu_time_iter0_highest_freq;
                } else {
                	ratio_slack_pred = 1.0 - (double)nb/(m-j*nb);
					cpu_time_pred = cpu_time_pred * ratio_slack_pred;
					gpu_time_pred = gpu_time_pred * ratio_slack_pred * ratio_slack_pred;
					 
//					printf("iter %d: cpu_time_pred = %f\n", j, cpu_time_pred);
//					printf("iter %d: gpu_time_pred = %f\n", j, gpu_time_pred);
                }
                
//                
//                if(j > 1 && j < 10)
//				{
//					ratio_split_freq = (cpu_time_pred - gpu_time_pred) / (gpu_time_pred * ((gpu_time_iter0_lowest_freq / gpu_time_iter0_highest_freq) - 1));
//					//printf("iter %d: ratio_split_freq = %f\n", j, ratio_split_freq);
//					gpu_time_this_iter_lowest_freq = gpu_time_this_iter_lowest_freq * ratio_slack_pred * ratio_slack_pred;
//					//printf("iter %d: gpu_time_this_iter_lowest_freq = %f\n", j, gpu_time_this_iter_lowest_freq);
//					//gpu_time_this_iter_lowest_freq = gpu_time_this_iter_lowest_freq * ratio_slack_pred_gpu;
//					seconds_until_interrupt = gpu_time_this_iter_lowest_freq * ratio_split_freq;
//					//printf("iter %d: seconds_until_interrupt = %f\n", j, seconds_until_interrupt);
//					//double DVFS_overhead_adjustment = 0.6;//0.0014
//					//if(ratio_split_freq < 1) seconds_until_interrupt *= DVFS_overhead_adjustment;//-=
//					if(j > 1)//if(seconds_until_interrupt > 0.0029)//cpu_time_pred - gpu_time_pred > 0.001
//					{
//						initialize_handler();
//						SetGPUFreq(324, 324);
//						if(ratio_split_freq < 1) set_alarm(seconds_until_interrupt);
//						else set_alarm(cpu_time_pred);
//						//SetGPUFreq(2600, 614);
//					}
//				}
//

                // do the cpu part
                rows = m - j*nb;
                magma_queue_sync( stream[0] );
                
//                if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//                {
//                    //cpu_time_cuda_temp = magma_wtime();
//                    cudaEventCreate(&start_cpu);
//                    cudaEventCreate(&stop_cpu);
//                    cudaEventRecord(start_cpu, 0);
//                }
                        
                
        		//PAPI timing start
//        		if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
//        			cout << "PAPI ERROR" << endl;
//        			return -1;
//        		} 
//                
                lapackf77_dgetrf( &rows, &nb, work, &lda, ipiv+j*nb, &iinfo);
                
        		//PAPI timing start
//        		if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
//        			cout << "PAPI ERROR" << endl;
//        			return -1;
//        		} 
//        		
//        		//cout<<"cpu time:"<<real_time<<endl;
//        		cpu_time_this_iter = real_time;
//				PAPI_shutdown();
//				
//        		real_time = 0.0;
//				proc_time = 0.0;
//				flpins = 0.0;
//				mflops = 0.0;
//                
//                if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//                {
//                    cudaEventRecord(stop_cpu, 0);
//                    cudaEventSynchronize(stop_cpu);
//                    cudaEventElapsedTime(&cpu_time_cuda_temp, start_cpu, stop_cpu);
//                    cudaEventDestroy(start_cpu);
//                    cudaEventDestroy(stop_cpu);
//                    total_cpu_time_cuda += cpu_time_cuda_temp;
//                    if(ALGORITHMIC_SLACK_PREDICTION)
//                    {
//                        if(!cpu_time_iter0_flag)
//                        {
//                            cpu_time_iter0 = cpu_time_cuda_temp/1000;
//                            cpu_time_pred = cpu_time_iter0;
//                            cpu_time_iter0_flag = 1;
//                        }
//                       cpu_time_this_iter = cpu_time_cuda_temp/1000;
//                        printf("iter %d: cpu = %f\n", j, cpu_time_this_iter);
//                        printf("iter %d: gpu = %f\n", j, gpu_time_this_iter);
//                        printf("iter %d: slack = %f\n\n", j, cpu_time_this_iter-gpu_time_this_iter);
//                        if(j>1)
//                        {
//                        	diff_total_cpu += (cpu_time_pred - cpu_time_this_iter)/cpu_time_this_iter;
//                        	diff_total_slack += ((cpu_time_pred - gpu_time_pred) - (cpu_time_this_iter - gpu_time_this_iter))/(cpu_time_this_iter - gpu_time_this_iter);
//                        	
//                        	double diff_cpu = (cpu_time_pred - cpu_time_this_iter)/cpu_time_this_iter;
//                        	double diff_slack = ((cpu_time_pred - gpu_time_pred) - (cpu_time_this_iter - gpu_time_this_iter))/(cpu_time_this_iter - gpu_time_this_iter);
//                        	
//                        	printf("iter %d: diff_cpu = %f\n", j, diff_cpu);
//                        	printf("iter %d: diff_slack = %f\n", j, diff_slack);
//                        	
//                        }
    
//                        cpu_time_pred = cpu_time_this_iter;//Prediction without this line is worse.
//                    }
//                }

                if(CPU_SLACK_RECLAMATION)
                {
                    if(j > 50 && cpu_switched_flag1)
                    {
                        system("echo 2500000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
                        cpu_switched_flag1 = 0;
                    }
                }

                if(GPU_SLACK_RECLAMATION)
                    if(!cpu_time_iter0_flag)
                    {
                        cpu_time_pred = cpu_time_iter0_highest_freq;
                        cpu_time_iter0_flag = 1;
                    }

                if(RACE_TO_HALT)
                {
                    SetGPUFreq(2600, 705);
                    //SetCPUFreq(1200000);
                }
            }
            /* END OF PARALLEL */
            
            
            if (*info == 0 && iinfo > 0)
                *info = iinfo + j*nb;

//            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//            {
//                cudaEventCreate(&start_upload_copy);
//                cudaEventCreate(&stop_upload_copy);
//                cudaEventRecord(start_upload_copy, 0);
//            }

            // upload j-th panel
            magma_dsetmatrix_async( m-j*nb, nb, work, lda, dA, cols,
                                    stream[0]);

            for( i=j*nb; i < j*nb + nb; ++i ) {
                ipiv[i] += j*nb;
            }
            magmablas_dlaswp( n, dAT, ldda, j*nb + 1, j*nb + nb, ipiv, 1 );

            magma_queue_sync( stream[0] );
          
            magmablas_dtranspose( cols, nb, dA, cols, dAT(j,j), ldda );

//            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//            {
//                cudaEventRecord(stop_upload_copy, 0);
//                cudaEventSynchronize(stop_upload_copy);
//                cudaEventElapsedTime(&upload_copy_time_cuda_temp, start_upload_copy, stop_upload_copy);
//                cudaEventDestroy(start_upload_copy);
//                cudaEventDestroy(stop_upload_copy);
//                total_copy_time_cuda += upload_copy_time_cuda_temp;
//            }

            // do the small non-parallel computations (next panel update)
            if (s > (j+1)) {
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dAT(j, j  ), ldda,
                                    dAT(j, j+1), ldda);
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             nb, m-(j+1)*nb, nb,
                             c_neg_one, dAT(j,   j+1), ldda,
                                        dAT(j+1, j  ), ldda,
                             c_one,     dAT(j+1, j+1), ldda );
            }
            else {
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-s*nb, nb,
                             c_one, dAT(j, j  ), ldda,
                                    dAT(j, j+1), ldda);
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+1)*nb, m-(j+1)*nb, nb,
                             c_neg_one, dAT(j,   j+1), ldda,
                                        dAT(j+1, j  ), ldda,
                             c_one,     dAT(j+1, j+1), ldda );
            }

//            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//            {
//            	//per_iter_time_cuda_temp = magma_wtime() - per_iter_time_cuda_temp;
//                cudaEventRecord(stop_per_iter, 0);
//                cudaEventSynchronize(stop_per_iter);
//                cudaEventElapsedTime(&per_iter_time_cuda_temp, start_per_iter, stop_per_iter);
//                cudaEventDestroy(start_per_iter);
//                cudaEventDestroy(stop_per_iter);
//            }

//            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//            {
//            	if(cpu_time_cuda_temp/1000 - gpu_time_cuda_temp/1000 < 0)
//                    total_slack_overflow += cpu_time_cuda_temp/1000 - gpu_time_cuda_temp/1000;
//                printf("iter %d: slack_cuda = %.6f\n", j, cpu_time_cuda_temp/1000 - gpu_time_cuda_temp/1000);
//                printf("iter %d: cpu_time_cuda = %.6f\n", j, cpu_time_cuda_temp/1000);
//                printf("iter %d: gpu_time_cuda = %.6f\n", j, gpu_time_cuda_temp/1000);
//                printf("iter %d: download_copy_time_cuda = %.6f\n", j, download_copy_time_cuda_temp/1000);
//                printf("iter %d: upload_copy_time_cuda = %.6f\n", j, upload_copy_time_cuda_temp/1000);
//                printf("iter %d: per_iter_time_cuda = %.6f\n\n", j, per_iter_time_cuda_temp/1000);
//            }
        }
        cudaProfilerStop();
        
//        if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//        {
//            cudaEventRecord(stop_main_loop, 0);
//            cudaEventSynchronize(stop_main_loop);
//            cudaEventElapsedTime(&main_loop_time_cuda, start_main_loop, stop_main_loop);
//            cudaEventDestroy(start_main_loop);
//            cudaEventDestroy(stop_main_loop);
//        }
//
//        if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
//        {
//            printf("total_slack_overflow = %.6f\n", total_slack_overflow);
//            printf("total_cpu_time_cuda = %.6f\n", total_cpu_time_cuda/1000);
//            printf("total_gpu_time_cuda = %.6f\n", total_gpu_time_cuda/1000);
//            printf("total_copy_time_cuda = %.6f\n", total_copy_time_cuda/1000);
//            printf("main_loop_time_cuda = %.6f\n", main_loop_time_cuda/1000);
//            printf("Normalized difference of predicted CPU runtime per iteration is: %.6f\%\n", 100*diff_total_cpu/(m/nb-1));
//            printf("Normalized difference of predicted GPU runtime per iteration is: %.6f\%\n", 100*diff_total_gpu/(m/nb-1));
//            printf("Normalized difference of predicted slack per iteration is: %.6f\%\n", 100*diff_total_slack/(m/nb-1));
//        }

        magma_int_t nb0 = min(m - s*nb, n - s*nb);
        if ( nb0 > 0 ) {
            rows = m - s*nb;
            cols = maxm - s*nb;
    
            magmablas_dtranspose( nb0, rows, dAT(s,s), ldda, dA, cols );
            magma_dgetmatrix( rows, nb0, dA, cols, work, lda );
    
            // make sure that gpu queue is empty
            magma_device_sync();
    
            // do the cpu part
            lapackf77_dgetrf( &rows, &nb0, work, &lda, ipiv+s*nb, &iinfo);
            if (*info == 0 && iinfo > 0)
                *info = iinfo + s*nb;
            
            for( i=s*nb; i < s*nb + nb0; ++i ) {
                ipiv[i] += s*nb;
            }
            magmablas_dlaswp( n, dAT, ldda, s*nb + 1, s*nb + nb0, ipiv, 1 );

            // upload j-th panel
            magma_dsetmatrix( rows, nb0, work, lda, dA, cols );
            magmablas_dtranspose( rows, nb0, dA, cols, dAT(s,s), ldda );
    
            magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-s*nb-nb0, nb0,
                         c_one, dAT(s,s),     ldda,
                                dAT(s,s)+nb0, ldda);
        }
       
        // undo transpose
        if (maxdim*maxdim < 2*maxm*maxn) {
            magmablas_dtranspose_inplace( ldda, dAT, ldda );
            magma_dgetmatrix( m, n, da, ldda, A, lda );
        }
        else {
            magmablas_dtranspose( n, m, dAT, ldda, da, maxm );
            magma_dgetmatrix( m, n, da, maxm, A, lda );
            magma_free( dAT );
        }

        magma_free( dA );
 
        magma_queue_destroy( stream[0] );
        if (orig_stream == NULL) {
            magma_queue_destroy( stream[1] );
        }
        magmablasSetKernelStream( orig_stream );
    }
    
    return *info;
} /* magma_dgetrf */

#undef dAT
