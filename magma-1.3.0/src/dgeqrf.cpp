/*
    -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2012

       @generated d Wed Nov 14 22:53:15 2012

*/
#include "common_magma.h"
#include "../testing/testing_util.cpp"

extern "C" magma_int_t
magma_dgeqrf(magma_int_t m, magma_int_t n, 
             double *a,    magma_int_t lda, double *tau, 
             double *work, magma_int_t lwork,
             magma_int_t *info )
{
/*  -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2012

    Purpose
    =======
    DGEQRF computes a QR factorization of a DOUBLE_PRECISION M-by-N matrix A:
    A = Q * R. This version does not require work space on the GPU
    passed as input. GPU memory is allocated in the routine.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) DOUBLE_PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) DOUBLE_PRECISION array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) DOUBLE_PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= N*NB,
            where NB can be obtained through magma_get_dgeqrf_nb(M).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ===============
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).
    =====================================================================    */

    #define  a_ref(a_1,a_2) ( a+(a_2)*(lda) + (a_1))
    #define da_ref(a_1,a_2) (da+(a_2)*ldda  + (a_1))

    double *da, *dwork;
    double c_one = MAGMA_D_ONE;

    magma_int_t i, k, lddwork, old_i, old_ib;
    magma_int_t ib, ldda;

    /* Function Body */
    *info = 0;
    magma_int_t nb = magma_get_dgeqrf_nb(min(m, n));//nb is printed to be 128.

    magma_int_t lwkopt = n * nb;
    work[0] = MAGMA_D_MAKE( (double)lwkopt, 0 );
    int lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    } else if (lwork < max(1,n) && ! lquery) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    k = min(m,n);
    if (k == 0) {
        work[0] = c_one;
        return *info;
    }

    lddwork = ((n+31)/32)*32;
    ldda    = ((m+31)/32)*32;

    magma_int_t num_gpus = magma_num_gpus();
    if( num_gpus > 1 ) {
        /* call multiple-GPU interface  */
        return magma_dgeqrf4(num_gpus, m, n, a, lda, tau, work, lwork, info);
    }

    if (MAGMA_SUCCESS != magma_dmalloc( &da, (n)*ldda + nb*lddwork )) {
        /* Switch to the "out-of-core" (out of GPU-memory) version */
        return magma_dgeqrf_ooc(m, n, a, lda, tau, work, lwork, info);
    }

    cudaStream_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    dwork = da + ldda*(n);

    if ( (nb > 1) && (nb < k) ) {
	float cpu_time_cuda_temp, total_cpu_time_cuda = 0;
	float upload_copy_time1_cuda_temp, upload_copy_time2_cuda_temp, download_copy_time_cuda_temp, total_copy_time_cuda = 0;
	float main_loop_time_cuda = 0;
	float gpu_time1_cuda_temp, gpu_time2_cuda_temp, total_gpu_time_cuda = 0;

	cudaEvent_t start_main_loop, stop_main_loop;
	cudaEvent_t start_upload_copy1, stop_upload_copy1;
	cudaEvent_t start_upload_copy2, stop_upload_copy2;
	cudaEvent_t start_download_copy, stop_download_copy;
	cudaEvent_t start_cpu, stop_cpu;
	cudaEvent_t start_gpu1, stop_gpu1;
	cudaEvent_t start_gpu2, stop_gpu2;

        /* Use blocked code initially */
        magma_dsetmatrix_async( (m), (n-nb),
                                a_ref(0,nb),  lda,
                                da_ref(0,nb), ldda, stream[0] );

        old_i = 0; old_ib = nb;

	magma_int_t iter = 0;
	int DVFS_flag = 0;

	double ratio_slack_pred = 0;
        double ratio_split_freq = 0;
        double seconds_until_interrupt = 0;
        double diff_total_cpu = 0, diff_total_gpu = 0, diff_total_slack = 0;
        double gpu_time_pred = 0, cpu_time_pred = 0;
        double gpu_time_this_iter = 0, cpu_time_this_iter = 0;
        int gpu_time_iter0_flag = 0, cpu_time_iter0_flag = 0;
        double gpu_time_iter0, cpu_time_iter0;
        static double gpu_time_iter0_highest_freq = 0.288660, gpu_time_iter0_lowest_freq = 1.386685;
        static double cpu_time_iter0_highest_freq = 0.343753;
        double gpu_time_this_iter_lowest_freq = gpu_time_iter0_lowest_freq;

	#define TIME_MEASUREMENT 0
	#define TIME_DIFFERENT_FREQ 0
        #define ALGORITHMIC_SLACK_PREDICTION 0
        #define GPU_SLACK_RECLAMATION_ENABLED 1

	if(TIME_MEASUREMENT){
        //main_loop_time = magma_wtime();
        cudaEventCreate(&start_main_loop);
        cudaEventCreate(&stop_main_loop);
        cudaEventRecord(start_main_loop, 0);
        }

	if(TIME_DIFFERENT_FREQ) SetGPUFreq(324, 324);

        for (i = 0; i < k-nb; i += nb) {
            ib = min(k-i, nb);
            if (i>0){
		if(TIME_MEASUREMENT){
		//download_copy_time = magma_wtime();
		cudaEventCreate(&start_download_copy);
		cudaEventCreate(&stop_download_copy);
		cudaEventRecord(start_download_copy, 0);
		}
                magma_dgetmatrix_async( (m-i), ib,
                                        da_ref(i,i), ldda,
                                        a_ref(i,i),  lda, stream[1] );

                magma_dgetmatrix_async( i, ib,
                                        da_ref(0,i), ldda,
                                        a_ref(0,i),  lda, stream[0] );
		if(TIME_MEASUREMENT){
		//download_copy_time = magma_wtime() - download_copy_time;
		//total_copy_time += download_copy_time;
		cudaEventRecord(stop_download_copy, 0);
		cudaEventSynchronize(stop_download_copy);
		cudaEventElapsedTime(&download_copy_time_cuda_temp, start_download_copy, stop_download_copy);
		cudaEventDestroy(start_download_copy);
		cudaEventDestroy(stop_download_copy);
		total_copy_time_cuda += download_copy_time_cuda_temp;
		}

		if(ALGORITHMIC_SLACK_PREDICTION || GPU_SLACK_RECLAMATION_ENABLED)
                {
                    ratio_slack_pred = 1.0 - (double)nb/(m-iter*nb);
		    printf("iter %d: ratio_slack_pred = %f\n", iter, ratio_slack_pred);
                    cpu_time_pred = cpu_time_pred * ratio_slack_pred;
                    gpu_time_pred = gpu_time_pred * ratio_slack_pred * ratio_slack_pred;
                    printf("iter %d: cpu_time_pred = %f\n", iter, cpu_time_pred);
                    printf("iter %d: gpu_time_pred = %f\n", iter, gpu_time_pred);
                }

		if(GPU_SLACK_RECLAMATION_ENABLED)
                {
                    ratio_split_freq = (cpu_time_pred - gpu_time_pred) / (gpu_time_pred * ((gpu_time_iter0_lowest_freq / gpu_time_iter0_highest_freq) - 1));
                    printf("iter %d: ratio_split_freq = %f\n", iter, ratio_split_freq);
                    gpu_time_this_iter_lowest_freq = gpu_time_this_iter_lowest_freq * ratio_slack_pred * ratio_slack_pred;
                    seconds_until_interrupt = gpu_time_this_iter_lowest_freq * ratio_split_freq;
                    printf("iter %d: seconds_until_interrupt = %f\n", iter++, seconds_until_interrupt);
                    double DVFS_overhead_adjustment = 0.8;//0.029;
                    if(ratio_split_freq < 1) seconds_until_interrupt *= DVFS_overhead_adjustment;
                    initialize_handler();
                    SetGPUFreq(324, 324);
                    if(ratio_split_freq < 1) set_alarm(seconds_until_interrupt);
                    else set_alarm(cpu_time_pred);
                    //SetGPUFreq(2600, 614);
                }

		//if(iter>125)SetGPUFreq(324, 324);
		//else
		//if(!DVFS_flag){
		//SetGPUFreq(2600, 614);
		//DVFS_flag = 1;}
		if(TIME_MEASUREMENT){
                //gpu_time = magma_wtime();
                cudaEventCreate(&start_gpu1);
                cudaEventCreate(&stop_gpu1);
                cudaEventRecord(start_gpu1, 0);
                }
		//if(!DVFS_flag){//if(i%2==0)
		//SetGPUFreq(324, 324);//SetGPUFreq(2600, 614);
		//DVFS_flag = 1;}
                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_dlarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise, 
                                  m-old_i, n-old_i-2*old_ib, old_ib,
                                  da_ref(old_i, old_i),          ldda, dwork,        lddwork,
                                  da_ref(old_i, old_i+2*old_ib), ldda, dwork+old_ib, lddwork);
		//SetGPUFreq(2600, 758);
		if(TIME_MEASUREMENT){
                //gpu_time1 = magma_wtime() - gpu_time1;
                //total_gpu_time += gpu_time1;
                cudaEventRecord(stop_gpu1, 0);
                cudaEventSynchronize(stop_gpu1);
                cudaEventElapsedTime(&gpu_time1_cuda_temp, start_gpu1, stop_gpu1);
                cudaEventDestroy(start_gpu1);
                cudaEventDestroy(stop_gpu1);
                total_gpu_time_cuda += gpu_time1_cuda_temp;
                }
		//SetGPUFreq(2600, 758);

		if(GPU_SLACK_RECLAMATION_ENABLED)
                    if(!gpu_time_iter0_flag)
                    {
                        gpu_time_pred = gpu_time_iter0_highest_freq;
                        gpu_time_iter0_flag = 1;
                    }
            }

            magma_queue_sync( stream[1] );
            magma_int_t rows = m-i;

            if(TIME_MEASUREMENT){
            //cpu_time = magma_wtime();
            cudaEventCreate(&start_cpu);
            cudaEventCreate(&stop_cpu);
            cudaEventRecord(start_cpu, 0);
            }
            lapackf77_dgeqrf(&rows, &ib, a_ref(i,i), &lda, tau+i, work, &lwork, info);
            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr, 
                              &rows, &ib, a_ref(i,i), &lda, tau+i, work, &ib);
            if(TIME_MEASUREMENT){
            //cpu_time = magma_wtime() - cpu_time;
            //total_cpu_time += cpu_time;
            cudaEventRecord(stop_cpu, 0);
            cudaEventSynchronize(stop_cpu);
            cudaEventElapsedTime(&cpu_time_cuda_temp, start_cpu, stop_cpu);
            cudaEventDestroy(start_cpu);
            cudaEventDestroy(stop_cpu);
            total_cpu_time_cuda += cpu_time_cuda_temp;
            }

	    if(GPU_SLACK_RECLAMATION_ENABLED)
                    if(!cpu_time_iter0_flag)
                    {
                        cpu_time_pred = cpu_time_iter0_highest_freq;
                        cpu_time_iter0_flag = 1;
                    }

            if(TIME_MEASUREMENT){
            //upload_copy_time1 = magma_wtime();
            cudaEventCreate(&start_upload_copy1);
            cudaEventCreate(&stop_upload_copy1);
            cudaEventRecord(start_upload_copy1, 0);
            }
            dpanel_to_q(MagmaUpper, ib, a_ref(i,i), lda, work+ib*ib);
            magma_dsetmatrix( rows, ib, a_ref(i,i), lda, da_ref(i,i), ldda );
            dq_to_panel(MagmaUpper, ib, a_ref(i,i), lda, work+ib*ib);
            if(TIME_MEASUREMENT){
            //upload_copy_time1 = magma_wtime() - upload_copy_time1;
            //total_copy_time += upload_copy_time1;
            cudaEventRecord(stop_upload_copy1, 0);
            cudaEventSynchronize(stop_upload_copy1);
            cudaEventElapsedTime(&upload_copy_time1_cuda_temp, start_upload_copy1, stop_upload_copy1);
            cudaEventDestroy(start_upload_copy1);
            cudaEventDestroy(stop_upload_copy1);
            total_copy_time_cuda += upload_copy_time1_cuda_temp;
            }

            if (i + ib < n) {
		if(TIME_MEASUREMENT){
                //upload_copy_time2 = magma_wtime();
                cudaEventCreate(&start_upload_copy2);
                cudaEventCreate(&stop_upload_copy2);
                cudaEventRecord(start_upload_copy2, 0);
                }
                magma_dsetmatrix( ib, ib, work, ib, dwork, lddwork );
		if(TIME_MEASUREMENT){
                //upload_copy_time2 = magma_wtime() - upload_copy_time2;
                //total_copy_time += upload_copy_time2;
                cudaEventRecord(stop_upload_copy2, 0);
                cudaEventSynchronize(stop_upload_copy2);
                cudaEventElapsedTime(&upload_copy_time2_cuda_temp, start_upload_copy2, stop_upload_copy2);
                cudaEventDestroy(start_upload_copy2);
                cudaEventDestroy(stop_upload_copy2);
                total_copy_time_cuda += upload_copy_time2_cuda_temp;
                }

		if(TIME_MEASUREMENT){
                //gpu_time2 = magma_wtime();
                cudaEventCreate(&start_gpu2);
                cudaEventCreate(&stop_gpu2);
                cudaEventRecord(start_gpu2, 0);
                }
                if (i+ib < k-nb)
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
                    magma_dlarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise, 
                                      rows, ib, ib, 
                                      da_ref(i, i   ), ldda, dwork,    lddwork, 
                                      da_ref(i, i+ib), ldda, dwork+ib, lddwork);
                else
                    magma_dlarfb_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise, 
                                      rows, n-i-ib, ib, 
                                      da_ref(i, i   ), ldda, dwork,    lddwork, 
                                      da_ref(i, i+ib), ldda, dwork+ib, lddwork);
		if(TIME_MEASUREMENT){
                //gpu_time2 = magma_wtime() - gpu_time2;
                //total_copy_time += gpu_time2;
                cudaEventRecord(stop_gpu2, 0);
                cudaEventSynchronize(stop_gpu2);
		cudaEventElapsedTime(&gpu_time2_cuda_temp, start_gpu2, stop_gpu2);
                cudaEventDestroy(start_gpu2);
                cudaEventDestroy(stop_gpu2);
                total_gpu_time_cuda += gpu_time2_cuda_temp;
                }

                old_i  = i;
                old_ib = ib;
            }

            if(TIME_MEASUREMENT){
            //printf("iter %d: cpu_time = %.6f\n", iter, cpu_time);
            //printf("iter %d: gpu_time1 = %.6f\n", iter, gpu_time1);
            //printf("iter %d: gpu_time2 = %.6f\n", iter, gpu_time2);
            //printf("iter %d: upload_copy_time1 = %.6f\n", iter, upload_copy_time1);
            //printf("iter %d: download_copy_time = %.6f\n", iter, download_copy_time);
            //printf("iter %d: upload_copy_time2 = %.6f\n", iter, upload_copy_time2);
            printf("iter %d: cpu_time_cuda = %.6f\n", iter, cpu_time_cuda_temp/1000);
            printf("iter %d: gpu_time1_cuda = %.6f\n", iter, gpu_time1_cuda_temp/1000);
            printf("iter %d: gpu_time2_cuda = %.6f\n", iter, gpu_time2_cuda_temp/1000);
            printf("iter %d: upload_copy_time1_cuda = %.6f\n", iter, upload_copy_time1_cuda_temp/1000);
            printf("iter %d: download_copy_time_cuda = %.6f\n", iter, download_copy_time_cuda_temp/1000);
            printf("iter %d: upload_copy_time2_cuda = %.6f\n\n", iter++, upload_copy_time2_cuda_temp/1000);
            }
        }
	if(TIME_MEASUREMENT){
        //main_loop_time = magma_wtime() - main_loop_time;
        cudaEventRecord(stop_main_loop, 0);
        cudaEventSynchronize(stop_main_loop);
        cudaEventElapsedTime(&main_loop_time_cuda, start_main_loop, stop_main_loop);
        cudaEventDestroy(start_main_loop);
        cudaEventDestroy(stop_main_loop);
        }

	if(TIME_MEASUREMENT){
        //printf("total_cpu_time = %.6f\n", total_cpu_time);
        //printf("total_gpu_time = %.6f\n", total_gpu_time);
        //printf("total_copy_time = %.6f\n", total_copy_time);
        //printf("main_loop_time = %.6f\n", main_loop_time);
        printf("total_cpu_time_cuda = %.6f\n", total_cpu_time_cuda/1000);
        printf("total_gpu_time_cuda = %.6f\n", total_gpu_time_cuda/1000);
        printf("total_copy_time_cuda = %.6f\n", total_copy_time_cuda/1000);
        printf("main_loop_time_cuda = %.6f\n", main_loop_time_cuda/1000);
        }
    } else {
        i = 0;
    }
    
    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib = n-i;
        if (i!=0)
            magma_dgetmatrix( m, ib, da_ref(0,i), ldda, a_ref(0,i), lda );
        magma_int_t rows = m-i;
        lapackf77_dgeqrf(&rows, &ib, a_ref(i,i), &lda, tau+i, work, &lwork, info);
    }

    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
    magma_free( da );
    return *info;
} /* magma_dgeqrf */

