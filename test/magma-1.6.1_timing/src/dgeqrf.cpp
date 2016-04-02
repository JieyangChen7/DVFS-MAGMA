/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Stan Tomov
       @generated from zgeqrf.cpp normal z -> d, Fri Jan 30 19:00:16 2015

*/
#include "common_magma.h"
#include "../testing/testing_util.cpp"

/**
    Purpose
    -------
    DGEQRF computes a QR factorization of a DOUBLE_PRECISION M-by-N matrix A:
    A = Q * R. This version does not require work space on the GPU
    passed as input. GPU memory is allocated in the routine.

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
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    tau     DOUBLE_PRECISION array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    work    (workspace) DOUBLE_PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.
    \n
            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= max( N*NB, 2*NB*NB ),
            where NB can be obtained through magma_get_dgeqrf_nb(M).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_dgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dgeqrf(
    magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda, double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info )
{
    #define  A(i,j) ( A + (i) + (j)*lda )
    #define dA(i,j) (dA + (i) + (j)*ldda)

    double *dA, *dwork, *dT;
    double c_one = MAGMA_D_ONE;

    magma_int_t i, k, lddwork, old_i, old_ib;
    magma_int_t ib, ldda;

    /* Function Body */
    *info = 0;
    magma_int_t nb = 120;//optimal//magma_get_dgeqrf_nb(min(m, n));

    // need 2*nb*nb to store T and upper triangle of V simultaneously
    magma_int_t lwkopt = max(n*nb, 2*nb*nb);
    work[0] = MAGMA_D_MAKE( (double)lwkopt, 0 );
    int lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    } else if (lwork < max(1, lwkopt) && ! lquery) {
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

    // largest N for larfb is n-nb (trailing matrix lacks 1st panel)
    lddwork = ((n+31)/32)*32 - nb;
    ldda    = ((m+31)/32)*32;

    magma_int_t ngpu = magma_num_gpus();
    if ( ngpu > 1 ) {
        /* call multiple-GPU interface  */
        return magma_dgeqrf4(ngpu, m, n, A, lda, tau, work, lwork, info);
    }

    // allocate space for dA, dwork, and dT
    if (MAGMA_SUCCESS != magma_dmalloc( &dA, n*ldda + nb*lddwork + nb*nb )) {
        /* Switch to the "out-of-core" (out of GPU-memory) version */
        return magma_dgeqrf_ooc(m, n, A, lda, tau, work, lwork, info);
    }

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

    dwork = dA + n*ldda;
    dT    = dA + n*ldda + nb*lddwork;

    if ( (nb > 1) && (nb < k) ) {
/***********
 * GreenLA *
 ***********/

	float cpu_time_cuda_temp, total_cpu_time_cuda = 0;
        float upload_copy_time1_cuda_temp, upload_copy_time2_cuda_temp, download_copy_time1_cuda_temp, download_copy_time2_cuda_temp, total_copy_time_cuda = 0;
        float main_loop_time_cuda = 0;
        float gpu_time1_cuda_temp, gpu_time2_cuda_temp, total_gpu_time_cuda = 0;

        cudaEvent_t start_main_loop, stop_main_loop;
        cudaEvent_t start_upload_copy1, stop_upload_copy1;
        cudaEvent_t start_upload_copy2, stop_upload_copy2;
        cudaEvent_t start_download_copy1, stop_download_copy1;
	cudaEvent_t start_download_copy2, stop_download_copy2;
        cudaEvent_t start_cpu, stop_cpu;
        cudaEvent_t start_gpu1, stop_gpu1;
        cudaEvent_t start_gpu2, stop_gpu2;

	magma_int_t iter = 0;
        int DVFS_flag = 0;
	double total_slack_overflow = 0;

        double ratio_slack_pred = 0;
        double ratio_split_freq = 0;
        double seconds_until_interrupt = 0;
        double diff_total_cpu = 0, diff_total_gpu1 = 0, diff_total_gpu2 = 0, diff_total_slack = 0;
        double gpu_time1_pred = 0, gpu_time2_pred = 0, cpu_time_pred = 0;
        double gpu_time1_this_iter = 0, gpu_time2_this_iter = 0, cpu_time_this_iter = 0;
        int gpu_time1_iter0_flag = 0, gpu_time2_iter0_flag = 0, cpu_time_iter0_flag = 0;
        double gpu_time1_iter0, gpu_time2_iter0, cpu_time_iter0;
        static double gpu_time1_iter0_highest_freq = 0.153406, gpu_time1_iter0_lowest_freq = 0.689468;
	static double gpu_time2_iter0_highest_freq = 0.005942, gpu_time2_iter0_lowest_freq = 0.016684;
        static double cpu_time_iter0_highest_freq = 0.217511;
        double gpu_time1_this_iter_lowest_freq = gpu_time1_iter0_lowest_freq;
	double gpu_time2_this_iter_lowest_freq = gpu_time2_iter0_lowest_freq;

        #define TIME_MEASUREMENT 0
        #define TIME_DIFF_CPU_FREQ 0
	#define TIME_DIFF_GPU_FREQ 0
        #define ALGORITHMIC_SLACK_PREDICTION 0

        #define GPU_SLACK_RECLAMATION 1

	if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
	{
            cudaEventCreate(&start_main_loop);
            cudaEventCreate(&stop_main_loop);
            cudaEventRecord(start_main_loop, 0);
	}

	if(TIME_DIFF_CPU_FREQ) SetCPUFreq(1200000);
	if(TIME_DIFF_GPU_FREQ) SetGPUFreq(324, 324);

        /* Use blocked code initially.
           Asynchronously send the matrix to the GPU except the first panel. */
        magma_dsetmatrix_async( m, n-nb,
                                A(0,nb),  lda,
                                dA(0,nb), ldda, stream[0] );

        old_i = 0;
        old_ib = nb;
        for (i = 0; i < k-nb; i += nb) {
            ib = min(k-i, nb);
            if (i > 0) {
		if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
		{
                    cudaEventCreate(&start_download_copy1);
                    cudaEventCreate(&stop_download_copy1);
                    cudaEventRecord(start_download_copy1, 0);
		}

                /* download i-th panel */
                magma_queue_sync( stream[1] );
                magma_dgetmatrix_async( m-i, ib,
                                        dA(i,i), ldda,
                                        A(i,i),  lda, stream[0] );

		if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
		{
                    cudaEventRecord(stop_download_copy1, 0);
                    cudaEventSynchronize(stop_download_copy1);
                    cudaEventElapsedTime(&download_copy_time1_cuda_temp, start_download_copy1, stop_download_copy1);
                    cudaEventDestroy(start_download_copy1);
                    cudaEventDestroy(stop_download_copy1);
                    total_copy_time_cuda += download_copy_time1_cuda_temp;
		}

		if(ALGORITHMIC_SLACK_PREDICTION || GPU_SLACK_RECLAMATION)
                {
                    ratio_slack_pred = 1.0 - (double)nb/(m-iter*nb);
                    //printf("iter %d: ratio_slack_pred = %f\n", iter, ratio_slack_pred);
                    cpu_time_pred = cpu_time_pred * ratio_slack_pred;
                    gpu_time1_pred = gpu_time1_pred * ratio_slack_pred * ratio_slack_pred;
                    gpu_time2_pred = gpu_time2_pred * ratio_slack_pred * ratio_slack_pred;
                    //printf("iter %d: cpu_time_pred = %f\n", iter, cpu_time_pred);
                    //printf("iter %d: gpu_time1_pred = %f\n", iter, gpu_time1_pred);
                    //printf("iter %d: gpu_time2_pred = %f\n", iter, gpu_time2_pred);
                    //printf("iter %d: slack_pred = %f\n", iter, cpu_time_pred - (gpu_time1_pred+gpu_time2_pred));
                }

                if(i > nb && GPU_SLACK_RECLAMATION)//iter > 1
                {
                    ratio_split_freq = (cpu_time_pred - (gpu_time1_pred+gpu_time2_pred)) / ((gpu_time1_pred+gpu_time2_pred) * (((gpu_time1_iter0_lowest_freq+gpu_time2_iter0_lowest_freq) / (gpu_time1_iter0_highest_freq+gpu_time2_iter0_highest_freq)) - 1));
                    ////ratio_split_freq = (cpu_time_pred - gpu_time1_pred) / (gpu_time1_pred * ((gpu_time1_iter0_lowest_freq / gpu_time1_iter0_highest_freq) - 1));
                    //printf("iter %d: ratio_split_freq = %f\n", iter, ratio_split_freq);
                    gpu_time1_this_iter_lowest_freq = gpu_time1_this_iter_lowest_freq * ratio_slack_pred * ratio_slack_pred;
                    gpu_time2_this_iter_lowest_freq = gpu_time2_this_iter_lowest_freq * ratio_slack_pred * ratio_slack_pred;
                    seconds_until_interrupt = (gpu_time1_this_iter_lowest_freq+gpu_time2_this_iter_lowest_freq) * ratio_split_freq;
                    ////seconds_until_interrupt = gpu_time1_this_iter_lowest_freq * ratio_split_freq;
                    //printf("iter %d: seconds_until_interrupt = %f\n", iter, seconds_until_interrupt);
                    //////double DVFS_overhead_adjustment = 0.9;//0.029;
                    //////if(ratio_split_freq < 1) seconds_until_interrupt *= DVFS_overhead_adjustment;//-=
                    initialize_handler();
                    SetGPUFreq(324, 324);
                    if(ratio_split_freq < 1) set_alarm(seconds_until_interrupt);
                    else set_alarm(cpu_time_pred);
                    //SetGPUFreq(2600, 705);//SetGPUFreq(2600, 614);
                }

		if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
		{
                    cudaEventCreate(&start_gpu1);
                    cudaEventCreate(&stop_gpu1);
                    cudaEventRecord(start_gpu1, 0);
		}

                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, n-old_i-2*old_ib, old_ib,
                                  dA(old_i, old_i),          ldda, dT,    nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork, lddwork);

		if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
		{
                    cudaEventRecord(stop_gpu1, 0);
                    cudaEventSynchronize(stop_gpu1);
                    cudaEventElapsedTime(&gpu_time1_cuda_temp, start_gpu1, stop_gpu1);
                    cudaEventDestroy(start_gpu1);
                    cudaEventDestroy(stop_gpu1);
                    total_gpu_time_cuda += gpu_time1_cuda_temp;
                    if(ALGORITHMIC_SLACK_PREDICTION)
                    {
                        if(!gpu_time1_iter0_flag)
                        {
                            gpu_time1_iter0 = gpu_time1_cuda_temp/1000;
                            gpu_time1_pred = gpu_time1_iter0;
                            gpu_time1_iter0_flag = 1;
                        }
                        gpu_time1_this_iter = gpu_time1_cuda_temp/1000;
                        diff_total_gpu1 += (gpu_time1_pred - gpu_time1_this_iter)/gpu_time1_this_iter;
                        gpu_time1_pred = gpu_time1_this_iter;//Prediction without this line is worse.
                    }
		}

		if(GPU_SLACK_RECLAMATION)
                    if(!gpu_time1_iter0_flag)
                    {
                        gpu_time1_pred = gpu_time1_iter0_highest_freq;
                        gpu_time1_iter0_flag = 1;
                    }

		if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
                {
                    cudaEventCreate(&start_download_copy2);
                    cudaEventCreate(&stop_download_copy2);
                    cudaEventRecord(start_download_copy2, 0);
                }

                magma_dgetmatrix_async( i, ib,
                                        dA(0,i), ldda,
                                        A(0,i),  lda, stream[1] );
                magma_queue_sync( stream[0] );

		if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
                {
                    cudaEventRecord(stop_download_copy2, 0);
                    cudaEventSynchronize(stop_download_copy2);
                    cudaEventElapsedTime(&download_copy_time2_cuda_temp, start_download_copy2, stop_download_copy2);
                    cudaEventDestroy(start_download_copy2);
                    cudaEventDestroy(stop_download_copy2);
                    total_copy_time_cuda += download_copy_time2_cuda_temp;
                }
            }

            magma_int_t rows = m-i;

            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
		cudaEventCreate(&start_cpu);
                cudaEventCreate(&stop_cpu);
                cudaEventRecord(start_cpu, 0);
            }

            lapackf77_dgeqrf(&rows, &ib, A(i,i), &lda, tau+i, work, &lwork, info);
            
            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib, A(i,i), &lda, tau+i, work, &ib);

            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
		cudaEventRecord(stop_cpu, 0);
                cudaEventSynchronize(stop_cpu);
                cudaEventElapsedTime(&cpu_time_cuda_temp, start_cpu, stop_cpu);
                cudaEventDestroy(start_cpu);
                cudaEventDestroy(stop_cpu);
                total_cpu_time_cuda += cpu_time_cuda_temp;
		if(ALGORITHMIC_SLACK_PREDICTION)
                {
                    if(!cpu_time_iter0_flag)
                    {
                        cpu_time_iter0 = cpu_time_cuda_temp/1000;
                        cpu_time_pred = cpu_time_iter0;
                        cpu_time_iter0_flag = 1;
                    }
                    cpu_time_this_iter = cpu_time_cuda_temp/1000;
                    diff_total_cpu += (cpu_time_pred - cpu_time_this_iter)/cpu_time_this_iter;
                    if(iter>1) diff_total_slack += ((cpu_time_pred - (gpu_time1_pred+gpu_time2_pred)) - (cpu_time_this_iter - (gpu_time1_this_iter+gpu_time2_this_iter)))/(cpu_time_this_iter - (gpu_time1_this_iter+gpu_time2_this_iter));//(slack_pred - slack_measured) / slack_measured
                    cpu_time_pred = cpu_time_this_iter;//Prediction without this line is worse.
                    }
            }

            if(GPU_SLACK_RECLAMATION)
                if(!cpu_time_iter0_flag)
                {
                    cpu_time_pred = cpu_time_iter0_highest_freq;
                    cpu_time_iter0_flag = 1;
                }

            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
		cudaEventCreate(&start_upload_copy1);
                cudaEventCreate(&stop_upload_copy1);
                cudaEventRecord(start_upload_copy1, 0);
            }

            dpanel_to_q(MagmaUpper, ib, A(i,i), lda, work+ib*ib);

            /* download the i-th V matrix */
            magma_dsetmatrix_async( rows, ib, A(i,i), lda, dA(i,i), ldda, stream[0] );

            /* download the T matrix */
            magma_queue_sync( stream[1] );

            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
                cudaEventRecord(stop_upload_copy1, 0);
                cudaEventSynchronize(stop_upload_copy1);
                cudaEventElapsedTime(&upload_copy_time1_cuda_temp, start_upload_copy1, stop_upload_copy1);
                cudaEventDestroy(start_upload_copy1);
                cudaEventDestroy(stop_upload_copy1);
                total_copy_time_cuda += upload_copy_time1_cuda_temp;
            }

            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
                cudaEventCreate(&start_upload_copy2);
                cudaEventCreate(&stop_upload_copy2);
                cudaEventRecord(start_upload_copy2, 0);
            }

            magma_dsetmatrix_async( ib, ib, work, ib, dT, nb, stream[0] );
            magma_queue_sync( stream[0] );

            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
		cudaEventRecord(stop_upload_copy2, 0);
                cudaEventSynchronize(stop_upload_copy2);
                cudaEventElapsedTime(&upload_copy_time2_cuda_temp, start_upload_copy2, stop_upload_copy2);
                cudaEventDestroy(start_upload_copy2);
                cudaEventDestroy(stop_upload_copy2);
                total_copy_time_cuda += upload_copy_time2_cuda_temp;
            }

            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
		cudaEventCreate(&start_gpu2);
                cudaEventCreate(&stop_gpu2);
                cudaEventRecord(start_gpu2, 0);
            }

            if (i + ib < n) {
                if (i+ib < k-nb) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left (look-ahead) */
                    magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(i, i   ), ldda, dT,    nb,
                                      dA(i, i+ib), ldda, dwork, lddwork);
                    dq_to_panel(MagmaUpper, ib, A(i,i), lda, work+ib*ib);
                }
                else {
                    /* After last panel, update whole trailing matrix. */
                    /* Apply H' to A(i:m,i+ib:n) from the left */
                    magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, n-i-ib, ib,
                                      dA(i, i   ), ldda, dT,    nb,
                                      dA(i, i+ib), ldda, dwork, lddwork);
                    dq_to_panel(MagmaUpper, ib, A(i,i), lda, work+ib*ib);
                }

                old_i  = i;
                old_ib = ib;
            }

            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
                cudaEventRecord(stop_gpu2, 0);
                cudaEventSynchronize(stop_gpu2);
                cudaEventElapsedTime(&gpu_time2_cuda_temp, start_gpu2, stop_gpu2);
                cudaEventDestroy(start_gpu2);
                cudaEventDestroy(stop_gpu2);
                total_gpu_time_cuda += gpu_time2_cuda_temp;
                if(ALGORITHMIC_SLACK_PREDICTION)
                {   
                    if(!gpu_time2_iter0_flag)
                    {   
                        gpu_time2_iter0 = gpu_time2_cuda_temp/1000;
                        gpu_time2_pred = gpu_time2_iter0;
                        gpu_time2_iter0_flag = 1;
                    }
                    gpu_time2_this_iter = gpu_time2_cuda_temp/1000;
                    diff_total_gpu2 += (gpu_time2_pred - gpu_time2_this_iter)/gpu_time2_this_iter;
                    gpu_time2_pred = gpu_time2_this_iter;//Prediction without this line is worse.
                }
            }

            if(GPU_SLACK_RECLAMATION)
                if(!gpu_time2_iter0_flag)
                {
                    gpu_time2_pred = gpu_time2_iter0_highest_freq;
                    gpu_time2_iter0_flag = 1;
                }

            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
                if(cpu_time_cuda_temp/1000 - (gpu_time1_cuda_temp+gpu_time2_cuda_temp)/1000 < 0)
                    total_slack_overflow += cpu_time_cuda_temp/1000 - (gpu_time1_cuda_temp+gpu_time2_cuda_temp)/1000;
                printf("iter %d: slack_cuda = %.6f\n", iter, cpu_time_cuda_temp/1000 - (gpu_time1_cuda_temp+gpu_time2_cuda_temp)/1000);
                printf("iter %d: cpu_time_cuda = %.6f\n", iter, cpu_time_cuda_temp/1000);
                printf("iter %d: gpu_time1_cuda = %.6f\n", iter, gpu_time1_cuda_temp/1000);
                printf("iter %d: gpu_time2_cuda = %.6f\n", iter, gpu_time2_cuda_temp/1000);
                printf("iter %d: upload_copy_time1_cuda = %.6f\n", iter, upload_copy_time1_cuda_temp/1000);
                printf("iter %d: download_copy_time_cuda = %.6f\n", iter, download_copy_time1_cuda_temp/1000);
                printf("iter %d: upload_copy_time2_cuda = %.6f\n", iter, upload_copy_time2_cuda_temp/1000);
                printf("iter %d: download_copy_time_cuda = %.6f\n\n", iter++, download_copy_time2_cuda_temp/1000);
            }
        }

	if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
	{
            cudaEventRecord(stop_main_loop, 0);
            cudaEventSynchronize(stop_main_loop);
            cudaEventElapsedTime(&main_loop_time_cuda, start_main_loop, stop_main_loop);
            cudaEventDestroy(start_main_loop);
            cudaEventDestroy(stop_main_loop);
	}

	if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
	{
            printf("total_slack_overflow = %.6f\n", total_slack_overflow);
            printf("total_cpu_time_cuda = %.6f\n", total_cpu_time_cuda/1000);
            printf("total_gpu_time_cuda = %.6f\n", total_gpu_time_cuda/1000);
            printf("total_copy_time_cuda = %.6f\n", total_copy_time_cuda/1000);
            printf("main_loop_time_cuda = %.6f\n", main_loop_time_cuda/1000);
            printf("Normalized difference of predicted CPU runtime per iteration is: %.6f\%\n", 100*diff_total_cpu/(m/nb));
            printf("Normalized difference of predicted GPU runtime1 per iteration is: %.6f\%\n", 100*diff_total_gpu1/(m/nb));
            printf("Normalized difference of predicted GPU runtime2 per iteration is: %.6f\%\n", 100*diff_total_gpu2/(m/nb));
            printf("Normalized difference of predicted slack per iteration is: %.6f\%\n", 100*diff_total_slack/(m/nb-1));
	}
    } else {
        i = 0;
    }
    
    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib = n-i;
        if (i != 0) {
            magma_dgetmatrix_async( m, ib, dA(0,i), ldda, A(0,i), lda, stream[1] );
            magma_queue_sync( stream[1] );
        }
        magma_int_t rows = m-i;
        lapackf77_dgeqrf(&rows, &ib, A(i,i), &lda, tau+i, work, &lwork, info);
    }

    magma_queue_destroy( stream[0] );
    if (orig_stream == NULL) {
        magma_queue_destroy( stream[1] );
    }
    magmablasSetKernelStream( orig_stream );

    magma_free( dA );
    
    return *info;
} /* magma_dgeqrf */
