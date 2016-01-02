/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @author Stan Tomov
       @generated from zgetrf.cpp normal z -> d, Wed Sep 17 15:08:32 2014
*/
#include "common_magma.h"
#include "../testing/testing_util.cpp"


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
magma_dgetrf(magma_int_t m, magma_int_t n, double *A, magma_int_t lda,
             magma_int_t *ipiv, magma_int_t *info)
{
#define dAT(i,j) (dAT + (i)*nb*ldda + (j)*nb)

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

    //nb = magma_get_dgetrf_nb(m);
    nb = 64;

    if ( (nb <= 1) || (nb >= min(m,n)) ) {
        /* Use CPU code. */
        lapackf77_dgetrf(&m, &n, A, &lda, ipiv, info);
    } else {
        /* Use hybrid blocked code. */
        magma_int_t maxm, maxn, ldda, maxdim;
        magma_int_t i, rows, cols, s = min(m, n)/nb;
        
        maxm = ((m + 31)/32)*32;
        maxn = ((n + 31)/32)*32;
        maxdim = max(maxm, maxn);

        /* set number of GPUs */
        magma_int_t num_gpus = magma_num_gpus();
        if ( num_gpus > 1 ) {
            /* call multi-GPU non-GPU-resident interface  */
            magma_dgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
            return *info;
        }

        /* explicitly checking the memory requirement */
        size_t freeMem, totalMem;
        cudaMemGetInfo( &freeMem, &totalMem );
        freeMem /= sizeof(double);

        int h = 1+(2+num_gpus), num_gpus2 = num_gpus;
        int NB = (magma_int_t)(0.8*freeMem/maxm-h*nb);
        const char* ngr_nb_char = getenv("MAGMA_NGR_NB");
        if ( ngr_nb_char != NULL )
            NB = max( nb, min( NB, atoi(ngr_nb_char) ) );

        if ( num_gpus > ceil((double)NB/nb) ) {
            num_gpus2 = (int)ceil((double)NB/nb);
            h = 1+(2+num_gpus2);
            NB = (magma_int_t)(0.8*freeMem/maxm-h*nb);
        }
        if ( num_gpus2*NB < n ) {
            /* require too much memory, so call non-GPU-resident version */
            magma_dgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
            return *info;
        }

        ldda = maxn;
        work = A;
        if (maxdim*maxdim < 2*maxm*maxn) {
            // if close to square, allocate square matrix and transpose in-place
            if (MAGMA_SUCCESS != magma_dmalloc( &dA, nb*maxm + maxdim*maxdim )) {
                /* alloc failed so call non-GPU-resident version */
                magma_dgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
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
                magma_dgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
                return *info;
            }
            da = dA + nb*maxm;
            
            magma_dsetmatrix( m, n, A, lda, da, maxm );
            
            if (MAGMA_SUCCESS != magma_dmalloc( &dAT, maxm*maxn )) {
                /* alloc failed so call non-GPU-resident version */
                magma_free( dA );
                magma_dgetrf_m(num_gpus, m, n, A, lda, ipiv, info);
                return *info;
            }

            magmablas_dtranspose( m, n, da, maxm, dAT, ldda );
        }
        
        lapackf77_dgetrf( &m, &nb, work, &lda, ipiv, &iinfo);

        /* Define user stream if current stream is NULL */
        magma_queue_t stream[2];
        
        magma_queue_t orig_stream;
        magmablasGetKernelStream( &orig_stream );

        magma_queue_create( &stream[0] );
        if (orig_stream == NULL) {
            magma_queue_create( &stream[1] );
            magmablasSetKernelStream(stream[1]);
        }
        else
            stream[1] = orig_stream;

/**************
 * Definition *
 **************/

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

	double ratio_slack_pred = 0;
	double ratio_split_freq = 0;
	double seconds_until_interrupt = 0;
	double diff_total_cpu = 0, diff_total_gpu = 0, diff_total_slack = 0;
	double gpu_time_pred = 0, cpu_time_pred = 0;
	double gpu_time_this_iter = 0, cpu_time_this_iter = 0;
	int gpu_time_iter0_flag = 0, cpu_time_iter0_flag = 0, iter;
	double gpu_time_iter0, cpu_time_iter0;
	static double gpu_time_iter0_highest_freq = 0.005428, gpu_time_iter0_lowest_freq = 0.033626;
	static double cpu_time_iter0_highest_freq = 0.007614;
	double gpu_time_this_iter_lowest_freq = gpu_time_iter0_lowest_freq;

	#define TIME_MEASUREMENT 0
	#define TIME_DIFFERENT_FREQ 0
	#define ALGORITHMIC_SLACK_PREDICTION 0
	#define GPU_SLACK_RECLAMATION_ENABLED 0

	if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
	{
		cudaEventCreate(&start_main_loop);
        	cudaEventCreate(&stop_main_loop);
        	cudaEventRecord(start_main_loop, 0);
	}

	if(TIME_DIFFERENT_FREQ) SetGPUFreq(324, 324);

        for( i = 0; i < s; i++ )
	{
            if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
            {
                cudaEventCreate(&start_per_iter);
                cudaEventCreate(&stop_per_iter);
                cudaEventRecord(start_per_iter, 0);
            }

            // download i-th panel
            cols = maxm - i*nb;
            
            if (i > 0)
            {
		if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
		{
		    cudaEventCreate(&start_download_copy);
                    cudaEventCreate(&stop_download_copy);
                    cudaEventRecord(start_download_copy, 0);
		}

                // download i-th panel
                magmablas_dtranspose( nb, cols, dAT(i,i), ldda, dA, cols );

                // make sure that gpu queue is empty
                magma_device_sync();

                magma_dgetmatrix_async( m-i*nb, nb, dA, cols, work, lda,
                                        stream[0]);

		if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
		{
		    cudaEventRecord(stop_download_copy, 0);
                    cudaEventSynchronize(stop_download_copy);
                    cudaEventElapsedTime(&download_copy_time_cuda_temp, start_download_copy, stop_download_copy);
                    cudaEventDestroy(start_download_copy);
                    cudaEventDestroy(stop_download_copy);
                    total_copy_time_cuda += download_copy_time_cuda_temp;
		}

		if(TIME_MEASUREMENT)
		{
		    cudaEventCreate(&start_gpu1);
                    cudaEventCreate(&stop_gpu1);
                    cudaEventRecord(start_gpu1, 0);
		}

                //dtrsm is on the critical path, so cannot be slowed down.
                for(iter = 0; iter < 10; iter++)
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n - (i+1)*nb, nb,
                             c_one, dAT(i-1,i-1), ldda,
                                    dAT(i-1,i+1), ldda );

		if(ALGORITHMIC_SLACK_PREDICTION || GPU_SLACK_RECLAMATION_ENABLED)
		{
		    ratio_slack_pred = 1.0 - (double)nb/(m-i*nb);
		    cpu_time_pred = cpu_time_pred * ratio_slack_pred;
		    gpu_time_pred = gpu_time_pred * ratio_slack_pred * ratio_slack_pred;
		    printf("iter %d: cpu_time_pred = %f\n", i, cpu_time_pred);
                    printf("iter %d: gpu_time_pred = %f\n", i, gpu_time_pred);
		}

		if(GPU_SLACK_RECLAMATION_ENABLED)
		{
		    ratio_split_freq = (cpu_time_pred - gpu_time_pred) / (gpu_time_pred * ((gpu_time_iter0_lowest_freq / gpu_time_iter0_highest_freq) - 1));
		    printf("iter %d: ratio_split_freq = %f\n", i, ratio_split_freq);
		    gpu_time_this_iter_lowest_freq = gpu_time_this_iter_lowest_freq * ratio_slack_pred * ratio_slack_pred;
		    seconds_until_interrupt = gpu_time_this_iter_lowest_freq * ratio_split_freq;
		    printf("iter %d: seconds_until_interrupt = %f\n", i, seconds_until_interrupt);
		    ////double DVFS_overhead_adjustment = 0.0014;
		    ////if(ratio_split_freq < 1) seconds_until_interrupt -= DVFS_overhead_adjustment;
		    initialize_handler();
                    SetGPUFreq(324, 324);
                    if(ratio_split_freq < 1) set_alarm(seconds_until_interrupt);
                    else set_alarm(cpu_time_pred);
		    //SetGPUFreq(2600, 614);
		}

		if(ALGORITHMIC_SLACK_PREDICTION)
                {
                    cudaEventCreate(&start_gpu1);
                    cudaEventCreate(&stop_gpu1);
                    cudaEventRecord(start_gpu1, 0);
                }

		//Slow down trailing matrix update only.
		for(iter = 0; iter < 10; iter++)
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(i+1)*nb, m-i*nb, nb,
                             c_neg_one, dAT(i-1,i+1), ldda,
                                        dAT(i,  i-1), ldda,
                             c_one,     dAT(i,  i+1), ldda );

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
			if(!gpu_time_iter0_flag)
                	{
                            gpu_time_iter0 = gpu_time1_cuda_temp/1000;
                            gpu_time_pred = gpu_time_iter0;
                            gpu_time_iter0_flag = 1;
                	}
                        gpu_time_this_iter = gpu_time1_cuda_temp/1000;
                  	diff_total_gpu += (gpu_time_pred - gpu_time_this_iter)/gpu_time_this_iter;
		    }
		}

		if(GPU_SLACK_RECLAMATION_ENABLED)
		    if(!gpu_time_iter0_flag)
		    {
			gpu_time_pred = gpu_time_iter0_highest_freq;
			gpu_time_iter0_flag = 1;
		    }

		if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
		{
		    cudaEventCreate(&start_cpu);
                    cudaEventCreate(&stop_cpu);
                    cudaEventRecord(start_cpu, 0);
		}

                // do the cpu part
                rows = m - i*nb;
                magma_queue_sync( stream[0] );
		for(iter = 0; iter < 10; iter++)
                lapackf77_dgetrf( &rows, &nb, work, &lda, ipiv+i*nb, &iinfo);

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
                	diff_total_slack += ((cpu_time_pred - gpu_time_pred) - (cpu_time_this_iter - gpu_time_this_iter))/(cpu_time_this_iter - gpu_time_this_iter);//(slack_pred - slack_measured) / slack_measured
		    }
		}

		if(GPU_SLACK_RECLAMATION_ENABLED)
		    if(!cpu_time_iter0_flag)
		    {
			cpu_time_pred = cpu_time_iter0_highest_freq;
			cpu_time_iter0_flag = 1;
		    }
            }

            if (*info == 0 && iinfo > 0)
                *info = iinfo + i*nb;

	    if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
	    {
		cudaEventCreate(&start_upload_copy);
                cudaEventCreate(&stop_upload_copy);
                cudaEventRecord(start_upload_copy, 0);
	    }

            // upload i-th panel
            magma_dsetmatrix_async( m-i*nb, nb, work, lda, dA, cols,
                                    stream[0]);

            magmablas_dpermute_long2( ldda, dAT, ldda, ipiv, nb, i*nb );

            magma_queue_sync( stream[0] );
            magmablas_dtranspose( cols, nb, dA, cols, dAT(i,i), ldda );

	    if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
	    {
		cudaEventRecord(stop_upload_copy, 0);
                cudaEventSynchronize(stop_upload_copy);
                cudaEventElapsedTime(&upload_copy_time_cuda_temp, start_upload_copy, stop_upload_copy);
                cudaEventDestroy(start_upload_copy);
                cudaEventDestroy(stop_upload_copy);
                total_copy_time_cuda += upload_copy_time_cuda_temp;
	    }

            // do the small non-parallel computations
            if (s > (i+1)) {
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dAT(i, i  ), ldda,
                                    dAT(i, i+1), ldda);
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             nb, m-(i+1)*nb, nb,
                             c_neg_one, dAT(i,   i+1), ldda,
                                        dAT(i+1, i  ), ldda,
                             c_one,     dAT(i+1, i+1), ldda );
            }
            else {
                magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-s*nb, nb,
                             c_one, dAT(i, i  ), ldda,
                                    dAT(i, i+1), ldda);
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(i+1)*nb, m-(i+1)*nb, nb,
                             c_neg_one, dAT(i,   i+1), ldda,
                                        dAT(i+1, i  ), ldda,
                             c_one,     dAT(i+1, i+1), ldda );
            }

	    if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
	    {
		cudaEventRecord(stop_per_iter, 0);
                cudaEventSynchronize(stop_per_iter);
                cudaEventElapsedTime(&per_iter_time_cuda_temp, start_per_iter, stop_per_iter);
                cudaEventDestroy(start_per_iter);
                cudaEventDestroy(stop_per_iter);
	    }

	    if(TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION)
	    {
		printf("iter %d: slack_cuda = %.6f\n", i, cpu_time_cuda_temp/1000 - gpu_time1_cuda_temp/1000);
		printf("iter %d: cpu_time_cuda = %.6f\n", i, cpu_time_cuda_temp/1000);
                printf("iter %d: gpu_time_cuda = %.6f\n", i, gpu_time1_cuda_temp/1000);
		printf("iter %d: download_copy_time_cuda = %.6f\n", i, download_copy_time_cuda_temp/1000);
                printf("iter %d: upload_copy_time_cuda = %.6f\n", i, upload_copy_time_cuda_temp/1000);
                printf("iter %d: per_iter_time_cuda = %.6f\n\n", i, per_iter_time_cuda_temp/1000);
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
	    printf("total_cpu_time_cuda = %.6f\n", total_cpu_time_cuda/1000);
            printf("total_gpu_time_cuda = %.6f\n", total_gpu_time_cuda/1000);
            printf("total_copy_time_cuda = %.6f\n", total_copy_time_cuda/1000);
            printf("main_loop_time_cuda = %.6f\n", main_loop_time_cuda/1000);
            printf("Normalized difference of predicted CPU runtime per iteration is: %.6f\%\n", 100*diff_total_cpu/(m/nb));
            printf("Normalized difference of predicted GPU runtime per iteration is: %.6f\%\n", 100*diff_total_gpu/(m/nb));
            printf("Normalized difference of predicted slack per iteration is: %.6f\%\n", 100*diff_total_slack/(m/nb));
	}
        
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
            magmablas_dpermute_long2( ldda, dAT, ldda, ipiv, nb0, s*nb );
    
            magma_dsetmatrix( rows, nb0, work, lda, dA, cols );
            magmablas_dtranspose( rows, nb0, dA, cols, dAT(s,s), ldda );
    
            magma_dtrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-s*nb-nb0, nb0,
                         c_one, dAT(s, s),     ldda,
                                dAT(s, s)+nb0, ldda);
        }
       
        if (maxdim*maxdim < 2*maxm*maxn) {
            magmablas_dtranspose_inplace( ldda, dAT, ldda );
            magma_dgetmatrix( m, n, da, ldda, A, lda );
        } else {
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
