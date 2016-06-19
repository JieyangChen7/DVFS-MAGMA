/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Stan Tomov
       @generated from src/zpotrf.cpp normal z -> d, Mon May  2 23:30:01 2016
*/
#include "magma_internal.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "dvfs.h"


// === Define what BLAS to use ============================================
    #undef  magma_dtrsm
    #define magma_dtrsm magmablas_dtrsm
// === End defining what BLAS to use ======================================

/**
    Purpose
    -------
    DPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine.

    The factorization has the form
        A = U**H * U,  if uplo = MagmaUpper, or
        A = L  * L**H, if uplo = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    This uses multiple queues to overlap communication and computation.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    A       DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If uplo = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If uplo = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H * U or A = L * L**H.
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_dposv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dpotrf(
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info )
{
    #define  A(i_, j_)  (A + (i_) + (j_)*lda)
    
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif
    
    /* Constants */
    const double c_one     = MAGMA_D_ONE;
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const double d_one     =  1.0;
    const double d_neg_one = -1.0;
    
    /* Local variables */
    const char* uplo_ = lapack_uplo_const( uplo );
    bool upper = (uplo == MagmaUpper);
    
    magma_int_t j, jb, ldda, nb;
    magmaDouble_ptr dA = NULL;
    
    /* Check arguments */
    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    /* Quick return */
    if ( n == 0 )
        return *info;
    
    nb = magma_get_dpotrf_nb( n );
    
    if (nb <= 1 || nb >= n) {
        lapackf77_dpotrf( uplo_, &n, A, &lda, info );
    }
    else {
        /* Use hybrid blocked code. */
        ldda = magma_roundup( n, 32 );
        
        magma_int_t ngpu = magma_num_gpus();
        if ( ngpu > 1 ) {
            /* call multi-GPU non-GPU-resident interface */
            return magma_dpotrf_m( ngpu, uplo, n, A, lda, info );
        }
        
        if (MAGMA_SUCCESS != magma_dmalloc( &dA, n*ldda )) {
            /* alloc failed so call the non-GPU-resident version */
            return magma_dpotrf_m( ngpu, uplo, n, A, lda, info );
        }
        
        magma_queue_t queues[2] = { NULL, NULL };
        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queues[0] );
        magma_queue_create( cdev, &queues[1] );
        
        if (upper) {
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j < n; j += nb) {
                /* Update and factorize the current diagonal block and test
                   for non-positive-definiteness. */
                jb = min( nb, n-j );
                magma_dsetmatrix_async( jb, n-j,
                                         A(j, j), lda,
                                        dA(j, j), ldda, queues[1] );
                
                magma_dsyrk( MagmaUpper, MagmaConjTrans, jb, j,
                             d_neg_one, dA(0, j), ldda,
                             d_one,     dA(j, j), ldda, queues[1] );
                magma_queue_sync( queues[1] );
                
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                         A(j, j), lda, queues[0] );
                
                if (j+jb < n) {
                    magma_dgemm( MagmaConjTrans, MagmaNoTrans,
                                 jb, n-j-jb, j,
                                 c_neg_one, dA(0, j   ), ldda,
                                            dA(0, j+jb), ldda,
                                 c_one,     dA(j, j+jb), ldda, queues[1] );
                }
                
                magma_queue_sync( queues[0] );
                
                // this could be on any queue; it isn't needed until exit.
                magma_dgetmatrix_async( j, jb,
                                        dA(0, j), ldda,
                                         A(0, j), lda, queues[0] );
                
                lapackf77_dpotrf( MagmaUpperStr, &jb, A(j, j), &lda, info );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                magma_dsetmatrix_async( jb, jb,
                                         A(j, j), lda,
                                        dA(j, j), ldda, queues[0] );
                magma_queue_sync( queues[0] );
                
                if (j+jb < n) {
                    magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                 jb, n-j-jb,
                                 c_one, dA(j, j   ), ldda,
                                        dA(j, j+jb), ldda, queues[1] );
                }
            }
        }
        else {


            //used for timing CPU and GPU
            int iter = 0;
            float cpu_time = 0.0;
            float gpu_time = 0.0;

            double gpu_iter1_low = 2103.143311;
            double gpu_iter1_high = 754.506104;
            double cpu_iter1_low = 794.636108;
            double cpu_iter1_high = 600.295227;

            double gpu_pred_high = gpu_iter1_high;
            double gpu_pred_low = gpu_iter1_low;
            double cpu_pred_high = cpu_iter1_high;
            double cpu_pred_low = cpu_iter1_low;

            double ratio_split_freq = 0;
            double time_until_interrupt = 0;

            cudaEvent_t start_cpu, stop_cpu;
            cudaEvent_t start_gpu, stop_gpu;

            // switches for different modes
            bool timing = false; //for initial setting only, greatly impact performance
            bool dvfs = false; //turn on dvfs energy saving
            bool relax = false; //turn on relax scheme
            bool r2h = false; // turn on race to halt

            //these parameters need to be tuned in future works.
            double dvfs_converage = 0.5;
            double prediction_offset_gpu = 0.65;
            double prediction_offset_cpu = 0.65;

            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j < n; j += nb) {
                //  Update and factorize the current diagonal block and test
                //  for non-positive-definiteness.
                jb = min( nb, n-j );
                magma_dsetmatrix_async( n-j, jb,
                                         A(j, j), lda,
                                        dA(j, j), ldda, queues[1] );
                
                magma_dsyrk( MagmaLower, MagmaNoTrans, jb, j,
                             d_neg_one, dA(j, 0), ldda,
                             d_one,     dA(j, j), ldda, queues[1] );
                magma_queue_sync( queues[1] );
                
                magma_dgetmatrix_async( jb, jb,
                                        dA(j,j), ldda,
                                         A(j,j), lda, queues[0] );
                
                if (timing) {
                    //start gpu timing
                    cudaEventCreate(&start_gpu);
                    cudaEventCreate(&stop_gpu);
                    cudaEventRecord(start_gpu, 0);
                }
                if (j+jb < n) {
                    magma_dgemm( MagmaNoTrans, MagmaConjTrans,
                                 n-j-jb, jb, j,
                                 c_neg_one, dA(j+jb, 0), ldda,
                                            dA(j,    0), ldda,
                                 c_one,     dA(j+jb, j), ldda, queues[1] );
                }


                double ratio_slack_pred = 1.0 - (double)nb/(m-iter*nb);
                cpu_pred_high = cpu_pred_high * ratio_slack_pred;
                cpu_pred_low = cpu_pred_low * ratio_slack_pred;
                gpu_pred_high = gpu_pred_high * ratio_slack_pred * ratio_slack_pred;
                gpu_pred_low = gpu_pred_low * ratio_slack_pred * ratio_slack_pred;

                if (timing) {
                    printf("iter:%d GPU time pred:%f\n", iter, gpu_pred_high);
                    printf("iter:%d CPU time pred:%f\n", iter, cpu_pred_high);
                }

                if (iter < dvfs_converage*(n/nb)) {
                    if (cpu_pred_high > gpu_pred_high) { //slack on GPU
                        ratio_split_freq = (cpu_pred_high - gpu_pred_high) / (gpu_pred_high * ((gpu_iter1_low / gpu_iter1_high) - 1));
                        time_until_interrupt = gpu_pred_low * ratio_split_freq;
                         //printf("iter:%d time_until_interrupt:%f\n", iter, time_until_interrupt);
                        // printf("iter:%d ratio_split_freq:%f\n", iter, ratio_split_freq);
                        if (dvfs) {
                            if ((!relax) || (relax && ratio_split_freq > 0.05)) {
                                if (ratio_split_freq < 1)
                                    dvfs_adjust(time_until_interrupt*prediction_offset_gpu, 'g');
                                else
                                    dvfs_adjust(cpu_pred_high, 'g');
                            }
                        } else if (r2h) {
                            r2h_adjust(gpu_pred_high, cpu_pred_high - gpu_pred_high, 'g');
                        }
                    } else { //slack on CPU
                        ratio_split_freq = (gpu_pred_high - cpu_pred_high) / (cpu_pred_high * ((cpu_iter1_low / cpu_iter1_high) - 1));
                        time_until_interrupt = cpu_pred_low * ratio_split_freq;
                        if (dvfs) {
                            if ((!relax) || (relax && ratio_split_freq > 0.05)) {
                                if (ratio_split_freq < 1)
                                    dvfs_adjust(time_until_interrupt*prediction_offset_cpu, 'c');
                                else
                                    dvfs_adjust(gpu_pred_high, 'c');
                            }
                        } else if (r2h) {
                            r2h_adjust(cpu_pred_high, gpu_pred_high - cpu_pred_high, 'c');
                        }
                    }
                }


                if (timing) {
                    //end gpu timing
                    cudaEventRecord(stop_gpu, 0);
                    cudaEventSynchronize(stop_gpu);
                    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
                    cudaEventDestroy(start_gpu);
                    cudaEventDestroy(stop_gpu);

                    //printf("iter:%d GPU time:%f\n", iter, gpu_time);
                }
                magma_queue_sync( queues[0] );
                
                // this could be on any queue; it isn't needed until exit.
                magma_dgetmatrix_async( jb, j,
                                        dA(j, 0), ldda,
                                         A(j, 0), lda, queues[0] );
                
                if (timing) {
                    //start cpu timing
                    cudaEventCreate(&start_cpu);
                    cudaEventCreate(&stop_cpu);
                    cudaEventRecord(start_cpu, 0);
                }

                lapackf77_dpotrf( MagmaLowerStr, &jb, A(j, j), &lda, info );

                if (timing) {
                    //end cpu timing
                    cudaEventRecord(stop_cpu, 0);
                    cudaEventSynchronize(stop_cpu);
                    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
                    cudaEventDestroy(start_cpu);
                    cudaEventDestroy(stop_cpu);
                    // printf("iter:%d CPU time:%f\n", iter, cpu_time);
                    // if (gpu_time < cpu_time) {
                    //     printf("slack: +\n");
                    // } else {
                    //     printf("slack: -\n");
                    // }
                }

                
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                magma_dsetmatrix_async( jb, jb,
                                         A(j, j), lda,
                                        dA(j, j), ldda, queues[0] );
                magma_queue_sync( queues[0] );
                
                if (j+jb < n) {
                    magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                 n-j-jb, jb,
                                 c_one, dA(j,    j), ldda,
                                        dA(j+jb, j), ldda, queues[1] );
                }
            }
        }
        magma_queue_destroy( queues[0] );
        magma_queue_destroy( queues[1] );
        
        magma_free( dA );
    }
    
    return *info;
} /* magma_dpotrf */
