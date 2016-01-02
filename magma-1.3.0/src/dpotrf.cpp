/*
    -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2012

       @generated d Wed Nov 14 22:52:59 2012

*/
#include "common_magma.h"
#include "../testing/testing_util.cpp"

// === Define what BLAS to use ============================================
#define PRECISION_d
#if (GPUSHMEM <= 200) && (defined(PRECISION_s) || defined(PRECISION_d))
  #define magma_dgemm magmablas_dgemm
  #define magma_dtrsm magmablas_dtrsm
#endif

#if (GPUSHMEM == 200)
#if (defined(PRECISION_s))
     #undef  magma_sgemm
     #define magma_sgemm magmablas_sgemm_fermi80
  #endif
#endif
// === End defining what BLAS to use ======================================

// ========================================================================
// definition of a non-GPU-resident interface to multiple GPUs
extern "C" magma_int_t
magma_dpotrf_m(magma_int_t num_gpus, char uplo, magma_int_t n,
               double *a, magma_int_t lda, magma_int_t *info);
// ========================================================================

#define A(i, j)  (a   +(j)*lda  + (i))
#define dA(i, j) (work+(j)*ldda + (i))

extern "C" magma_int_t 
magma_dpotrf(char uplo, magma_int_t n, 
             double *a, magma_int_t lda, magma_int_t *info)
{
/*  -- MAGMA (version 1.3.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2012

    Purpose   
    =======   

    DPOTRF computes the Cholesky factorization of a real symmetric   
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine.

    The factorization has the form   
       A = U**T * U,  if UPLO = 'U', or   
       A = L  * L**T, if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) DOUBLE_PRECISION array, dimension (LDA,N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U**T * U or A = L * L**T.   

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value 
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    =====================================================================    */


    /* Local variables */
    char uplo_[2] = {uplo, 0};
    magma_int_t        ldda, nb;
    magma_int_t j, jb;
    double    c_one     = MAGMA_D_ONE;
    double    c_neg_one = MAGMA_D_NEG_ONE;
    double   *work;
    double             d_one     =  1.0;
    double             d_neg_one = -1.0;
    int upper = lapackf77_lsame(uplo_, "U");

    *info = 0;
    if ((! upper) && (! lapackf77_lsame(uplo_, "L"))) {
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

    magma_int_t num_gpus = magma_num_gpus();
    if( num_gpus > 1 ) {
      /* call multiple-GPU interface  */
      return magma_dpotrf_m(num_gpus, uplo, n, a, lda, info);
    }

    ldda = ((n+31)/32)*32;
    
    if (MAGMA_SUCCESS != magma_dmalloc( &work, (n)*ldda )) {
        /* alloc failed so call the non-GPU-resident version */
        return magma_dpotrf_m(num_gpus, uplo, n, a, lda, info);
    }

    cudaStream_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    //nb = magma_get_dpotrf_nb(n);
    nb = 64;//optimal block size

    if (nb <= 1 || nb >= n) {
        lapackf77_dpotrf(uplo_, &n, a, &lda, info);
    } else {


        /* Use hybrid blocked code. */
        if (upper) {
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j<n; j += nb) {
                /* Update and factorize the current diagonal block and test   
                   for non-positive-definiteness. Computing MIN */
                jb = min(nb, (n-j));
                magma_dsetmatrix( jb, (n-j), A(j, j), lda, dA(j, j), ldda );
                
                magma_dsyrk(MagmaUpper, MagmaTrans, jb, j, 
                            d_neg_one, dA(0, j), ldda, 
                            d_one,     dA(j, j), ldda);

                magma_dgetmatrix_async( (j+jb), jb,
                                        dA(0, j), ldda,
                                        A(0, j),  lda, stream[1] );
                
                if ( (j+jb) < n) {
                    magma_dgemm(MagmaTrans, MagmaNoTrans, 
                                jb, (n-j-jb), j,
                                c_neg_one, dA(0, j   ), ldda, 
                                           dA(0, j+jb), ldda,
                                c_one,     dA(j, j+jb), ldda);
                }
             
                magma_queue_sync( stream[1] );
                lapackf77_dpotrf(MagmaUpperStr, &jb, A(j, j), &lda, info);
                if (*info != 0) {
                  *info = *info + j;
                  break;
                }
                magma_dsetmatrix_async( jb, jb,
                                        A(j, j),  lda,
                                        dA(j, j), ldda, stream[0] );
                
                if ( (j+jb) < n )
                  magma_dtrsm(MagmaLeft, MagmaUpper, MagmaTrans, MagmaNonUnit, 
                              jb, (n-j-jb),
                              c_one, dA(j, j   ), ldda, 
                                     dA(j, j+jb), ldda);
            }
        } else {
            //real_Double_t cpu_time, gpu_time_dsyrk, gpu_time_dgemm, gpu_time_dtrsm, upload_copy_time1, upload_copy_time2, download_copy_time;
            //real_Double_t total_cpu_time = 0, total_gpu_time = 0, total_copy_time = 0, main_loop_time = 0;
            float cpu_time_cuda_temp, total_cpu_time_cuda = 0;
            float upload_copy_time1_cuda_temp, upload_copy_time2_cuda_temp, download_copy_time_cuda_temp, total_copy_time_cuda = 0;
            float main_loop_time_cuda = 0;
            float gpu_time_dsyrk_cuda_temp, gpu_time_dgemm_cuda_temp, gpu_time_dtrsm_cuda_temp, total_gpu_time_cuda = 0;

            cudaEvent_t start_main_loop, stop_main_loop;
            cudaEvent_t start_upload_copy1, stop_upload_copy1;
            cudaEvent_t start_gpu_dsyrk, stop_gpu_dsyrk;
            cudaEvent_t start_download_copy, stop_download_copy;
            cudaEvent_t start_gpu_dgemm, stop_gpu_dgemm;
            cudaEvent_t start_cpu, stop_cpu;
            cudaEvent_t start_gpu_dtrsm, stop_gpu_dtrsm;
            cudaEvent_t start_upload_copy2, stop_upload_copy2;

            magma_int_t iter = 0;

            #define TIME_MEASUREMENT 1

            if(TIME_MEASUREMENT){
            //main_loop_time = magma_wtime();
            cudaEventCreate(&start_main_loop);
            cudaEventCreate(&stop_main_loop);
            cudaEventRecord(start_main_loop, 0);
            }

            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j<n; j+=nb) {
                if(TIME_MEASUREMENT){
                //upload_copy_time1 = magma_wtime();
                cudaEventCreate(&start_upload_copy1);
                cudaEventCreate(&stop_upload_copy1);
                cudaEventRecord(start_upload_copy1, 0);
                }
                //  Update and factorize the current diagonal block and test   
                //  for non-positive-definiteness. Computing MIN 
                jb = min(nb, (n-j));
                magma_dsetmatrix( (n-j), jb, A(j, j), lda, dA(j, j), ldda );
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

                if(TIME_MEASUREMENT){
                //gpu_time_dsyrk = magma_wtime();
                cudaEventCreate(&start_gpu_dsyrk);
                cudaEventCreate(&stop_gpu_dsyrk);
                cudaEventRecord(start_gpu_dsyrk, 0);
                }
                magma_dsyrk(MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dA(j, 0), ldda, 
                            d_one,     dA(j, j), ldda);
                if(TIME_MEASUREMENT){
                //gpu_time_dsyrk = magma_wtime() - gpu_time_dsyrk;
		//total_gpu_time += gpu_time_dsyrk;
                cudaEventRecord(stop_gpu_dsyrk, 0);
                cudaEventSynchronize(stop_gpu_dsyrk);
                cudaEventElapsedTime(&gpu_time_dsyrk_cuda_temp, start_gpu_dsyrk, stop_gpu_dsyrk);
                cudaEventDestroy(start_gpu_dsyrk);
                cudaEventDestroy(stop_gpu_dsyrk);
		total_gpu_time_cuda += gpu_time_dsyrk_cuda_temp;
                }

                if(TIME_MEASUREMENT){
                //download_copy_time = magma_wtime();
                cudaEventCreate(&start_download_copy);
                cudaEventCreate(&stop_download_copy);
                cudaEventRecord(start_download_copy, 0);
                }
                /*
                magma_dgetmatrix_async( jb, j+jb,
                                        dA(j,0), ldda,
                                        A(j, 0), lda, stream[1] );
                */
                magma_dgetmatrix_async( jb, jb,
                                        dA(j,j), ldda,
                                        A(j,j),  lda, stream[1] );
                magma_dgetmatrix_async( jb, j,
                                        dA(j, 0), ldda,
                                        A(j, 0),  lda, stream[0] );
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

                if(TIME_MEASUREMENT){
                //gpu_time_dgemm = magma_wtime();
                cudaEventCreate(&start_gpu_dgemm);
                cudaEventCreate(&stop_gpu_dgemm);
                cudaEventRecord(start_gpu_dgemm, 0);
                }
                if ( (j+jb) < n) {
                    magma_dgemm( MagmaNoTrans, MagmaTrans, 
                                 (n-j-jb), jb, j,
                                 c_neg_one, dA(j+jb, 0), ldda, 
                                            dA(j,    0), ldda,
                                 c_one,     dA(j+jb, j), ldda);
                }
                if(TIME_MEASUREMENT){
                //gpu_time_dgemm = magma_wtime() - gpu_time_dgemm;
		//total_gpu_time += gpu_time_dgemm;
                cudaEventRecord(stop_gpu_dgemm, 0);
                cudaEventSynchronize(stop_gpu_dgemm);
                cudaEventElapsedTime(&gpu_time_dgemm_cuda_temp, start_gpu_dgemm, stop_gpu_dgemm);
                cudaEventDestroy(start_gpu_dgemm);
                cudaEventDestroy(stop_gpu_dgemm);
		total_gpu_time_cuda += gpu_time_dgemm_cuda_temp;
                }

		//SetCPUFreq("1200000");
		//system("echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
                if(TIME_MEASUREMENT){
                //cpu_time = magma_wtime();
                cudaEventCreate(&start_cpu);
                cudaEventCreate(&stop_cpu);
                cudaEventRecord(start_cpu, 0);
                }
                magma_queue_sync( stream[1] );
                lapackf77_dpotrf(MagmaLowerStr, &jb, A(j, j), &lda, info);
                if (*info != 0){
                    *info = *info + j;
                    break;
                }
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
		//system("echo 2600000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");

                if(TIME_MEASUREMENT){
                //upload_copy_time2 = magma_wtime();
                cudaEventCreate(&start_upload_copy2);
                cudaEventCreate(&stop_upload_copy2);
                cudaEventRecord(start_upload_copy2, 0);
                }
                magma_dsetmatrix_async( jb, jb,
                                        A(j, j),  lda,
                                        dA(j, j), ldda, stream[0] );
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
                //gpu_time_dtrsm = magma_wtime();
                cudaEventCreate(&start_gpu_dtrsm);
                cudaEventCreate(&stop_gpu_dtrsm);
                cudaEventRecord(start_gpu_dtrsm, 0);
                }
                if ( (j+jb) < n)
                    magma_dtrsm(MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit, 
                                (n-j-jb), jb, 
                                c_one, dA(j,    j), ldda, 
                                       dA(j+jb, j), ldda);
                if(TIME_MEASUREMENT){
                //gpu_time_dtrsm = magma_wtime() - gpu_time_dtrsm;
                //total_gpu_time += gput_time_dtrsm;
                cudaEventRecord(stop_gpu_dtrsm, 0);
                cudaEventSynchronize(stop_gpu_dtrsm);
                cudaEventElapsedTime(&gpu_time_dtrsm_cuda_temp, start_gpu_dtrsm, stop_gpu_dtrsm);
                cudaEventDestroy(start_gpu_dtrsm);
                cudaEventDestroy(stop_gpu_dtrsm);
		total_gpu_time_cuda += gpu_time_dtrsm_cuda_temp;
                }

                if(TIME_MEASUREMENT){
                //printf("iter %d: cpu_time = %.6f\n", iter, cpu_time);
                //printf("iter %d: gpu_time_dsyrk = %.6f\n", iter, gpu_time_dsyrk);
                //printf("iter %d: gpu_time_dgemm = %.6f\n", iter, gpu_time_dgemm);
                //printf("iter %d: gpu_time_dtrsm = %.6f\n", iter, gpu_time_dtrsm);
                //printf("iter %d: upload_copy_time1 = %.6f\n", iter, upload_copy_time1);
                //printf("iter %d: download_copy_time = %.6f\n", iter, download_copy_time);
                //printf("iter %d: upload_copy_time2 = %.6f\n", iter, upload_copy_time2);
                printf("iter %d: cpu_time_cuda = %.6f\n", iter, cpu_time_cuda_temp/1000);
                printf("iter %d: gpu_time_dsyrk_cuda = %.6f\n", iter, gpu_time_dsyrk_cuda_temp/1000);
                printf("iter %d: gpu_time_dgemm_cuda = %.6f\n", iter, gpu_time_dgemm_cuda_temp/1000);
		printf("iter %d: gpu_time_dtrsm_cuda = %.6f\n", iter, gpu_time_dtrsm_cuda_temp/1000);
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
        }
    }
    
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );

    magma_free( work );
    
    return *info;
} /* magma_dpotrf */

