/*
 -- MAGMA (version 1.6.1) --
 Univ. of Tennessee, Knoxville
 Univ. of California, Berkeley
 Univ. of Colorado, Denver
 @date January 2015

 @author Stan Tomov
 @generated from zpotrf.cpp normal z -> d, Fri Jan 30 19:00:14 2015
 */
#include "common_magma.h"
#include "../testing/testing_util.cpp"
#include "cuda_profiler_api.h"

#define PRECISION_d

// === Define what BLAS to use ============================================
//#if defined(PRECISION_s) || defined(PRECISION_d)
#define magma_dtrsm magmablas_dtrsm
//#endif
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
 
 If the current stream is NULL, this version replaces it with a new
 stream to overlap computation with communication.

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
 A       DOUBLE_PRECISION array, dimension (LDA,N)
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
extern "C" magma_int_t magma_dpotrf(magma_uplo_t uplo, magma_int_t n, double *A,
		magma_int_t lda, magma_int_t *info) {
#define  A(i_, j_)  (A + (j_)*lda  + (i_))
#define dA(i_, j_) (dA + (j_)*ldda + (i_))

	/* Local variables */
	const char* uplo_ = lapack_uplo_const(uplo);
	magma_int_t ldda, nb;
	magma_int_t j, jb;
	double c_one = MAGMA_D_ONE;
	double c_neg_one = MAGMA_D_NEG_ONE;
	magmaDouble_ptr dA;
	double d_one = 1.0;
	double d_neg_one = -1.0;
	int upper = (uplo == MagmaUpper);

	*info = 0;
	if (!upper && uplo != MagmaLower) {
		*info = -1;
	} else if (n < 0) {
		*info = -2;
	} else if (lda < max(1, n)) {
		*info = -4;
	}
	if (*info != 0) {
		magma_xerbla(__func__, -(*info));
		return *info;
	}

	/* Quick return */
	if (n == 0)
		return *info;

	magma_int_t ngpu = magma_num_gpus();
	if (ngpu > 1) {
		/* call multiple-GPU interface  */
		return magma_dpotrf_m(ngpu, uplo, n, A, lda, info);
	}

	ldda = ((n + 31) / 32) * 32;

	if (MAGMA_SUCCESS != magma_dmalloc(&dA, (n) * ldda)) {
		/* alloc failed so call the non-GPU-resident version */
		return magma_dpotrf_m(ngpu, uplo, n, A, lda, info);
	}

	/* Define user stream if current stream is NULL */
	magma_queue_t stream[3];

	magma_queue_t orig_stream;
	magmablasGetKernelStream(&orig_stream);

	magma_queue_create(&stream[0]);
	magma_queue_create(&stream[2]);

	if (orig_stream == NULL) {
		magma_queue_create(&stream[1]);
		magmablasSetKernelStream(stream[1]);
	} else {
		stream[1] = orig_stream;
	}

	nb = magma_get_dpotrf_nb(n);
	//nb = 103; //optimal

	if (nb <= 1 || nb >= n) {
		lapackf77_dpotrf(uplo_, &n, A, &lda, info);
	} else {
		/* Use hybrid blocked code. */
		if (upper) {
			/* Compute the Cholesky factorization A = U'*U. */
			for (j = 0; j < n; j += nb) {
				/* Update and factorize the current diagonal block and test
				 for non-positive-definiteness. Computing MIN */
				jb = min(nb, (n - j));
				magma_dsetmatrix_async(jb, (n - j), A(j, j), lda, dA(j, j),
						ldda, stream[1]);

				magma_dsyrk(MagmaUpper, MagmaConjTrans, jb, j, d_neg_one,
						dA(0, j), ldda, d_one, dA(j, j), ldda);
				magma_queue_sync(stream[1]);

				magma_dgetmatrix_async(jb, jb, dA(j, j), ldda, A(j, j), lda,
						stream[0]);

				if ((j + jb) < n) {
					magma_dgemm(MagmaConjTrans, MagmaNoTrans, jb, (n - j - jb),
							j, c_neg_one, dA(0, j ), ldda, dA(0, j+jb), ldda,
							c_one, dA(j, j+jb), ldda);
				}

				magma_dgetmatrix_async(j, jb, dA(0, j), ldda, A (0, j), lda,
						stream[2]);

				magma_queue_sync(stream[0]);
				lapackf77_dpotrf(MagmaUpperStr, &jb, A(j, j), &lda, info);
				if (*info != 0) {
					*info = *info + j;
					break;
				}
				magma_dsetmatrix_async(jb, jb, A(j, j), lda, dA(j, j), ldda,
						stream[0]);
				magma_queue_sync(stream[0]);

				if ((j + jb) < n) {
					magma_dtrsm(MagmaLeft, MagmaUpper, MagmaConjTrans,
							MagmaNonUnit, jb, (n - j - jb), c_one, dA(j, j ),
							ldda, dA(j, j+jb), ldda);
				}
			}
		} else {
			/***********
			 * GreenLA *
			 ***********/

			float cpu_time_cuda_temp, total_cpu_time_cuda = 0;
			float upload_copy_time1_cuda_temp, upload_copy_time2_cuda_temp,
					download_copy_time1_cuda_temp,
					download_copy_time2_cuda_temp, total_copy_time_cuda = 0;
			float main_loop_time_cuda = 0;
			float gpu_time_dsyrk_cuda_temp, gpu_time_dgemm_cuda_temp,
					gpu_time_dtrsm_cuda_temp, total_gpu_time_cuda = 0;

			cudaEvent_t start_main_loop, stop_main_loop;
			cudaEvent_t start_upload_copy1, stop_upload_copy1;
			cudaEvent_t start_gpu_dsyrk, stop_gpu_dsyrk;
			cudaEvent_t start_download_copy1, stop_download_copy1;
			cudaEvent_t start_gpu_dgemm, stop_gpu_dgemm;
			cudaEvent_t start_cpu, stop_cpu;
			cudaEvent_t start_gpu_dtrsm, stop_gpu_dtrsm;
			cudaEvent_t start_upload_copy2, stop_upload_copy2;
			cudaEvent_t start_download_copy2, stop_download_copy2;

			magma_int_t iter = 0;
			double ratio_slack_pred = 0;
			double ratio_split_freq = 0;
			double seconds_until_interrupt = 0;
			double diff_total_cpu = 0, diff_total_gpu_dgemm = 0,
					diff_total_slack = 0;
			double gpu_time_dgemm_pred = 0, cpu_time_pred = 0;
			double gpu_time_dgemm_this_iter = 0, cpu_time_this_iter = 0;
			int gpu_time_dgemm_iter0_flag = 0, cpu_time_iter0_flag = 0;
			double gpu_time_dgemm_iter0, cpu_time_iter0;
			
			//static double gpu_time_dgemm_iter0_highest_freq = 0,
			//		gpu_time_dgemm_iter0_lowest_freq = 0;
			//static double cpu_time_iter0_highest_freq = 0;
			
			//double gpu_time_dgemm_this_iter_lowest_freq = gpu_time_dgemm_iter0_lowest_freq;
			int cpu_switched_flag1 = 0;

#define TIME_MEASUREMENT 1
#define TIME_DIFF_CPU_FREQ 0
#define TIME_DIFF_GPU_FREQ 0
#define ALGORITHMIC_SLACK_PREDICTION 0
			
			

			static double gpu_time0_hi = 0.401762;
			static double gpu_time0_lo = 1.753552;
			static double cpu_time0_hi = 1.043;
			static double cpu_time0_lo = 1.043;
			
			
			static double gpu_time_hi = gpu_time0_hi;
			static double gpu_time_lo = gpu_time0_lo;
			static double cpu_time_hi = cpu_time0_hi;
			static double cpu_time_lo = cpu_time0_lo;
			
			
			if (TIME_DIFF_CPU_FREQ)
				SetCPUFreq(1200000);
			if (TIME_DIFF_GPU_FREQ)
				SetGPUFreq(324, 324);

			//=========================================================
			// Compute the Cholesky factorization A = L*L'.
			cudaProfilerStart();
			for (j = 0; j < n; j += nb) {            ////if(j > n/2){nb = 103;

				//  Update and factorize the current diagonal block and test
				//  for non-positive-definiteness. Computing MIN
				jb = min(nb, (n - j));
				magma_dsetmatrix_async((n - j), jb, A(j, j), lda, dA(j, j),
						ldda, stream[1]);


				magma_dsyrk(MagmaLower, MagmaNoTrans, jb, j, d_neg_one,
						dA(j, 0), ldda, d_one, dA(j, j), ldda);


				magma_queue_sync(stream[1]);

				magma_dgetmatrix_async(jb, jb, dA(j,j), ldda, A(j,j), lda,
						stream[0]);


				if (ALGORITHMIC_SLACK_PREDICTION) {
					ratio_slack_pred = 1.0 - (double) nb / (n - iter * nb);
					////cpu_time_pred = cpu_time_pred * ratio_slack_pred;
					gpu_time_dgemm_pred = gpu_time_dgemm_pred * ratio_slack_pred
							* ratio_slack_pred;
					printf("iter %d: cpu_time_pred = %f\n", iter,
							cpu_time_pred);
					printf("iter %d: gpu_time_dgemm_pred = %f\n", iter,
							gpu_time_dgemm_pred);
					printf("iter %d: slack_pred = %f\n", iter,
							cpu_time_pred - gpu_time_dgemm_pred);
				}

				if (TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION) {
					cudaEventCreate(&start_gpu_dgemm);
					cudaEventCreate(&stop_gpu_dgemm);
					cudaEventRecord(start_gpu_dgemm, 0);
				}

				if ((j + jb) < n) {
					magma_dgemm(MagmaNoTrans, MagmaConjTrans, (n - j - jb), jb,
							j, c_neg_one, dA(j+jb, 0), ldda, dA(j, 0), ldda,
							c_one, dA(j+jb, j), ldda);
				}

				if (TIME_MEASUREMENT) {
					cudaEventRecord(stop_gpu_dgemm, 0);
					cudaEventSynchronize(stop_gpu_dgemm);
					cudaEventElapsedTime(&gpu_time_dgemm_cuda_temp,
							start_gpu_dgemm, stop_gpu_dgemm);
					cudaEventDestroy(start_gpu_dgemm);
					cudaEventDestroy(stop_gpu_dgemm);
				}

				magma_dgetmatrix_async(jb, j, dA(j, 0), ldda, A(j, 0), lda,
						stream[2]);

				magma_queue_sync(stream[0]);


				if (TIME_MEASUREMENT) {
					cudaEventCreate(&start_cpu);
					cudaEventCreate(&stop_cpu);
					cudaEventRecord(start_cpu, 0);
				}
				
				
				if ()

				lapackf77_dpotrf(MagmaLowerStr, &jb, A(j, j), &lda, info);

				if (TIME_MEASUREMENT) {
					cudaEventRecord(stop_cpu, 0);
					cudaEventSynchronize(stop_cpu);
					cudaEventElapsedTime(&cpu_time_cuda_temp, start_cpu,
							stop_cpu);
					cudaEventDestroy(start_cpu);
					cudaEventDestroy(stop_cpu);
					total_cpu_time_cuda += cpu_time_cuda_temp;
				}

				if (*info != 0) {
					*info = *info + j;
					break;
				}
				magma_dsetmatrix_async(jb, jb, A(j, j), lda, dA(j, j), ldda,
						stream[0]);
				magma_queue_sync(stream[0]);

				
				if ((j + jb) < n) {
					magma_dtrsm(MagmaRight, MagmaLower, MagmaConjTrans,
							MagmaNonUnit, (n - j - jb), jb, c_one, dA(j, j),
							ldda, dA(j+jb, j), ldda);
				}


				if (TIME_MEASUREMENT || ALGORITHMIC_SLACK_PREDICTION) {
					printf("iter %d: cpu_time_cuda = %.6f\n", iter,
							cpu_time_cuda_temp / 1000);
					printf("iter %d: gpu_time_dgemm_cuda = %.6f\n", iter,
							gpu_time_dgemm_cuda_temp / 1000);
					iter++;
				}
			}               
			cudaProfilerStop();
		}
		
	}

	magma_queue_destroy(stream[0]);
	magma_queue_destroy(stream[2]);
	if (orig_stream == NULL) {
		magma_queue_destroy(stream[1]);
	}
	magmablasSetKernelStream(orig_stream);

	magma_free(dA);

	return *info;
} /* magma_dpotrf */
