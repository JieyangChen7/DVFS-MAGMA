/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Stan Tomov
       @generated from zpotrf_gpu.cpp normal z -> d, Fri Jan 30 19:00:13 2015
*/
#include "common_magma.h"
#include<iostream>


//#include"dpotrfFT.h"
//#include"dtrsmFT.h"
//#include"dsyrkFT.h"
//#include"dgemmFT.h"
#include"FT.h"
#include"papi.h"


using namespace std;

#define PRECISION_d

// === Define what BLAS to use ============================================
//#if (defined(PRECISION_s) || defined(PRECISION_d))
    #define magma_dtrsm magmablas_dtrsm
//#endif
// === End defining what BLAS to use =======================================

/**
    Purpose
    -------
    DPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.
    If the current stream is NULL, this version replaces it with a new
    stream to overlap computation with communication.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    dA      DOUBLE_PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the symmetric matrix dA.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_dposv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dpotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info)
{
#define dA(i, j) (dA + (j)*ldda + (i))

    magma_int_t     j, jb, nb;
    const char* uplo_ = lapack_uplo_const( uplo );
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double *work;
    double          d_one     =  1.0;
    double          d_neg_one = -1.0;
    int upper = (uplo == MagmaUpper);

    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    nb = magma_get_dpotrf_nb(n);

    //** debug **//
    //    nb = 1024;
        
        
    if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    /* Define user stream if current stream is NULL */
    magma_queue_t stream[5];
    
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[2] );
    magma_queue_create( &stream[3] );
    magma_queue_create( &stream[4] );
    if (orig_stream == NULL) {
        magma_queue_create( &stream[1] );
        magmablasSetKernelStream(stream[1]);
    }
    else {
        stream[1] = orig_stream;
    }
    
    
    //acommdation
    int B = nb;
    int N = n;
    //variables for FT
    bool FT = false;
    bool DEBUG = false;
	double * v;
	int v_ld;
	
	double * vd;
	size_t vd_pitch;
	int vd_ld;
	
	double * chk;
	int chk_ld;
	
	double * chk1d;
	size_t chk1d_pitch;
	int chk1d_ld;
	
	double * chk2d;
	size_t chk2d_pitch;
	int chk2d_ld;

	size_t checksum_pitch;
	double * checksum;
	int checksum_ld;
	
	double * temp;
	int temp_ld;
	
	double * chkd_updateA;
	int chkd_updateA_ld;
	
	double * chkd_updateC;
	int chkd_updateC_ld;
//	
//	        	float real_time = 0.0;
//				float proc_time = 0.0;
//				long long flpins = 0.0;
//				float mflops = 0.0;
//				//timing start***************
//				if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
//					cout << "PAPI ERROR" << endl;
//					return -1;
//				}

	if (FT) {
		//cout<<"check sum initialization started"<<endl;
		//intialize checksum vector on CPU

		/* v =
		 * 1 1 1 1
		 * 1 2 3 4 
		 */
		magma_dmalloc_pinned(&v, B * 2 * sizeof(double));
		v_ld = 2;
		for (int i = 0; i < B; ++i) {
			*(v + i * v_ld) = 1;
		}
		for (int i = 0; i < B; ++i) {
			*(v + i * v_ld + 1) = i+1;
		}
//		if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
//							cout << "PAPI ERROR" << endl;
//							return -1;
//						}
//		cout<<"a:"<<real_time<<endl;
//		
//		cout<<"vector on CPU"<<endl;
//		printMatrix_host(v, v_ld, 2, B);
		//cout<<"checksum vector on CPU initialized"<<endl;

		//intialize checksum vector on GPU		
		vd_pitch = magma_roundup(2 * sizeof(double), 32);
		vd_ld = vd_pitch / sizeof(double);	
		magma_dmalloc(&vd, vd_pitch * B * sizeof(double));
		magma_dsetmatrix(2, B, v, v_ld, vd, vd_ld);
		
//		cout<<"vector on GPU"<<endl;
//		printMatrix_gpu(vd, vd_ld, 2, B);
		//cout<<"checksum vector on gpu initialized"<<endl;

		//allocate space for update checksum on CPU
		magma_dmalloc_pinned(&chk, B * 2 * sizeof(double));
		chk_ld = 2;
		//cout<<"allocate space for recalculated checksum on CPU"<<endl;

		//allocate space for reclaculated checksum on GPU
		chk1d_pitch = magma_roundup((N / B) * sizeof(double), 32);
		chk1d_ld = chk1d_pitch / sizeof(double);
		magma_dmalloc(&chk1d, chk1d_pitch * N);
		
		chk2d_pitch = magma_roundup((N / B) * sizeof(double), 32);
		chk2d_ld = chk2d_pitch / sizeof(double);
		magma_dmalloc(&chk2d, chk2d_pitch * N);

		
		//cout<<"allocate space for recalculated checksum on GPU"<<endl;
 
		//initialize checksums
		size_t checksum_pitch = magma_roundup((N / B) * 2 * sizeof(double), 32);
		checksum_ld = checksum_pitch / sizeof(double);
		magma_dmalloc(&checksum, checksum_pitch * N);
		//cudaMemset2D(checksum, checksum_pitch, 0, (N / B) * 2 * sizeof(double), N);
		
		initializeChecksum(dA, ldda, N, B, vd, vd_ld, v, v_ld, checksum, checksum_ld, stream[0]);

		//cout<<"checksums initialized"<<endl;		
	}
    
    
    
    if (0) {
    //if ((nb <= 1) || (nb >= n)) {
        /* Use unblocked code. */
        magma_dgetmatrix_async( n, n, dA, ldda, work, n, stream[1] );
        magma_queue_sync( stream[1] );
        lapackf77_dpotrf(uplo_, &n, work, &n, info);
        magma_dsetmatrix_async( n, n, work, n, dA, ldda, stream[1] );
    }
    else {
        /* Use blocked code. */
        if (upper) {
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j < n; j += nb) {
                
                /* Update and factorize the current diagonal block and test
                   for non-positive-definiteness. Computing MIN */
                jb = min(nb, (n-j));
                
                magma_dsyrk(MagmaUpper, MagmaConjTrans, jb, j,
                            d_neg_one, dA(0, j), ldda,
                            d_one,     dA(j, j), ldda);

                magma_queue_sync( stream[1] );
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, stream[0] );
                
                if ( (j+jb) < n) {
                    /* Compute the current block row. */
                    magma_dgemm(MagmaConjTrans, MagmaNoTrans,
                                jb, (n-j-jb), j,
                                c_neg_one, dA(0, j   ), ldda,
                                           dA(0, j+jb), ldda,
                                c_one,     dA(j, j+jb), ldda);
                }
                
                magma_queue_sync( stream[0] );
                lapackf77_dpotrf(MagmaUpperStr, &jb, work, &jb, info);
                magma_dsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, stream[1] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }

                if ( (j+jb) < n) {
                    magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                 jb, (n-j-jb),
                                 c_one, dA(j, j   ), ldda,
                                        dA(j, j+jb), ldda);
                }
            }
        }
        else {
//        	float noFTtime = 0;
//        	float FTtime = 0;
//
//        	int numOfCore = magma_get_lapack_numthreads();
//        	cout<<"number of core=" << numOfCore<<endl;
//
//        	float real_time = 0.0;
//			float proc_time = 0.0;
//			long long flpins = 0.0;
//			float mflops = 0.0;
//			//timing start***************
//			if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
//				cout << "PAPI ERROR" << endl;
//				return -1;
//			}
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j < n; j += nb) {
                //  Update and factorize the current diagonal block and test
                //  for non-positive-definiteness. Computing MIN
                //jb = min(nb, (n-j));
            	jb = nb;
                if (j > 0) {
  
					dsyrkFT(jb, j, dA(j, 0), ldda, dA(j, j), ldda,
							checksum + (j / jb) * 2, checksum_ld, 
							checksum + (j / jb) * 2 + j * checksum_ld, checksum_ld,
							vd, vd_ld, 
							v, v_ld,
							chk1d, chk1d_ld, 
							chk2d, chk2d_ld, 
							stream,
							FT, DEBUG);
                }

                magma_queue_sync( stream[1] );
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, stream[0] );
                if (FT) {
                	magma_dgetmatrix_async( 2, jb,
                						checksum + (j / B) * 2 + j * checksum_ld, checksum_ld,
										chk,     chk_ld, stream[0] );
                }
                           
                if ( (j+jb) < n && j > 0) {
 
                	dgemmFT((n-j-jb), jb, j, dA(j+jb, 0), ldda,
                			dA(j,    0), ldda, dA(j+jb, j), ldda, 
                			checksum + ((j + jb) / jb) * 2, checksum_ld, 
                			checksum + j * checksum_ld + ((j + jb) / jb) * 2, checksum_ld,
                			vd, vd_ld,
                			chk1d, chk1d_ld,
                			chk2d, chk2d_ld,
                			stream,
                			FT, DEBUG);
                }
                magma_queue_sync( stream[0] );
                dpotrfFT(work, B, B, info, 
                		chk,     chk_ld, 
                		v, v_ld, 
                		FT, DEBUG);
                                
                magma_dsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, stream[1] );
                if (FT) {
                magma_dsetmatrix_async( 2, jb,
                						chk,     chk_ld,
                						checksum + (j / B) * 2 + j * checksum_ld, checksum_ld,
										stream[1] );
                }
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                if ( (j+jb) < n) {     

                	dtrsmFT((n-j-jb), jb, dA(j,    j), ldda,
                			dA(j+jb, j), ldda,
                			checksum + ((j + jb) / jb) * 2 + j * checksum_ld, checksum_ld,
                			vd, vd_ld, 
                			chk1d, chk1d_ld,
                			chk2d, chk2d_ld,
                			FT, DEBUG, stream);
                }
                
            }
            if (FT) {
            	//checksum recalculation
            	char T = 'T';
				char N = 'N';
				char L = 'L';
            	for (int i = 0; i < N; i += B) {
					magma_dgemm(MagmaNoTrans, MagmaNoTrans,
								1, i, B,
								MAGMA_D_ONE, vd, vd_ld,
								dA + i, ldda,
								MAGMA_D_ZERO, chk1d + (i / B), chk1d_ld);
					magma_dgemm(MagmaNoTrans, MagmaNoTrans,
								1, i, B,
								MAGMA_D_ONE, vd + 1, vd_ld,
								dA + i, ldda,
								MAGMA_D_ZERO, chk2d + (i / B), chk2d_ld);
					
					
					blasf77_dtrmv(  &L, &T, &N,
									&B,
									dA + i + i * ldda, &ldda,
									chk1d + (i / B) + i * chk1d_ld, &chk1d_ld );
					blasf77_dtrmv(  &L, &T, &N,
									&B,
									dA + i + i * ldda, &ldda,
									chk2d + (i / B) + i * chk2d_ld, &chk2d_ld );
					
					ErrorDetectAndCorrect(dA + i, ldda, B, B, i + B,
							checksum + (i / B) * 2, checksum_ld,
									chk1d + (i / B), chk1d_ld,
									chk2d + (i / B), chk2d_ld, stream[1]);
					
            	}
            }
            magma_queue_sync( stream[0] );
            magma_queue_sync( stream[1] );
            magma_queue_sync( stream[2] );
            magma_queue_sync( stream[3] );
            magma_queue_sync( stream[4] );
     	
        }
 

        magma_free_pinned( work );
        if (FT) {
        	magma_free(checksum);
        	magma_free(chk1d);
        }
		magma_queue_destroy( stream[0] );
		if (orig_stream == NULL) {
			magma_queue_destroy( stream[1] );
		}
		magmablasSetKernelStream( orig_stream );
		}

    return *info;
} /* magma_dpotrf_gpu */
