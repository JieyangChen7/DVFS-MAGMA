/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Stan Tomov
       @generated from src/zgeqrf.cpp normal z -> d, Mon May  2 23:30:09 2016

*/
#include "magma_internal.h"
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "dvfs.h"

/**
    Purpose
    -------
    DGEQRF computes a QR factorization of a DOUBLE PRECISION M-by-N matrix A:
    A = Q * R. This version does not require work space on the GPU
    passed as input. GPU memory is allocated in the routine.

    This uses 2 queues to overlap communication and computation.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       DOUBLE PRECISION array, dimension (LDA,N)
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
    tau     DOUBLE PRECISION array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.
    \n
            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= max( N*NB, 2*NB*NB ),
            where NB can be obtained through magma_get_dgeqrf_nb( M, N ).
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
    double *A,    magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info )
{
    #define  A(i_,j_)  (A + (i_) + (j_)*lda)
    
    #ifdef HAVE_clBLAS
    #define dA(i_,j_)  dA,    ((i_) + (j_)*ldda + dA_offset)
    #define dT(i_,j_)  dT,    ((i_) + (j_)*nb   + dT_offset)
    #define dwork(i_)  dwork, ((i_)             + dwork_offset)
    #else
    #define dA(i_,j_) (dA    + (i_) + (j_)*ldda)
    #define dT(i_,j_) (dT    + (i_) + (j_)*nb)
    #define dwork(i_) (dwork + (i_))
    #endif
    
    /* Constants */
    const double c_one = MAGMA_D_ONE;
    
    /* Local variables */
    magmaDouble_ptr dA, dT, dwork;
    magma_int_t i, ib, min_mn, ldda, lddwork, old_i, old_ib;
    
    /* Function Body */
    *info = 0;
    magma_int_t nb = magma_get_dgeqrf_nb( m, n );
    
    // need 2*nb*nb to store T and upper triangle of V simultaneously
    magma_int_t lwkopt = max( n*nb, 2*nb*nb );
    work[0] = magma_dmake_lwork( lwkopt );
    bool lquery = (lwork == -1);
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
    
    min_mn = min( m, n );
    if (min_mn == 0) {
        work[0] = c_one;
        return *info;
    }
    
    // largest N for larfb is n-nb (trailing matrix lacks 1st panel)
    lddwork = magma_roundup( n, 32 ) - nb;
    ldda    = magma_roundup( m, 32 );
    
    magma_int_t ngpu = magma_num_gpus();
    if ( ngpu > 1 ) {
        /* call multiple-GPU interface  */
        return magma_dgeqrf_m( ngpu, m, n, A, lda, tau, work, lwork, info );
    }
    
    // allocate space for dA, dwork, and dT
    if (MAGMA_SUCCESS != magma_dmalloc( &dA, n*ldda + nb*lddwork + nb*nb )) {
        /* alloc failed so call non-GPU-resident version */
        return magma_dgeqrf_ooc( m, n, A, lda, tau, work, lwork, info );
    }
    
    dwork = dA + n*ldda;
    dT    = dA + n*ldda + nb*lddwork;
    
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    

    //used for timing CPU and GPU
    int iter = 0;
    float cpu_time = 0.0;
    float gpu_time = 0.0;
    magma_set_lapack_numthreads(1);
    // SetGPUFreq(324, 324);
    // system("echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");

    double gpu_iter1_low = 2103.143311;
    double gpu_iter1_high = 462.029846;
    double cpu_iter1_low = 794.636108;
    double cpu_iter1_high = 1010.850525;

    double gpu_pred_high = gpu_iter1_high;
    double gpu_pred_low = gpu_iter1_low;
    double cpu_pred_high = cpu_iter1_high;
    double cpu_pred_low = cpu_iter1_low;

    double ratio_split_freq = 0;
    double seconds_until_interrupt = 0;

    cudaEvent_t start_cpu, stop_cpu;
    cudaEvent_t start_gpu, stop_gpu;

    bool timing = true;
    bool dvfs = false;
    bool relax = false;

    if ( (nb > 1) && (nb < min_mn) ) {
        /* Use blocked code initially.
           Asynchronously send the matrix to the GPU except the first panel. */
        magma_dsetmatrix_async( m, n-nb,
                                 A(0,nb), lda,
                                dA(0,nb), ldda, queues[0] );
        
        old_i = 0;
        old_ib = nb;
        for (i = 0; i < min_mn-nb; i += nb) {
            ib = min( min_mn-i, nb );
            if (i > 0) {
                /* get i-th panel from device */
                magma_queue_sync( queues[1] );
                magma_dgetmatrix_async( m-i, ib,
                                        dA(i,i), ldda,
                                         A(i,i), lda, queues[0] );
                
                if (timing) {
                    //start gpu timing
                    cudaEventCreate(&start_gpu);
                    cudaEventCreate(&stop_gpu);
                    cudaEventRecord(start_gpu, 0);
                }

                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, n-old_i-2*old_ib, old_ib,
                                  dA(old_i, old_i),          ldda, dT(0,0),  nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork(0), lddwork, queues[1] );
                


                double ratio_slack_pred = 1.0 - (double)nb/(m-iter*nb);
                cpu_pred_high = cpu_pred_high * ratio_slack_pred;
                cpu_pred_low = cpu_pred_low * ratio_slack_pred;
                gpu_pred_high = gpu_pred_high * ratio_slack_pred * ratio_slack_pred;
                gpu_pred_low = gpu_pred_low * ratio_slack_pred * ratio_slack_pred;

                printf("iter:%d GPU time pred:%f\n", iter, gpu_pred_high);
                printf("iter:%d CPU time pred:%f\n", iter, cpu_pred_high);


                if (dvfs && iter > 1 && iter < 1*(min_mn-nb)/nb) {
                    if (cpu_pred_high > gpu_pred_high) { //slack on GPU
                        ratio_split_freq = (cpu_pred_high - gpu_pred_high) / (gpu_pred_high * ((gpu_iter1_low / gpu_iter1_high) - 1));
                        seconds_until_interrupt = gpu_pred_low * ratio_split_freq;
                        if (relax && ratio_split_freq > 0.05) {
                            initialize_handler(0);
                            SetGPUFreq(324, 324);
                            if (ratio_split_freq < 1)
                                //set_timer(seconds_until_interrupt);
                                set_alarm(seconds_until_interrupt);
                            else
                                //set_timer(cpu_time_pred);
                                set_alarm(cpu_pred_high);
                        }
                    } else { //slack on CPU
                        ratio_split_freq = (gpu_pred_high - cpu_pred_high) / (cpu_pred_high * ((cpu_iter1_low / cpu_iter1_high) - 1));
                        seconds_until_interrupt = cpu_pred_low * ratio_split_freq;
                        if (relax && ratio_split_freq > 0.05) {
                            initialize_handler(1);
                            system("echo 1200000 > /sys/devices/system/cpu/cpu0/cpufreq/scaling_setspeed");
                            if (ratio_split_freq < 1)
                                set_alarm(seconds_until_interrupt);
                            else
                                set_alarm(gpu_pred_high);
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

                    printf("iter:%d GPU time:%f\n", iter, gpu_time);
                }



                magma_dgetmatrix_async( i, ib,
                                        dA(0,i), ldda,
                                         A(0,i), lda, queues[1] );
                magma_queue_sync( queues[0] );
            }
            
            magma_int_t rows = m-i;

            if (timing) {
                //start cpu timing
                cudaEventCreate(&start_cpu);
                cudaEventCreate(&stop_cpu);
                cudaEventRecord(start_cpu, 0);
            }

            lapackf77_dgeqrf( &rows, &ib, A(i,i), &lda, tau+i, work, &lwork, info );
            
            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib, A(i,i), &lda, tau+i, work, &ib );
            

            if (timing) {
                //end cpu timing
                cudaEventRecord(stop_cpu, 0);
                cudaEventSynchronize(stop_cpu);
                cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
                cudaEventDestroy(start_cpu);
                cudaEventDestroy(stop_cpu);
                printf("iter:%d CPU time:%f\n", iter, cpu_time);
                if (gpu_time < cpu_time) {
                    printf("slack: +\n");
                } else {
                    printf("slack: -\n");
                }
            }

            magma_dpanel_to_q( MagmaUpper, ib, A(i,i), lda, work+ib*ib );
            
            /* put i-th V matrix onto device */
            magma_dsetmatrix_async( rows, ib, A(i,i), lda, dA(i,i), ldda, queues[0] );
            
            /* put T matrix onto device */
            magma_queue_sync( queues[1] );
            magma_dsetmatrix_async( ib, ib, work, ib, dT(0,0), nb, queues[0] );
            magma_queue_sync( queues[0] );
            
            if (i + ib < n) {
                if (i+ib < min_mn-nb) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left (look-ahead) */
                    magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(i, i   ), ldda, dT(0,0),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
                    magma_dq_to_panel( MagmaUpper, ib, A(i,i), lda, work+ib*ib );
                }
                else {
                    /* After last panel, update whole trailing matrix. */
                    /* Apply H' to A(i:m,i+ib:n) from the left */
                    magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, n-i-ib, ib,
                                      dA(i, i   ), ldda, dT(0,0),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
                    magma_dq_to_panel( MagmaUpper, ib, A(i,i), lda, work+ib*ib );
                }
                
                old_i  = i;
                old_ib = ib;
                iter ++;
            }
        }
    } else {
        i = 0;
    }
    
    /* Use unblocked code to factor the last or only block. */
    if (i < min_mn) {
        ib = n-i;
        if (i != 0) {
            magma_dgetmatrix( m, ib, dA(0,i), ldda, A(0,i), lda, queues[1] );
        }
        magma_int_t rows = m-i;
        lapackf77_dgeqrf( &rows, &ib, A(i,i), &lda, tau+i, work, &lwork, info );
    }
    
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    magma_free( dA );
    
    return *info;
} /* magma_dgeqrf */
