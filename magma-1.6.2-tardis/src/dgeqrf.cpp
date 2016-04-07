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
#include "cuda_profiler_api.h"

#include "nvml.h"
#include <sys/time.h>
#include <signal.h>

int SetGPUFreq(unsigned int clock_mem, unsigned int clock_core);
static void signal_handler(int signal);
static void set_alarm(double s);
static void initialize_handler(void);


static struct itimerval itv;

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
    magma_int_t nb = magma_get_dgeqrf_nb(min(m, n));

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
        
        double gpu_time0_lowest = 2103.143311;
        double gpu_time0_highest = 461.955383;
        double cpu_time0 = 794.636108;

        float cpu_time = 0.0;
        float gpu_time = 0.0;
        float dvfs_time = 0.0;
        cudaEvent_t start_cpu, stop_cpu;
        cudaEvent_t start_gpu, stop_gpu;
        cudaEvent_t start_dvfs, stop_dvfs;


        double gpu_time_pred = gpu_time0_highest;
        double gpu_time_pred_lowest = gpu_time0_lowest;
        double cpu_time_pred = cpu_time0;

        double ratio_split_freq = 0;
        double seconds_until_interrupt = 0;
        int iter = 0;
        //SetGPUFreq(2600, 705);
        //SetGPUFreq(324, 324);
        bool timing = true;
        bool dvfs = false;

        cudaProfilerStart();
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
                /* download i-th panel */
                magma_queue_sync( stream[1] );
                magma_dgetmatrix_async( m-i, ib,
                                        dA(i,i), ldda,
                                        A(i,i),  lda, stream[0] );



                if (timing) {
                    double ratio_slack_pred = 1.0 - (double)nb/(m-iter*nb);
                    cpu_time_pred = cpu_time_pred * ratio_slack_pred;
                    gpu_time_pred = gpu_time_pred * ratio_slack_pred * ratio_slack_pred;
                    gpu_time_pred_lowest = gpu_time_pred_lowest * ratio_slack_pred * ratio_slack_pred;
                    printf("iter:%d GPU time pred:%f\n", iter, gpu_time_pred);
                    printf("iter:%d CPU time pred:%f\n", iter, cpu_time_pred);
                    
                    ratio_split_freq = (cpu_time_pred - gpu_time_pred) / (gpu_time_pred * ((gpu_time0_lowest / gpu_time0_highest) - 1));
                    seconds_until_interrupt = gpu_time_pred_lowest * ratio_split_freq;
                    printf("iter:%d ratio_split_freq:%f\n", iter, ratio_split_freq);
                    printf("iter:%d seconds_until_interrupt:%f\n", iter, seconds_until_interrupt);
                }

                if (!timing && dvfs && iter > 1) {
                    double ratio_slack_pred = 1.0 - (double)nb/(m-iter*nb);
                    cpu_time_pred = cpu_time_pred * ratio_slack_pred;
                    gpu_time_pred = gpu_time_pred * ratio_slack_pred * ratio_slack_pred;
                    gpu_time_pred_lowest = gpu_time_pred_lowest * ratio_slack_pred * ratio_slack_pred;

                    ratio_split_freq = (cpu_time_pred - gpu_time_pred) / (gpu_time_pred * ((gpu_time0_lowest / gpu_time0_highest) - 1));
                    seconds_until_interrupt = gpu_time_pred_lowest * ratio_split_freq;

                    initialize_handler();
                    SetGPUFreq(324, 324);
                    if (ratio_split_freq < 1)
                        set_alarm(seconds_until_interrupt);
                    else
                        set_alarm(cpu_time_pred);
                }

if (timing) {
                    //start gpu timing
                    cudaEventCreate(&start_dvfs);
                    cudaEventCreate(&stop_dvfs);
                    cudaEventRecord(start_dvfs, 0);

                    SetGPUFreq(324, 324);

                    //end gpu timing
                    cudaEventRecord(stop_dvfs, 0);
                    cudaEventSynchronize(stop_dvfs);
                    cudaEventElapsedTime(&dvfs_time, start_dvfs, stop_dvfs);
                    cudaEventDestroy(start_dvfs);
                    cudaEventDestroy(stop_dvfs);
                    printf("iter:%d dvfs time:%f\n", iter, dvfs_time);

                    // //start gpu timing
                    // cudaEventCreate(&start_dvfs);
                    // cudaEventCreate(&stop_dvfs);
                    // cudaEventRecord(start_dvfs, 0);
                    // SetGPUFreq(2600, 705);
                    
                    // //end gpu timing
                    // cudaEventRecord(stop_dvfs, 0);
                    // cudaEventSynchronize(stop_dvfs);
                    // cudaEventElapsedTime(&dvfs_time, start_dvfs, stop_dvfs);
                    // cudaEventDestroy(start_dvfs);
                    // cudaEventDestroy(stop_dvfs);
                    // printf("iter:%d dvfs time:%f\n", iter, dvfs_time);

                    //                     //start gpu timing
                    // cudaEventCreate(&start_dvfs);
                    // cudaEventCreate(&stop_dvfs);
                    // cudaEventRecord(start_dvfs, 0);
                    // SetGPUFreq(324, 324);
                    
                    // //end gpu timing
                    // cudaEventRecord(stop_dvfs, 0);
                    // cudaEventSynchronize(stop_dvfs);
                    // cudaEventElapsedTime(&dvfs_time, start_dvfs, stop_dvfs);
                    // cudaEventDestroy(start_dvfs);
                    // cudaEventDestroy(stop_dvfs);
                    // printf("iter:%d dvfs time:%f\n", iter, dvfs_time);

                    // //start gpu timing
                    // cudaEventCreate(&start_dvfs);
                    // cudaEventCreate(&stop_dvfs);
                    // cudaEventRecord(start_dvfs, 0);
                    // SetGPUFreq(2600, 705);
                    
                    // //end gpu timing
                    // cudaEventRecord(stop_dvfs, 0);
                    // cudaEventSynchronize(stop_dvfs);
                    // cudaEventElapsedTime(&dvfs_time, start_dvfs, stop_dvfs);
                    // cudaEventDestroy(start_dvfs);
                    // cudaEventDestroy(stop_dvfs);
                    // printf("iter:%d dvfs time:%f\n", iter, dvfs_time);

                    //                     //start gpu timing
                    // cudaEventCreate(&start_dvfs);
                    // cudaEventCreate(&stop_dvfs);
                    // cudaEventRecord(start_dvfs, 0);
                    // SetGPUFreq(324, 324);
                    
                    // //end gpu timing
                    // cudaEventRecord(stop_dvfs, 0);
                    // cudaEventSynchronize(stop_dvfs);
                    // cudaEventElapsedTime(&dvfs_time, start_dvfs, stop_dvfs);
                    // cudaEventDestroy(start_dvfs);
                    // cudaEventDestroy(stop_dvfs);
                    // printf("iter:%d dvfs time:%f\n", iter, dvfs_time);

                    // cudaEventCreate(&start_dvfs);
                    // cudaEventCreate(&stop_dvfs);
                    // cudaEventRecord(start_dvfs, 0);
                    // SetGPUFreq(324, 324);
                    
                    // //end gpu timing
                    // cudaEventRecord(stop_dvfs, 0);
                    // cudaEventSynchronize(stop_dvfs);
                    // cudaEventElapsedTime(&dvfs_time, start_dvfs, stop_dvfs);
                    // cudaEventDestroy(start_dvfs);
                    // cudaEventDestroy(stop_dvfs);
                    // printf("iter:%d dvfs time:%f\n", iter, dvfs_time);

                    // //start gpu timing
                    // cudaEventCreate(&start_dvfs);
                    // cudaEventCreate(&stop_dvfs);
                    // cudaEventRecord(start_dvfs, 0);
                    // SetGPUFreq(2600, 705);
                    
                    // //end gpu timing
                    // cudaEventRecord(stop_dvfs, 0);
                    // cudaEventSynchronize(stop_dvfs);
                    // cudaEventElapsedTime(&dvfs_time, start_dvfs, stop_dvfs);
                    // cudaEventDestroy(start_dvfs);
                    // cudaEventDestroy(stop_dvfs);
                    // printf("iter:%d dvfs time:%f\n", iter, dvfs_time);


                    // //start gpu timing
                    // cudaEventCreate(&start_dvfs);
                    // cudaEventCreate(&stop_dvfs);
                    // cudaEventRecord(start_dvfs, 0);
                    // SetGPUFreq(2600, 705);
                    
                    // //end gpu timing
                    // cudaEventRecord(stop_dvfs, 0);
                    // cudaEventSynchronize(stop_dvfs);
                    // cudaEventElapsedTime(&dvfs_time, start_dvfs, stop_dvfs);
                    // cudaEventDestroy(start_dvfs);
                    // cudaEventDestroy(stop_dvfs);
                    // printf("iter:%d dvfs time:%f\n", iter, dvfs_time);


                }


                if (timing) {
                    //start gpu timing
                    cudaEventCreate(&start_gpu);
                    cudaEventCreate(&stop_gpu);
                    cudaEventRecord(start_gpu, 0);
                }

                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, n-old_i-2*old_ib, old_ib,
                                  dA(old_i, old_i),          ldda, dT,    nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork, lddwork);

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
                                        A(0,i),  lda, stream[1] );
                magma_queue_sync( stream[0] );
            }

            magma_int_t rows = m-i;

            if (timing) {
                //start cpu timing
                cudaEventCreate(&start_cpu);
                cudaEventCreate(&stop_cpu);
                cudaEventRecord(start_cpu, 0);
            }

            lapackf77_dgeqrf(&rows, &ib, A(i,i), &lda, tau+i, work, &lwork, info);
            

            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib, A(i,i), &lda, tau+i, work, &ib);

            if (timing) {
                //end cpu timing
                cudaEventRecord(stop_cpu, 0);
                cudaEventSynchronize(stop_cpu);
                cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
                cudaEventDestroy(start_cpu);
                cudaEventDestroy(stop_cpu);
                printf("iter:%d CPU time:%f\n", iter, cpu_time);
            }


        
            // if (iter == 1) {
            //     cpu_time_pred = cpu_time;
            //     gpu_time_pred = gpu_time;
            // }

            dpanel_to_q(MagmaUpper, ib, A(i,i), lda, work+ib*ib);

            /* download the i-th V matrix */
            magma_dsetmatrix_async( rows, ib, A(i,i), lda, dA(i,i), ldda, stream[0] );

            /* download the T matrix */
            magma_queue_sync( stream[1] );
            magma_dsetmatrix_async( ib, ib, work, ib, dT, nb, stream[0] );
            magma_queue_sync( stream[0] );

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
            iter++;

            if (timing) {
                     cudaEventCreate(&start_dvfs);
                    cudaEventCreate(&stop_dvfs);
                    cudaEventRecord(start_dvfs, 0);

                    SetGPUFreq(2600, 705);

                    //end gpu timing
                    cudaEventRecord(stop_dvfs, 0);
                    cudaEventSynchronize(stop_dvfs);
                    cudaEventElapsedTime(&dvfs_time, start_dvfs, stop_dvfs);
                    cudaEventDestroy(start_dvfs);
                    cudaEventDestroy(stop_dvfs);
                    printf("iter:%d dvfs time:%f\n", iter, dvfs_time);
                }
        }
        cudaProfilerStop();
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



// NVIDIA NVML library function wrapper for GPU DVFS.
int SetGPUFreq(unsigned int clock_mem, unsigned int clock_core) {
    nvmlDevice_t device;//int device;
    nvmlReturn_t result;
    result = nvmlInit();
    result = nvmlDeviceGetHandleByIndex(0, &device);//cudaGetDevice(&device);
    result = nvmlDeviceSetApplicationsClocks(device, clock_mem, clock_core);//(nvmlDevice_t)device
    if(result != NVML_SUCCESS)
    {
        printf("Failed to set GPU core and memory frequencies: %s\n", nvmlErrorString(result));
        return 1;
    }
    else
    {
        nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_GRAPHICS, &clock_core);
        nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_MEM, &clock_mem);
        printf("GPU core frequency is now set to %d MHz; GPU memory frequency is now set to %d MHz", clock_core, clock_mem);
        return 0;
    }
}


static void signal_handler(int signal) {
    SetGPUFreq(2600, 705);//SetGPUFreq(2600, 758);//758 is not stable, it changes to 705 if temp. is high.
    //SetCPUFreq(2500000);
}

static void set_alarm(double s) {
    itv.it_value.tv_sec = (suseconds_t)s;
    itv.it_value.tv_usec = (suseconds_t) ((s-floor(s))*1000000.0);
    setitimer(ITIMER_REAL, &itv, NULL);
}

static void initialize_handler(void) {
    sigset_t sig;
    struct sigaction act;
    sigemptyset(&sig);
    act.sa_handler = signal_handler;
    act.sa_flags = SA_RESTART;
    act.sa_mask = sig;
    sigaction(SIGALRM, &act, NULL);
}
