/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @author Stan Tomov
       @author Raffaele Solca

       @generated from zlatrd_mgpu.cpp normal z -> s, Sat Nov 15 19:54:10 2014

*/
#include "common_magma.h"
#include "trace.h"


#define PRECISION_s

#define MAGMABLAS_SSYMV_MGPU

/**
    Purpose
    -------
    SLATRD reduces NB rows and columns of a real symmetric matrix A to
    symmetric tridiagonal form by an orthogonal similarity
    transformation Q' * A * Q, and returns the matrices V and W which are
    needed to apply the transformation to the unreduced part of A.

    If UPLO = MagmaUpper, SLATRD reduces the last NB rows and columns of a
    matrix, of which the upper triangle is supplied;
    if UPLO = MagmaLower, SLATRD reduces the first NB rows and columns of a
    matrix, of which the lower triangle is supplied.

    This is an auxiliary routine called by SSYTRD.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored:
      -     = MagmaUpper: Upper triangular
      -     = MagmaLower: Lower triangular

    @param[in]
    n       INTEGER
            The order of the matrix A.

    @param[in]
    nb      INTEGER
            The number of rows and columns to be reduced.

    @param[in,out]
    A       REAL array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = MagmaUpper, the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit:
      -     if UPLO = MagmaUpper, the last NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements above the diagonal
              with the array TAU, represent the orthogonal matrix Q as a
              product of elementary reflectors;
      -     if UPLO = MagmaLower, the first NB columns have been reduced to
              tridiagonal form, with the diagonal elements overwriting
              the diagonal elements of A; the elements below the diagonal
              with the array TAU, represent the  orthogonal matrix Q as a
              product of elementary reflectors.
            See Further Details.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= (1,N).

    @param[out]
    e       REAL array, dimension (N-1)
            If UPLO = MagmaUpper, E(n-nb:n-1) contains the superdiagonal
            elements of the last NB columns of the reduced matrix;
            if UPLO = MagmaLower, E(1:nb) contains the subdiagonal elements of
            the first NB columns of the reduced matrix.

    @param[out]
    tau     REAL array, dimension (N-1)
            The scalar factors of the elementary reflectors, stored in
            TAU(n-nb:n-1) if UPLO = MagmaUpper, and in TAU(1:nb) if UPLO = MagmaLower.
            See Further Details.

    @param[out]
    W       REAL array, dimension (LDW,NB)
            The n-by-nb matrix W required to update the unreduced part
            of A.

    @param[in]
    ldw     INTEGER
            The leading dimension of the array W. LDW >= max(1,N).

    Further Details
    ---------------
    If UPLO = MagmaUpper, the matrix Q is represented as a product of elementary
    reflectors

       Q = H(n) H(n-1) . . . H(n-nb+1).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i),
    and tau in TAU(i-1).

    If UPLO = MagmaLower, the matrix Q is represented as a product of elementary
    reflectors

       Q = H(1) H(2) . . . H(nb).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i) = 0 and v(i+1) = 1; v(i+1:n) is stored on exit in A(i+1:n,i),
    and tau in TAU(i).

    The elements of the vectors v together form the n-by-nb matrix V
    which is needed, with W, to apply the transformation to the unreduced
    part of the matrix, using a symmetric rank-2k update of the form:
    A := A - V*W' - W*V'.

    The contents of A on exit are illustrated by the following examples
    with n = 5 and nb = 2:

    if UPLO = MagmaUpper:                       if UPLO = MagmaLower:

      (  a   a   a   v4  v5 )              (  d                  )
      (      a   a   v4  v5 )              (  1   d              )
      (          a   1   v5 )              (  v1  1   a          )
      (              d   1  )              (  v1  v2  a   a      )
      (                  d  )              (  v1  v2  a   a   a  )

    where d denotes a diagonal element of the reduced matrix, a denotes
    an element of the original matrix that is unchanged, and vi denotes
    an element of the vector defining H(i).

    @ingroup magma_ssyev_aux
    ********************************************************************/
extern "C" magma_int_t
magma_slatrd_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo,
    magma_int_t n0, magma_int_t n, magma_int_t nb, magma_int_t nb0,
    float *A,  magma_int_t lda,
    float *e, float *tau,
    float *W,          magma_int_t ldw,
    magmaFloat_ptr dA[],    magma_int_t ldda, magma_int_t offset,
    magmaFloat_ptr dW[],    magma_int_t lddw,
    magmaFloat_ptr dwork[], magma_int_t ldwork,
    magma_int_t k,
    magmaFloat_ptr dx[],
    magmaFloat_ptr dy[],
    float *work,
    magma_queue_t queues[][10],
    float *times)
{
#define A(i, j) (A + (j)*lda + (i))
#define W(i, j) (W + (j)*ldw + (i))

#define dA(id, i, j)  (dA[(id)] + ((j)+loffset)*ldda + (i) + offset)
#define dW(id, i, j)  (dW[(id)] + (j)          *lddw + (i))
#define dW1(id, i, j) (dW[(id)] + ((j)+nb)     *lddw + (i))

    //float mv_time = 0.0;
    magma_int_t i;
#ifndef MAGMABLAS_SSYMV_MGPU
    magma_int_t loffset = nb0*((offset/nb0)/ngpu);
#endif

    float c_neg_one = MAGMA_S_NEG_ONE;
    float c_one     = MAGMA_S_ONE;
    float c_zero    = MAGMA_S_ZERO;
    float value     = MAGMA_S_ZERO;
    magma_int_t id, idw, i_one = 1;

    //magma_int_t kk;
    magma_int_t ione = 1;

    magma_int_t i_n, i_1, iw;

    float alpha;

    float *dx2[MagmaMaxGPUs];
    float *f;

    // TODO check arguments
    magma_int_t info = 0;
    if (n <= 0) {
        return info;
    }
    
    magma_smalloc_cpu( &f, n );
    if ( f == NULL ) {
        info = MAGMA_ERR_HOST_ALLOC;
        return info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
//#define PROFILE_SYMV
#ifdef PROFILE_SYMV
    magma_event_t start, stop;
    float etime;
    magma_timestr_t cpu_start, cpu_end;
    magma_setdevice(0);
    magma_event_create( &start );
    magma_event_create( &stop  );
#endif

    if (uplo == MagmaUpper) {
        /* Reduce last NB columns of upper triangle */
        for (i = n-1; i >= n - nb; --i) {
            i_1 = i + 1;
            i_n = n - i - 1;
            iw = i - n + nb;
            if (i < n-1) {
                /* Update A(1:i,i) */
                float wii = *W(i, iw+1);
                #if defined(PRECISION_z) || defined(PRECISION_c)
                    lapackf77_slacgv(&i_one, &wii, &ldw);
                #endif
                wii = -wii;
                blasf77_saxpy(&i_1, &wii, A(0, i+1), &i_one, A(0, i), &ione);

                wii = *A(i, i+1);
                #if defined(PRECISION_z) || defined(PRECISION_c)
                    lapackf77_slacgv(&i_one, &wii, &ldw);
                #endif
                wii = -wii;
                blasf77_saxpy(&i_1, &wii, W(0, iw+1), &i_one, A(0, i), &ione);
            }
            if (i > 0) {
                /* Generate elementary reflector H(i) to annihilate A(1:i-2,i) */
                alpha = *A(i-1, i);
                lapackf77_slarfg(&i, &alpha, A(0, i), &ione, &tau[i - 1]);

                e[i-1] = MAGMA_S_REAL( alpha );
                *A(i-1,i) = MAGMA_S_MAKE( 1, 0 );
                for( id=0; id < ngpu; id++ ) {
                    magma_setdevice(id);
                    dx2[id] = dW1(id, 0, iw);
                    magma_ssetvector_async( n, A(0,i), 1, dW1(id, 0, iw), 1, queues[id][0]);
#ifndef  MAGMABLAS_SSYMV_MGPU
                    magma_ssetvector_async( i, A(0,i), 1, dx[id], 1, queues[id][0] );
#endif
                }
                magmablas_ssymv_mgpu(ngpu, k, MagmaUpper, i, nb0, c_one, dA, ldda, 0,
                                     dx2, ione, c_zero, dy, ione, dwork, ldwork,
                                     work, W(0, iw), queues );

                if (i < n-1) {
                    blasf77_sgemv(MagmaConjTransStr, &i, &i_n, &c_one, W(0, iw+1), &ldw,
                                  A(0, i), &ione, &c_zero, W(i+1, iw), &ione);
                }

                /* overlap update */
                if ( i < n-1 && i-1 >= n - nb ) {
                    magma_int_t im1_1 = i_1 - 1;
                    magma_int_t im1   = i-1;
                    /* Update A(1:i,i) */
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        magma_int_t im1_n = i_n + 1;
                        lapackf77_slacgv(&im1_n, W(im1, iw+1), &ldw);
                    #endif
                    blasf77_sgemv("No transpose", &im1_1, &i_n, &c_neg_one, A(0, i+1), &lda,
                                  W(im1, iw+1), &ldw, &c_one, A(0, i-1), &ione);
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_slacgv(&im1_n, W(im1, iw+1), &ldw);
                        lapackf77_slacgv(&im1_n, A(im1, i +1), &lda);
                    #endif
                    blasf77_sgemv("No transpose", &im1_1, &i_n, &c_neg_one, W(0, iw+1), &ldw,
                                  A(im1, i+1), &lda, &c_one, A(0, i-1), &ione);
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_slacgv(&im1_n, A(im1, i+1), &lda);
                    #endif
                }

                // 3. Here is where we need it // TODO find the right place
                magmablas_ssymv_sync(ngpu, k, i, work, W(0, iw), queues );

                if (i < n-1) {
                    blasf77_sgemv("No transpose", &i, &i_n, &c_neg_one, A(0, i+1), &lda,
                                  W(i+1, iw), &ione, &c_one, W(0, iw), &ione);

                    blasf77_sgemv(MagmaConjTransStr, &i, &i_n, &c_one, A(0, i+1), &lda,
                                  A(0, i), &ione, &c_zero, W(i+1, iw), &ione);

                    blasf77_sgemv("No transpose", &i, &i_n, &c_neg_one, W(0, iw+1), &ldw,
                                  W(i+1, iw), &ione, &c_one, W(0, iw), &ione);
                }

                blasf77_sscal(&i, &tau[i - 1], W(0, iw), &ione);

                value = magma_cblas_sdot( i, W(0,iw), ione, A(0,i), ione );
                alpha = tau[i - 1] * -.5f * value;
                blasf77_saxpy(&i, &alpha, A(0, i), &ione, W(0, iw), &ione);

                for( id=0; id < ngpu; id++ ) {
                    magma_setdevice(id);
                    if ( k > 1 ) {
                        magma_ssetvector_async( n, W(0,iw), 1, dW(id, 0, iw), 1, queues[id][1] );
                    } else {
                        magma_ssetvector_async( n, W(0,iw), 1, dW(id, 0, iw), 1, queues[id][0] );
                    }
                }
            }
        }
    } else {
        /*  Reduce first NB columns of lower triangle */
        for (i = 0; i < nb; ++i) {
            /* Update A(i:n,i) */
            i_n = n - i;
            idw = ((offset+i)/nb)%ngpu;
            if ( i > 0 ) {
                trace_cpu_start( 0, "gemv", "gemv" );
                float wii = *W(i, i-1);
                #if defined(PRECISION_z) || defined(PRECISION_c)
                    lapackf77_slacgv(&i_one, &wii, &ldw);
                #endif
                wii = -wii;
                blasf77_saxpy( &i_n, &wii, A(i, i-1), &ione, A(i, i), &ione);

                wii = *A(i, i-1);
                #if defined(PRECISION_z) || defined(PRECISION_c)
                    lapackf77_slacgv(&i_one, &wii, &lda);
                #endif
                wii = -wii;
                blasf77_saxpy( &i_n, &wii, W(i, i-1), &ione, A(i, i), &ione);
            }

            if (i < n-1) {
                /* Generate elementary reflector H(i) to annihilate A(i+2:n,i) */
                i_n = n - i - 1;
                trace_cpu_start( 0, "larfg", "larfg" );
                alpha = *A(i+1, i);
#ifdef PROFILE_SYMV
                cpu_start = get_current_time();
#endif
                lapackf77_slarfg(&i_n, &alpha, A(min(i+2,n-1), i), &ione, &tau[i]);
#ifdef PROFILE_SYMV
                cpu_end = get_current_time();
                times[0] += GetTimerValue(cpu_start,cpu_end)/1000.0;
#endif
                e[i] = MAGMA_S_REAL( alpha );
                *A(i+1,i) = MAGMA_S_MAKE( 1, 0 );
                trace_cpu_end( 0 );

                /* Compute W(i+1:n,i) */
                // 1. Send the block reflector  A(i+1:n,i) to the GPU
                //trace_gpu_start(  idw, 0, "comm", "comm1" );
#ifndef  MAGMABLAS_SSYMV_MGPU
                magma_setdevice(idw);
                magma_ssetvector( i_n, A(i+1,i), 1, dA(idw, i+1, i), 1 );
#endif
                for( id=0; id < ngpu; id++ ) {
                    magma_setdevice(id);
                    trace_gpu_start( id, 0, "comm", "comm" );
#ifdef MAGMABLAS_SSYMV_MGPU
                    dx2[id] = dW1(id, 0, i)-offset;
#else
                    dx2[id] = dx[id];
                    magma_ssetvector( i_n, A(i+1,i), 1, dx[id], 1 );
#endif
                    magma_ssetvector_async( n, A(0,i), 1, dW1(id, 0, i), 1, queues[id][0] );
                    trace_gpu_end( id, 0 );
                }
                /* mat-vec on multiple GPUs */
#ifdef PROFILE_SYMV
                magma_setdevice(0);
                magma_event_record(start, queues[0][0]);
#endif
                magmablas_ssymv_mgpu(ngpu, k, MagmaLower, i_n, nb0, c_one, dA, ldda, offset+i+1,
                                       dx2, ione, c_zero, dy, ione, dwork, ldwork,
                                       work, W(i+1,i), queues );
#ifdef PROFILE_SYMV
                magma_setdevice(0);
                magma_event_record(stop, queues[0][0]);
#endif
                trace_cpu_start( 0, "gemv", "gemv" );
                blasf77_sgemv(MagmaConjTransStr, &i_n, &i, &c_one, W(i+1, 0), &ldw,
                              A(i+1, i), &ione, &c_zero, W(0, i), &ione);
                blasf77_sgemv("No transpose", &i_n, &i, &c_neg_one, A(i+1, 0), &lda,
                              W(0, i), &ione, &c_zero, f, &ione);
                blasf77_sgemv(MagmaConjTransStr, &i_n, &i, &c_one, A(i+1, 0), &lda,
                              A(i+1, i), &ione, &c_zero, W(0, i), &ione);
                trace_cpu_end( 0 );

                /* overlap update */
                if ( i > 0 && i+1 < n ) {
                    magma_int_t ip1 = i+1;
                    trace_cpu_start( 0, "gemv", "gemv" );
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_slacgv(&i, W(ip1, 0), &ldw);
                    #endif
                    blasf77_sgemv("No transpose", &i_n, &i, &c_neg_one, A(ip1, 0), &lda,
                                  W(ip1, 0), &ldw, &c_one, A(ip1, ip1), &ione);
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_slacgv(&i, W(ip1, 0), &ldw);
                        lapackf77_slacgv(&i, A(ip1, 0), &lda);
                    #endif
                    blasf77_sgemv("No transpose", &i_n, &i, &c_neg_one, W(ip1, 0), &ldw,
                                  A(ip1, 0), &lda, &c_one, A(ip1, ip1), &ione);
                    #if defined(PRECISION_z) || defined(PRECISION_c)
                        lapackf77_slacgv(&i, A(ip1, 0), &lda);
                    #endif
                    trace_cpu_end( 0 );
                }

                /* synchronize */
                magmablas_ssymv_sync(ngpu, k, i_n, work, W(i+1,i), queues );
#ifdef PROFILE_SYMV
                cudaEventElapsedTime(&etime, start, stop);
                //mv_time += (etime/1000.0);
                times[1+(i_n/(n0/10))] += (etime/1000.0);
#endif
                trace_cpu_start( 0, "axpy", "axpy" );
                if (i != 0)
                    blasf77_saxpy(&i_n, &c_one, f, &ione, W(i+1, i), &ione);

                blasf77_sgemv("No transpose", &i_n, &i, &c_neg_one, W(i+1, 0), &ldw,
                              W(0, i), &ione, &c_one, W(i+1, i), &ione);
                blasf77_sscal(&i_n, &tau[i], W(i+1,i), &ione);

                value = magma_cblas_sdot( i_n, W(i+1,i), ione, A(i+1,i), ione );
                alpha = tau[i]* -.5f * value;
                blasf77_saxpy(&i_n, &alpha, A(i+1, i), &ione, W(i+1,i), &ione);
                trace_cpu_end( 0 );
                for( id=0; id < ngpu; id++ ) {
                    magma_setdevice(id);
                    if ( k > 1 ) {
                        magma_ssetvector_async( n, W(0,i), 1, dW(id, 0, i), 1, queues[id][1] );
                    } else {
                        magma_ssetvector_async( n, W(0,i), 1, dW(id, 0, i), 1, queues[id][0] );
                    }
                }
            }
        }
    }

#ifdef PROFILE_SYMV
    magma_setdevice(0);
    magma_event_destory( start );
    magma_event_destory( stop  );
#endif
    for( id=0; id < ngpu; id++ ) {
        magma_setdevice(id);
        if ( k > 1 )
            magma_queue_sync( queues[id][1] );
    }
    magma_free_cpu(f);

    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );
    
    return info;
} /* magma_slatrd_mgpu */

#undef A
#undef W
#undef dA
#undef dW
#undef dW1


// ----------------------------------------------------------------------
extern "C" magma_int_t
magmablas_ssymv_mgpu(
    magma_int_t ngpu,
    magma_int_t k, magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb,
    float alpha,
    magmaFloat_ptr dA[],    magma_int_t ldda, magma_int_t offset,
    magmaFloat_ptr dx[],    magma_int_t incx,
    float beta,
    magmaFloat_ptr dy[],    magma_int_t incy,
    magmaFloat_ptr dwork[], magma_int_t ldwork,
    float *work, float *W,
    magma_queue_t queues[][10] )
{
#define dX(id, i)    (dx[(id)]+incx*(i))
#define dY(id, i, j) (dy[(id)]+incy*(i)+n*(j))

    magma_int_t id;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
#ifdef MAGMABLAS_SSYMV_MGPU
    for( id=0; id < ngpu; id++ ) {
        magma_setdevice(id);
        magmablasSetKernelStream( queues[id][0] );
        trace_gpu_start( id, 0, "memset", "memset" );
        cudaMemset( dwork[id], 0, ldwork*sizeof(float) );
        trace_gpu_end( id, 0 );
        trace_gpu_start( id, 0, "symv", "symv" );
    }

    if ( nb == 32 ) {
        magmablas_ssymv_mgpu_32_offset( uplo, offset+n, alpha, dA, ldda,
                                        dx, incx,
                                        beta,
                                        dy, incy,
                                        dwork, ldwork,
                                        ngpu, nb, offset,
                                        queues );
    } else {
        magmablas_ssymv_mgpu_offset( uplo, offset+n, alpha, dA, ldda,
                                     dx, incx,
                                     beta,
                                     dy, incy,
                                     dwork, ldwork,
                                     ngpu, nb, offset,
                                     queues );
    }
    for( id=0; id < ngpu; id++ ) {
        magma_setdevice(id);
        trace_gpu_end( id, 0 );
        magmablasSetKernelStream(NULL);
    }
    //magma_setdevice(0);
    //magmablasSetKernelStream( queues[0][0] );
    //magma_ssymv(MagmaLower, n, alpha, &dA[0][offset+offset*ldda], ldda, &dx[0][offset], incx, beta, &dy[0][offset], incy );
    //magmablasSetKernelStream(NULL);

    /* send to CPU */
    magma_setdevice(0);
    trace_gpu_start( 0, 0, "comm", "comm" );
    magma_sgetvector_async( n, dY(0, offset, 0), 1, W, 1, queues[0][0] );
    trace_gpu_end( 0, 0 );
    magmablasSetKernelStream(NULL);

    for( id=1; id < ngpu; id++ ) {
        magma_setdevice(id);
        trace_gpu_start(  id, 0, "comm", "comm" );
        magma_sgetvector_async( n, dY(id, offset, 0), 1, &work[id*n], 1, queues[id][0] );
        trace_gpu_end( id, 0 );
        magmablasSetKernelStream(NULL);
    }
#else
    float c_one = MAGMA_S_ONE;
    const char* uplo_  = lapack_uplo_const( uplo  );
    magma_int_t i, ii, j, kk, ib, ib0, i_1, i_local, idw;
    magma_int_t i_0=n;
    magma_int_t loffset0 = nb*(offset/(nb*ngpu));
    magma_int_t loffset1 = offset%nb;
    magma_int_t loffset;
    
    //magma_ssymv(uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy );

    idw = (offset/nb)%ngpu;

    for( id=0; id < ngpu; id++ ) {
        magma_setdevice(id);
        magmablasSetKernelStream( queues[id][0] );
        cudaMemset( dy[id], 0, n*k*sizeof(float) );
    }

    if (uplo == MagmaLower) {
        /* the first block */
        if ( loffset1 > 0 ) {
            id = idw;
            kk = 0;

            magma_setdevice(id);
            magmablasSetKernelStream( queues[id][kk] );

            loffset = loffset0+loffset1;
            ib0 = min(nb-loffset1,n);
            // diagonal
            magma_ssymv(MagmaLower, ib0, c_one, dA(id, 0, 0 ), ldda,
                        dX(id, 0), incx, c_one, dY(id, 0, kk), incy);
            // off-diagonl
            if ( ib0 < n ) {
                for( j=ib0; j < n; j += i_0 ) {
                    i_1 = min(i_0, n-j);
                    magma_sgemv(MagmaNoTrans, i_1, ib0, c_one, dA(id, j, 0), ldda,
                                dX(id, 0), incx, c_one, dY(id, j, kk), incy);
                    magma_sgemv(MagmaConjTrans, i_1, ib0, c_one, dA(id, j, 0), ldda,
                                dX(id, j), incx, c_one, dY(id, 0, kk), incy);
                }
            }
        }
        else {
            ib0 = 0;
        }

        /* diagonal */
        for( i=ib0; i < n; i += nb ) {
            id = ((i+offset)/nb)%ngpu;
            kk = ((i+loffset1)/(nb*ngpu))%k;

            magma_setdevice(id);
            magmablasSetKernelStream( queues[id][kk] );

            i_local = (i+loffset1)/(nb*ngpu);
            ib = min(nb,n-i);

            ii = nb*i_local;

            loffset = loffset0;
            if ( id < idw )
                loffset += nb;
            magma_ssymv(MagmaLower,  ib, c_one, dA(id, i, ii), ldda,
                        dX(id, i), incx, c_one, dY(id, i, kk), incy);
        }

        /* off-diagonal */
        for( i=ib0; i < n-nb; i += nb ) {
            id = ((i+offset)/nb)%ngpu;
            kk = ((i+loffset1)/(nb*ngpu))%k;
            magma_setdevice(id);
            magmablasSetKernelStream( queues[id][kk] );

            i_local = ((i+loffset1)/nb)/ngpu;
            ii = nb*i_local;
            ib = min(nb,n-i);
            loffset = loffset0;
            if ( id < idw )
                loffset += nb;

            for( j=i+ib; j < n; j += i_0 ) {
                i_1 = min(i_0, n-j);
                magma_sgemv(MagmaNoTrans, i_1, ib, c_one, dA(id, j, ii), ldda,
                            dX(id, i), incx, c_one, dY(id, j, kk), incy);
                magma_sgemv(MagmaConjTrans, i_1, ib, c_one, dA(id, j, ii), ldda,
                            dX(id, j), incx, c_one, dY(id, i, kk), incy);
            }
        }
    } else { /* upper-triangular storage */
        loffset = 0;
        /* diagonal */
        for( i=0; i < n; i += nb ) {
            id = (i/nb)%ngpu;
            kk = (i/(nb*ngpu))%k;
            ib = min(nb,n-i);

            magma_setdevice(id);
            magmablasSetKernelStream( queues[id][kk] );

            i_local = i/(nb*ngpu);
            ii = nb*i_local;

            magma_ssymv(MagmaUpper, ib, c_one, dA(id, i, ii), ldda,
                        dX(id, i), incx, c_one, dY(id, i, kk), incy);
        }

        /* off-diagonal */
        for( i=nb; i < n; i += nb ) {
            id = (i/nb)%ngpu;
            kk = (i/(nb*ngpu))%k;
            magma_setdevice(id);
            magmablasSetKernelStream( queues[id][kk] );

            i_local = (i/nb)/ngpu;
            ii = nb*i_local;
            ib = min(nb,n-i);

            magma_sgemv(MagmaNoTrans, i, ib, c_one, dA(id, 0, ii), ldda,
                        dX(id, i), incx, c_one, dY(id, 0, kk), incy);
            magma_sgemv(MagmaConjTrans, i, ib, c_one, dA(id, 0, ii), ldda,
                        dX(id, 0), incx, c_one, dY(id, i, kk), incy);
        }
    }
    /* send to CPU */
    magma_setdevice(0);
    magma_sgetvector_async( n, dY(0, 0, 0), 1, W, 1, queues[0][0] );
    for( kk=1; kk < k; kk++ ) {
        magma_sgetvector_async( n, dY(0, 0, kk), 1, &work[kk*n], 1, queues[0][kk] );
    }

    for( id=1; id < ngpu; id++ ) {
        magma_setdevice(id);
        for( kk=0; kk < k; kk++ ) {
            magma_sgetvector_async( n, dY(id, 0, kk), 1, &work[id*k*n + kk*n], 1, queues[id][kk] );
        }
    }
#endif

    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );

    return 0;
}


// ----------------------------------------------------------------------
extern "C" magma_int_t
magmablas_ssymv_sync(
    magma_int_t ngpu,
    magma_int_t k,
    magma_int_t n, float *work, float *W,
    magma_queue_t queues[][10] )
{
    float c_one = MAGMA_S_ONE;
    magma_int_t ione = 1;
    magma_int_t id, kk;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    /* reduce on CPU */
    magma_setdevice(0);
    magma_queue_sync( queues[0][0] );
    for( kk=1; kk < k; kk++ ) {
        magma_queue_sync( queues[0][kk] );
        blasf77_saxpy( &n, &c_one, &work[kk*n], &ione, W, &ione );
    }
    for( id=1; id < ngpu; id++ ) {
        magma_setdevice(id);
        for( kk=0; kk < k; kk++ ) {
            magma_queue_sync( queues[id][kk] );
            blasf77_saxpy( &n, &c_one, &work[id*k*n + kk*n], &ione, W, &ione );
        }
    }

    magma_setdevice( orig_dev );
    
    return 0;
}
