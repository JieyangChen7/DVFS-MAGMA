/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
       @author Azzam Haidar
       
       @generated from zlaset.cu normal z -> d, Fri Jan 30 19:00:09 2015

*/
#include "common_magma.h"
#include "batched_kernel_param.h"

#define BLK_X 64
#define BLK_Y 32

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to dlacpy, dlag2s, clag2z, dgeadd.
*/
static __device__
void dlaset_full_device(
    int m, int n,
    double offdiag, double diag,
    double *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (below diag || above diag || offdiag == diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y || ind + BLK_X <= iby || MAGMA_D_EQUAL( offdiag, diag )));
    /* do only rows inside matrix */
    if ( ind < m ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block or offdiag == diag
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else
                    A[j*lda] = offdiag;
            }
        }
    }
}


/*
    Similar to dlaset_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.

    Code similar to dlacpy, zlat2c, clat2z.
*/
static __device__
void dlaset_lower_device(
    int m, int n,
    double offdiag, double diag,
    double *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m && ind + BLK_X > iby ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else if ( ind > iby+j )
                    A[j*lda] = offdiag;
            }
        }
    }
}


/*
    Similar to dlaset_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to dlacpy, zlat2c, clat2z.
*/
static __device__
void dlaset_upper_device(
    int m, int n,
    double offdiag, double diag,
    double *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < m && ind < iby + BLK_Y ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else if ( ind < iby+j )
                    A[j*lda] = offdiag;
            }
        }
    }
}
//////////////////////////////////////////////////////////////////////////////////////
/*
    kernel wrapper to call the device function.
*/
__global__
void dlaset_full_kernel(
    int m, int n,
    double offdiag, double diag,
    double *dA, int ldda )
{
    dlaset_full_device(m, n, offdiag, diag, dA, ldda);
}
__global__
void dlaset_lower_kernel(
    int m, int n,
    double offdiag, double diag,
    double *dA, int ldda )
{
    dlaset_lower_device(m, n, offdiag, diag, dA, ldda);
}
__global__
void dlaset_upper_kernel(
    int m, int n,
    double offdiag, double diag,
    double *dA, int ldda )
{
    dlaset_upper_device(m, n, offdiag, diag, dA, ldda);
}
//////////////////////////////////////////////////////////////////////////////////////
/*
    kernel wrapper to call the device function for the batched routine.
*/
__global__
void dlaset_full_kernel_batched(
    int m, int n,
    double offdiag, double diag,
    double **dAarray, int ldda )
{
    int batchid = blockIdx.z;
    dlaset_full_device(m, n, offdiag, diag, dAarray[batchid], ldda);
}
__global__
void dlaset_lower_kernel_batched(
    int m, int n,
    double offdiag, double diag,
    double **dAarray, int ldda )
{
    int batchid = blockIdx.z;
    dlaset_lower_device(m, n, offdiag, diag, dAarray[batchid], ldda);
}
__global__
void dlaset_upper_kernel_batched(
    int m, int n,
    double offdiag, double diag,
    double **dAarray, int ldda )
{
    int batchid = blockIdx.z;
    dlaset_upper_device(m, n, offdiag, diag, dAarray[batchid], ldda);
}
//////////////////////////////////////////////////////////////////////////////////////
/**
    Purpose
    -------
    DLASET_STREAM initializes a 2-D array A to DIAG on the diagonal and
    OFFDIAG on the off-diagonals.
    
    This is the same as DLASET, but adds queue argument.
    
    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be set.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:         All of the matrix dA is set.
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    offdiag DOUBLE_PRECISION
            The scalar OFFDIAG. (In LAPACK this is called ALPHA.)
    
    @param[in]
    diag    DOUBLE_PRECISION
            The scalar DIAG. (In LAPACK this is called BETA.)
    
    @param[in]
    dA      DOUBLE_PRECISION array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
            On exit, A(i,j) = OFFDIAG, 1 <= i <= m, 1 <= j <= n, i != j;
                     A(i,i) = DIAG,    1 <= i <= min(m,n)
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_daux2
    ********************************************************************/
extern "C"
void magmablas_dlaset_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m) )
        info = -7;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    dim3 threads( BLK_X, 1 );
    dim3 grid( (m-1)/BLK_X + 1, (n-1)/BLK_Y + 1 );
    
    if (uplo == MagmaLower) {
        dlaset_lower_kernel<<< grid, threads, 0, queue >>> (m, n, offdiag, diag, dA, ldda);
    }
    else if (uplo == MagmaUpper) {
        dlaset_upper_kernel<<< grid, threads, 0, queue >>> (m, n, offdiag, diag, dA, ldda);
    }
    else {
        // if continuous in memory & set to zero, cudaMemset is faster.
        if ( m == ldda &&
             MAGMA_D_EQUAL( offdiag, MAGMA_D_ZERO ) &&
             MAGMA_D_EQUAL( diag,    MAGMA_D_ZERO ) )
        {
            magma_int_t size = m*n;
            cudaMemsetAsync( dA, 0, size*sizeof(double), queue );
        }
        else {
            dlaset_full_kernel<<< grid, threads, 0, queue >>> (m, n, offdiag, diag, dA, ldda);
        }
    }
}
/**
    @see magmablas_dlaset_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C"
void magmablas_dlaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    magmaDouble_ptr dA, magma_int_t ldda )
{
    magmablas_dlaset_q( uplo, m, n, offdiag, diag, dA, ldda, magma_stream );
}
////////////////////////////////////////////////////////////////////////////////////////

extern "C"
void magmablas_dlaset_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    magmaDouble_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m) )
        info = -7;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    dim3 threads( BLK_X, 1 );
    dim3 grid( (m-1)/BLK_X + 1, (n-1)/BLK_Y + 1, batchCount );
    
    if (uplo == MagmaLower) {
        dlaset_lower_kernel_batched<<< grid, threads, 0, queue >>> (m, n, offdiag, diag, dAarray, ldda);
    }
    else if (uplo == MagmaUpper) {
        dlaset_upper_kernel_batched<<< grid, threads, 0, queue >>> (m, n, offdiag, diag, dAarray, ldda);
    }
    else {
        dlaset_full_kernel_batched<<< grid, threads, 0, queue >>> (m, n, offdiag, diag, dAarray, ldda);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#define A(n, bound) d##A[min(n, bound)]
#define TH_NCHUNK   8
static
__global__ void
zmemset_kernel_batched(int length, double **dAarray, double val)
{
    int tid = threadIdx.x;
    double *dA = dAarray[blockIdx.z];

    #pragma unroll
    for (int l = 0; l < TH_NCHUNK; l++)
        A(l*MAX_NTHREADS+tid, length) = val;
}
#undef A

extern "C"
void magmablas_dmemset_batched(magma_int_t length, 
        magmaDouble_ptr dAarray[], double val, 
        magma_int_t batchCount, magma_queue_t queue)
{

    magma_int_t size_per_block = TH_NCHUNK * MAX_NTHREADS;
    magma_int_t nblock = (length-1)/size_per_block + 1;
    dim3 grid(nblock, 1, batchCount );  // emulate 3D grid: NX * (NY*npages), for CUDA ARCH 1.x

    zmemset_kernel_batched<<< grid, MAX_NTHREADS, 0, queue >>>(length, dAarray, val); 
}


