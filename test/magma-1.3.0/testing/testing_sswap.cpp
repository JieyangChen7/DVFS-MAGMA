/*
 *  -- MAGMA (version 1.3.0) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2012
 *
 * @generated s Wed Nov 14 22:54:12 2012
 *
 **/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <assert.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_s

// if ( A==B ) return 0, else return 1
static int diff_matrix( magma_int_t m, magma_int_t n, float *A, magma_int_t lda, float *B, magma_int_t ldb )
{
    for( magma_int_t j = 0; j < n; j++ ) {
        for( magma_int_t i = 0; i < m; i++ ) {
            if ( ! MAGMA_S_EQUAL( A[lda*j+i], B[ldb*j+i] ) )
                return 1;
        }
    }
    return 0;
}

// fill matrix with entries Aij = offset + (i+1) + (j+1)/10000,
// which makes it easy to identify which rows & cols have been swapped.
static void init_matrix( magma_int_t m, magma_int_t n, float *A, magma_int_t lda, magma_int_t offset )
{
    assert( lda >= m );
    for( magma_int_t j = 0; j < n; ++j ) {
        for( magma_int_t i=0; i < m; ++i ) {
            A[i + j*lda] = MAGMA_S_MAKE( offset + (i+1) + (j+1)/10000., 0 );
        }
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sswap, sswapblk, spermute, slaswp, slaswpx
*/
int main( int argc, char** argv) 
{
    TESTING_CUDA_INIT();

    float *h_A1, *h_A2;
    float *d_A1, *d_A2;
    float *h_R1, *h_R2;
    
    // row-major and column-major performance
    real_Double_t row_perf0, col_perf0;
    real_Double_t row_perf1, col_perf1;
    real_Double_t row_perf2, col_perf2;
    real_Double_t row_perf3;
    real_Double_t row_perf4;
    real_Double_t row_perf5, col_perf5;
    real_Double_t row_perf6, col_perf6;
    real_Double_t row_perf7;

    real_Double_t time, gbytes;

    /* Matrix size */
    magma_int_t i, j;
    magma_int_t N=0, lda, ldda, npivots=0;
    const int MAXTESTS = 10;
    magma_int_t size[MAXTESTS] = { 1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112 };
    magma_int_t count = 1;
    
    magma_int_t ione = 1;
    magma_int_t *ipiv, *ipiv2;
    magma_int_t *d_ipiv;
    
    // process command line arguments
    printf( "\nUsage: %s -N n -npivot nb\n"
            "  -N can be repeated up to %d times.\n\n",
            argv[0], MAXTESTS );
    
    int ntest = 0;
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
            magma_assert( ntest < MAXTESTS, "error: -N repeated more than maximum %d tests\n", MAXTESTS );
            size[ ntest ] = atoi( argv[++i] );
            magma_assert( size[ ntest ] > 0, "error: -N %s is invalid; must be > 0.\n", argv[i] );
            N = max( N, size[ ntest ] );
            ntest++;
        }
        else if ( strcmp("-npivots", argv[i]) == 0 && i+1 < argc ) {
            npivots = atoi( argv[++i] );
            magma_assert( npivots > 0, "error: -npivots %s is invalid; must be > 0.\n", argv[i] );
        }
        else if ( strcmp("-count", argv[i]) == 0 && i+1 < argc ) {
            count = atoi( argv[++i] );
            magma_assert( count > 0, "error: -count %s is invalid; must be > 0.\n", argv[i] );
        }
        else {
            printf( "invalid argument: %s\n", argv[i] );
            exit(1);
        }
    }
    if ( ntest == 0 ) {
        ntest = MAXTESTS;
        N = size[ntest-1];
    }
    if ( npivots == 0 ) {
        npivots = magma_get_sgetrf_nb( N );
    }
    npivots = min( npivots, N );

    lda  = N;
    ldda = ((N+31)/32)*32;

    /* Allocate host memory for the matrix */
    TESTING_HOSTALLOC( h_A1, float, lda*N );
    TESTING_HOSTALLOC( h_A2, float, lda*N );
    TESTING_HOSTALLOC( h_R1, float, lda*N );
    TESTING_HOSTALLOC( h_R2, float, lda*N );

    TESTING_MALLOC( ipiv,  magma_int_t, npivots );
    TESTING_MALLOC( ipiv2, magma_int_t, npivots );
    TESTING_DEVALLOC( d_ipiv, magma_int_t, npivots );
    TESTING_DEVALLOC( d_A1, float, ldda*N );
    TESTING_DEVALLOC( d_A2, float, ldda*N );
    
    magma_queue_t queue = 0;
    
    printf( "npivots=%d\n", npivots );
    printf("        cublasSswap       sswap             sswapblk          slaswp   spermute slaswp2  slaswpx           scopymatrix                \n");
    printf("  N     row-maj/col-maj   row-maj/col-maj   row-maj/col-maj   row-maj  row-maj  row-maj  row-maj/col-maj   row-blk/col-blk   (GByte/s)\n");
    printf("=============================================================================================================================\n");
    for( i = 0; i < ntest; ++i ) {
    for( int cnt = 0; cnt < count; ++cnt ) {
        int shift = 1;
        int check = 0;
        N = size[i];
        ldda = ((N+31)/32)*32;
        
        // 2 because for each pair it reads 2 rows and writes 2 rows
        gbytes = 2. * sizeof(float) * npivots * N / 1e9;
        
        for( j=0; j < npivots; j++ ) {
            ipiv[j] = (magma_int_t) ((rand()*1.*N) / (RAND_MAX * 1.)) + 1;
        }

        /* =====================================================================
         * cublasSswap, row-by-row (2 matrices)
         */
        
        /* Row Major */
        init_matrix( N, N, h_A1, lda, 0 );
        init_matrix( N, N, h_A2, lda, 100 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );
        magma_ssetmatrix( N, N, h_A2, lda, d_A2, ldda );
        
        time = magma_sync_wtime( queue );
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                cublasSswap( N, d_A1+ldda*j, 1, d_A2+ldda*(ipiv[j]-1), 1);
            }
        }
        time = magma_sync_wtime( queue ) - time;
        row_perf0 = gbytes / time;
        
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+lda*j, &ione, h_A2+lda*(ipiv[j]-1), &ione);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        magma_sgetmatrix( N, N, d_A2, ldda, h_R2, lda );
        check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                  diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
        shift *= 2;

        /* Column Major */
        init_matrix( N, N, h_A1, lda, 0 );
        init_matrix( N, N, h_A2, lda, 100 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );
        magma_ssetmatrix( N, N, h_A2, lda, d_A2, ldda );

        time = magma_sync_wtime( queue );
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                cublasSswap( N, d_A1+j, ldda, d_A2+ipiv[j]-1, ldda);
            }
        }
        time = magma_sync_wtime( queue ) - time;
        col_perf0 = gbytes / time;
        
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+j, &lda, h_A2+(ipiv[j]-1), &lda);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        magma_sgetmatrix( N, N, d_A2, ldda, h_R2, lda );
        check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                  diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
        shift *= 2;

        /* =====================================================================
         * sswap, row-by-row (2 matrices)
         */
        
        /* Row Major */
        init_matrix( N, N, h_A1, lda, 0 );
        init_matrix( N, N, h_A2, lda, 100 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );
        magma_ssetmatrix( N, N, h_A2, lda, d_A2, ldda );
        
        time = magma_sync_wtime( queue );
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                magmablas_sswap( N, d_A1+ldda*j, 1, d_A2+ldda*(ipiv[j]-1), 1);
            }
        }
        time = magma_sync_wtime( queue ) - time;
        row_perf1 = gbytes / time;
        
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+lda*j, &ione, h_A2+lda*(ipiv[j]-1), &ione);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        magma_sgetmatrix( N, N, d_A2, ldda, h_R2, lda );
        check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                  diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
        shift *= 2;

        /* Column Major */
        init_matrix( N, N, h_A1, lda, 0 );
        init_matrix( N, N, h_A2, lda, 100 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );
        magma_ssetmatrix( N, N, h_A2, lda, d_A2, ldda );

        time = magma_sync_wtime( queue );
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                magmablas_sswap( N, d_A1+j, ldda, d_A2+ipiv[j]-1, ldda );
            }
        }
        time = magma_sync_wtime( queue ) - time;
        col_perf1 = gbytes / time;
        
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+j, &lda, h_A2+(ipiv[j]-1), &lda);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        magma_sgetmatrix( N, N, d_A2, ldda, h_R2, lda );
        check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                  diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
        shift *= 2;

        /* =====================================================================
         * sswapblk, blocked version (2 matrices)
         */

        /* Row Major */
        init_matrix( N, N, h_A1, lda, 0 );
        init_matrix( N, N, h_A2, lda, 100 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );
        magma_ssetmatrix( N, N, h_A2, lda, d_A2, ldda );

        time = magma_sync_wtime( queue );
        magmablas_sswapblk( 'R', N, d_A1, ldda, d_A2, ldda, 1, npivots, ipiv, 1, 0);
        time = magma_sync_wtime( queue ) - time;
        row_perf2 = gbytes / time;
        
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+lda*j, &ione, h_A2+lda*(ipiv[j]-1), &ione);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        magma_sgetmatrix( N, N, d_A2, ldda, h_R2, lda );
        check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                  diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
        shift *= 2;

        /* Column Major */
        init_matrix( N, N, h_A1, lda, 0 );
        init_matrix( N, N, h_A2, lda, 100 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );
        magma_ssetmatrix( N, N, h_A2, lda, d_A2, ldda );

        time = magma_sync_wtime( queue );
        magmablas_sswapblk( 'C', N, d_A1, ldda, d_A2, ldda, 1, npivots, ipiv, 1, 0);
        time = magma_sync_wtime( queue ) - time;
        col_perf2 = gbytes / time;
        
        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+j, &lda, h_A2+(ipiv[j]-1), &lda);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        magma_sgetmatrix( N, N, d_A2, ldda, h_R2, lda );
        check += (diff_matrix( N, N, h_A1, lda, h_R1, lda ) ||
                  diff_matrix( N, N, h_A2, lda, h_R2, lda ))*shift;
        shift *= 2;

        /* =====================================================================
         * spermute_long (1 matrix)
         */

        /* Row Major */
        memcpy( ipiv2, ipiv, npivots*sizeof(magma_int_t) );  // spermute updates ipiv2
        init_matrix( N, N, h_A1, lda, 0 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );

        time = magma_sync_wtime( queue );
        magmablas_spermute_long2( N, d_A1, ldda, ipiv2, npivots, 0 );
        time = magma_sync_wtime( queue ) - time;
        row_perf3 = gbytes / time;

        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+lda*j, &ione, h_A1+lda*(ipiv[j]-1), &ione);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
        shift *= 2;

        /* =====================================================================
         * LAPACK-style slaswp (1 matrix)
         */

        /* Row Major */
        init_matrix( N, N, h_A1, lda, 0 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );

        time = magma_sync_wtime( queue );
        magmablas_slaswp( N, d_A1, ldda, 1, npivots, ipiv, 1);
        time = magma_sync_wtime( queue ) - time;
        row_perf4 = gbytes / time;

        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+lda*j, &ione, h_A1+lda*(ipiv[j]-1), &ione);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
        shift *= 2;

        /* =====================================================================
         * LAPACK-style slaswp (1 matrix) - d_ipiv on GPU
         */

        /* Row Major */
        init_matrix( N, N, h_A1, lda, 0 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );

        time = magma_sync_wtime( queue );
        magma_setvector( npivots, sizeof(magma_int_t), ipiv, 1, d_ipiv, 1 );
        magmablas_slaswp2( N, d_A1, ldda, 1, npivots, d_ipiv );
        time = magma_sync_wtime( queue ) - time;
        row_perf7 = gbytes / time;

        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+lda*j, &ione, h_A1+lda*(ipiv[j]-1), &ione);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
        shift *= 2;
        
        /* =====================================================================
         * LAPACK-style slaswpx (extended for row- and col-major) (1 matrix)
         */

        /* Row Major */
        init_matrix( N, N, h_A1, lda, 0 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );

        time = magma_sync_wtime( queue );
        magmablas_slaswpx( N, d_A1, ldda, 1, 1, npivots, ipiv, 1);
        time = magma_sync_wtime( queue ) - time;
        row_perf5 = gbytes / time;

        for( j=0; j < npivots; j++) {
            if ( j != (ipiv[j]-1)) {
                blasf77_sswap( &N, h_A1+lda*j, &ione, h_A1+lda*(ipiv[j]-1), &ione);
            }
        }
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
        shift *= 2;

        /* Col Major */
        init_matrix( N, N, h_A1, lda, 0 );
        magma_ssetmatrix( N, N, h_A1, lda, d_A1, ldda );

        time = magma_sync_wtime( queue );
        magmablas_slaswpx( N, d_A1, 1, ldda, 1, npivots, ipiv, 1);
        time = magma_sync_wtime( queue ) - time;
        col_perf5 = gbytes / time;

        lapackf77_slaswp( &N, h_A1, &lda, &ione, &npivots, ipiv, &ione);
        magma_sgetmatrix( N, N, d_A1, ldda, h_R1, lda );
        check += diff_matrix( N, N, h_A1, lda, h_R1, lda )*shift;
        shift *= 2;
        
        /* =====================================================================
         * Copy matrix.
         */

        time = magma_sync_wtime( queue );
        magma_scopymatrix( N, npivots, d_A1, ldda, d_A2, ldda );
        time = magma_sync_wtime( queue ) - time;
        // copy reads 1 matrix and writes 1 matrix, so has half gbytes of swap
        col_perf6 = 0.5 * gbytes / time;

        time = magma_sync_wtime( queue );
        magma_scopymatrix( npivots, N, d_A1, ldda, d_A2, ldda );
        time = magma_sync_wtime( queue ) - time;
        // copy reads 1 matrix and writes 1 matrix, so has half gbytes of swap
        row_perf6 = 0.5 * gbytes / time;

        printf("%5d   %6.2f%c/ %6.2f%c  %6.2f%c/ %6.2f%c  %6.2f%c/ %6.2f%c  %6.2f%c  %6.2f%c  %6.2f%c  %6.2f%c/ %6.2f%c  %6.2f / %6.2f   (%s: check=0x%04x)\n",
               N,
               row_perf0, ((check & 0x001) != 0 ? '*' : ' '),
               col_perf0, ((check & 0x002) != 0 ? '*' : ' '),
               row_perf1, ((check & 0x004) != 0 ? '*' : ' '),
               col_perf1, ((check & 0x008) != 0 ? '*' : ' '),
               row_perf2, ((check & 0x010) != 0 ? '*' : ' '),
               col_perf2, ((check & 0x020) != 0 ? '*' : ' '),
               row_perf3, ((check & 0x040) != 0 ? '*' : ' '),          
               row_perf4, ((check & 0x080) != 0 ? '*' : ' '),          
               row_perf7, ((check & 0x100) != 0 ? '*' : ' '),          
               row_perf5, ((check & 0x200) != 0 ? '*' : ' '),
               col_perf5, ((check & 0x400) != 0 ? '*' : ' '),
               row_perf6,
               col_perf6,
               (check == 0 ? "all succeeded" : "* failures"), check );
        }
        if ( count > 1 ) {
            printf( "\n" );
        }
    }
    
    /* Memory clean up */
    TESTING_HOSTFREE( h_A1 );
    TESTING_HOSTFREE( h_A2 );
    TESTING_HOSTFREE( h_R1 );
    TESTING_HOSTFREE( h_R2 );
    TESTING_DEVFREE(  d_A1 );
    TESTING_DEVFREE(  d_A2 );
    TESTING_FREE( ipiv );

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
}
