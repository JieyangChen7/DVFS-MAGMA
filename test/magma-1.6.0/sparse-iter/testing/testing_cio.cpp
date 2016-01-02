/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from testing_zio.cpp normal z -> c, Sat Nov 15 19:54:24 2014
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver 
*/
int main(  int argc, char** argv )
{
    TESTING_INIT();

    magma_copts zopts;
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );
    
    int i=1;
    magma_cparse_opts( argc, argv, &zopts, &i, queue );


    real_Double_t res;
    magma_c_sparse_matrix A, A2, A3, A4, A5;

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_cm_5stencil(  laplace_size, &A, queue );
        } else {                        // file-matrix test
            magma_c_csr_mtx( &A,  argv[i], queue );
        }

        printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // filename for temporary matrix storage
        const char *filename = "testmatrix.mtx";

        // write to file
        write_c_csrtomtx( A, filename, queue );

        // read from file
        magma_c_csr_mtx( &A2, filename, queue );

        // delete temporary matrix
        unlink( filename );
                
        //visualize
        printf("A2:\n");
        magma_c_mvisu( A2, queue );
        
        //visualize
        magma_c_mconvert(A2, &A4, Magma_CSR, Magma_CSRL, queue );
        printf("A4:\n");
        magma_c_mvisu( A4, queue );
        magma_c_mconvert(A4, &A5, Magma_CSR, Magma_ELL, queue );
        printf("A5:\n");
        magma_c_mvisu( A5, queue );

        // pass it to another application and back
        magma_int_t m, n;
        magma_index_t *row, *col;
        magmaFloatComplex *val;
        magma_ccsrget( A2, &m, &n, &row, &col, &val, queue );
        magma_ccsrset( m, n, row, col, val, &A3, queue );

        magma_cmdiff( A, A2, &res, queue );
        printf("# ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester IO:  ok\n");
        else
            printf("# tester IO:  failed\n");

        magma_cmdiff( A, A3, &res, queue );
        printf("# ||A-B||_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester matrix interface:  ok\n");
        else
            printf("# tester matrix interface:  failed\n");

        magma_c_mfree(&A, queue ); 
        magma_c_mfree(&A2, queue ); 
        magma_c_mfree(&A4, queue ); 
        magma_c_mfree(&A5, queue ); 


        i++;
    }
    
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
