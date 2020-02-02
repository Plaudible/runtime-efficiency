#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <x86intrin.h>
#define ALIGN __attribute__ ((aligned (32)))
#define SIZE 1024
#define BLOCK_SIZE 4
double ALIGN a[SIZE * SIZE];
double ALIGN b[SIZE * SIZE];
double ALIGN c[SIZE * SIZE];
double ALIGN c1[SIZE * SIZE];

// naïve matrix multiplication
void dgemm(int n)
{
    int i,j,k;
    for(i=0; i<n; i++)
    {
        for(j=0; j<n; j++)
        {
            double cij = 0;
            for(k=0; k<n; k++)
                cij = cij + a[i*n+k] * b[k*n+j];
            c1[i*n+j] = cij;
        }
    }
}
/* Implement this function with multiple optimization techniques. */

void optimized_dgemm(int n)
{
    for ( int i = 0; i < n; i+=4 )
        
        for ( int j = 0; j < n; j++ ) {
            
            __m256d c0 = _mm256_load_pd(c+i+j*n); /* c0 = C[i][j] */
            
            for( int k = 0; k < n; k+=4 ){
                
                c0 = _mm256_add_pd(c0,_mm256_mul_pd(_mm256_load_pd(a+i+k*n),_mm256_broadcast_sd(b+k+j*n)));
                
                c0 = _mm256_add_pd(c0,_mm256_mul_pd(_mm256_load_pd(a+i+(k+1)*n),_mm256_broadcast_sd(b+(k+1)+j*n)));
                
                c0 = _mm256_add_pd(c0,_mm256_mul_pd(_mm256_load_pd(a+i+(k+2)*n),_mm256_broadcast_sd(b+(k+2)+j*n)));
                
                c0 = _mm256_add_pd(c0,_mm256_mul_pd(_mm256_load_pd(a+i+(k+3)*n),_mm256_broadcast_sd(b+(k+3)+j*n)));
                
            }                           /* c0 += A[i][k]*B[k][j] */

            _mm256_store_pd(c+i+j*n, c0); /* C[i][j] = c0 */
            
        }
    
}


void main(int argc, char** argv)
{
    int i, j;
    time_t t;
    struct timeval start, end;
    double elapsed_time;
    //Sample Code for DGEMM
    int check_correctness = 0;
    int correct = 1;
    if(argc > 1)
    {
        if(strcmp(argv[1], "corr") == 0)
        {
            check_correctness = 1;
        }
    }
    /* Initialize random number generator */
    srand((unsigned) time(&t));
    /* Populate the arrays with random values */
    for(i=0; i< SIZE; i++)
    {
        for(j=0; j< SIZE; j++)
        {
            a[i* SIZE +j] = (double)rand() / (RAND_MAX + 1.0);
            b[i* SIZE +j] = (double)rand() / (RAND_MAX + 1.0);
            c[i* SIZE +j] = 0.0;
            c1[i* SIZE +j] = 0.0;
        }
    }
    gettimeofday(&start, NULL);
    /* Call you optimized function optimized_dgemm */
    optimized_dgemm(SIZE);
    gettimeofday(&end, NULL);
    /* For TA use only */
    if(check_correctness)
    {
        dgemm(SIZE);
        for(i=0; (i< SIZE) && correct ; i++)
        {
            for(j=0; (j< SIZE) && correct; j++)
            {
                if(fabs(c[i* SIZE +j]-c1[i* SIZE +j]) >= 0.0000001)
                {
                    printf("%f != %f\n", c[i* SIZE +j], c1[i* SIZE +j]);
                    correct = 0;
                }
            }
        }
        if(correct)
            printf("Result is correct!\n");
        else
            printf("Result is incorrect!\n");
    }
    elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsed_time += (end.tv_usec - start.tv_usec) / 1000.0;
    printf("dgemm finished in %f milliseconds.\n", elapsed_time);
    
}
