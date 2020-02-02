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

//MAXWELL

void do_bublock(int n, int si, int sj, int sk, double *a, double *b, double *c)
{
 int i,j,k;
 for(i=si; i<si + BLOCK_SIZE; i++) {
    for(j=sj; j<sj+BLOCK_SIZE; j++){
       double cij = c[i*n+j];
       for (k=sk; k<sk+BLOCK_SIZE; k++){
           double s1 = a[i*n+k]*b[k*n+j];
           double s2 = a[i*n+(k+1)]*b[(k+1)*n+j];
           double s3 = a[i*n+(k+2)]*b[(k+2)*n+j];
           double s4 = a[i*n+(k+3)]*b[(k+3)*n+j];
           cij += s1 + s2 + s3 + s4;
       }
       c[i*n+j] = cij;
    }
  }
}

void optimized_dgemm_unrolling_block(int n)
{
 int i, j, k;
 for(i=0; i<n; i+=BLOCK_SIZE) {
    for(j=0; j<n; j+=BLOCK_SIZE) {
       c[i*n+j] = 0;
       for(k=0; k<n; k+=BLOCK_SIZE) 
          do_bublock(n, i, j, k, a, b, c);
    }
    }    
}

//OTITO

void optimized_dgemm_simd_unrolling(int n)
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

//NIKOLAI


void do_block_simd(int n, int si, int sj, int sk, double *a, double *b, double *c)
{

 int i, j, k;
 for (i=si; i<si+BLOCK_SIZE; i++)
 for (j=sj; j<sj+BLOCK_SIZE; j++) {
 double cij = c[i*n+j];
 __m256d c4 = _mm256_load_pd(&c[i * n+j]);
 for (k=sk; k<sk+BLOCK_SIZE; k++){
 
 __m256d a4 = _mm256_broadcast_sd(&a[i*n+k]);
 __m256d b4 = _mm256_load_pd(&b[k*n+j]);
 c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));

 }
 _mm256_store_pd(&c[i*n+j], c4);
}
}



void dgemm_blocking_simd(int n)
{
 int i, j, k;
 for(i=0; i<n; i+=BLOCK_SIZE)
 for(j=0; j<n; j+=BLOCK_SIZE) {
 c[i*n+j] = 0;

 for(k=0; k<n; k+=BLOCK_SIZE)
   do_block_simd(n, i, j, k, a, b, c);
 }
}



//GRIFFIN
void do_block_ubs(int n, int si, int sj, int sk, double *a, double *b, double *c) 
{
	int i, j, k;
    for (i=si;i<si+BLOCK_SIZE;i++) 
    {
    	for (j=sj;j<sj+BLOCK_SIZE;j+=4)
        {
        	__m256d c4 = _mm256_load_pd(&c[i*n+j]);
            for (k=0; k<sk+BLOCK_SIZE; k+=2)
            {
            	
            	__m256d a4 = _mm256_broadcast_sd(&a[i*n+k]);
                __m256d b4 = _mm256_load_pd(&b[k*n+j]);
                c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
                a4 = _mm256_broadcast_sd(&a[i*n+(k+1)]);
                b4 = _mm256_load_pd(&b[(k+1)*n+j]);
                c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
                //a4 = _mm256_broadcast_sd(&a[i*n+(k+2)]);
                //b4 = _mm256_load_pd(&b[(k+2)*n+j]);
                //c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
                //a4 = _mm256_broadcast_sd(&a[i*n+(k+3)]);
                //b4 = _mm256_load_pd(&b[(k+3)*n+j]);
                //c4 = _mm256_add_pd(c4, _mm256_mul_pd(a4, b4));
            }
            _mm256_store_pd(&c[i*n+j], c4);
            
        }
    }
}

void optimized_dgemm_unrolling_block_simd(int n)
{
	int i, j, k;
    for (i=0;i<n;i+=BLOCK_SIZE) 
    {
    	for (j=0;j<n;j+=BLOCK_SIZE) 
        {
        	c[i*n+j]=0;
            for(k=0; k<n; k+=BLOCK_SIZE)
            	//printf("hi %d %d %d", i, j, k);
            	do_block_ubs(n, i, j, k, a, b, c);
        }
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
	//dgemm(SIZE);
	//optimized_dgemm_unrolling_block(SIZE);
	//optimized_dgemm_simd_unrolling(SIZE);
	//dgemm_blocking_simd(SIZE);
	optimized_dgemm_unrolling_block_simd(SIZE);
	
	
	//Uncomment one of the above lines to run test. Call command as ./a.out "corr"
	
	
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