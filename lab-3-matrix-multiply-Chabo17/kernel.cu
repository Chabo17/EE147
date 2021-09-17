#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

	/*************************************************************************/
      
      __shared__ float ds_A [TILE_SIZE][TILE_SIZE];
      __shared__ float ds_B [TILE_SIZE][TILE_SIZE];

      int bx = blockIdx.x;
      int by = blockIdx.y;
      int tx = threadIdx.x;
      int ty = threadIdx.y;

      int Row = by * blockDim.y + ty;
      int Col = bx * blockDim.x + tx;
      float Pvalue = 0;

      for(int p = 0; p < (TILE_SIZE + k -1)/TILE_SIZE; p++){
 	if(p* TILE_SIZE + tx < k && Row < m){
	  ds_A[ty][tx] = A[Row*k + p*TILE_SIZE + tx];
	}
	else{
	  ds_A[ty][tx] = 0.0;
	}
	if (p*TILE_SIZE + ty < k && Col < n){
	  ds_B[ty][tx] = B[(p*TILE_SIZE + ty)*n + Col];
	}
	else{
	  ds_B[ty][tx] = 0.0;
	}
	__syncthreads();

	for(int i = 0; i < TILE_SIZE; i++){
	  Pvalue += ds_A[ty][i] * ds_B[i][tx];
	}
	
	__syncthreads();

	if(Row < m && Col < n){
	   C[Row*n + Col] = Pvalue;
	}
}
	   
      /*************************************************************************/
}

/********************************************************************
*
* You don't need to use 
*   transa, transb, alpha, beta, lda, ldb, and ldc
* in your code.
*
* They can be used in more complicated matrix multiplication 
* algorithms but not in this assignment.
* Feel free to explore these algorithms.
*
* You can get the complete score by only using
*   m, n, k, A, B, and C   
*
********************************************************************/


void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc){

    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
    dim3 dimGrid((n+ BLOCK_SIZE -1)/BLOCK_SIZE+1, (m+ BLOCK_SIZE + 1)/BLOCK_SIZE+1,1);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);

    mysgemm<<<dimGrid, dimBlock>>>(m,n,k,A,B,C);

}

