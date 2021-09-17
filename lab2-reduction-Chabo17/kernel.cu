#define BLOCK_SIZE 512

__global__ void naiveReductionKernel(float *out, float *in, unsigned size)
{
    /********************************************************************
    Implement the naive reduction you learned in class.
    ********************************************************************/
    __shared__ float partialSum[2*BLOCK_SIZE];
    unsigned int thread = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    partialSum[thread] = in[start + thread];
    partialSum[blockDim.x + thread] = in[start + blockDim.x + thread];	
    

   for (unsigned int i = 1; i <= blockDim.x; i *=2){
   __syncthreads();
   if(thread % i ==0){
	partialSum[2*thread] += partialSum[2*thread + i];
   }	 
  }
   
  if(thread == 0){
    out[blockIdx.x] = partialSum[0];
 }
  __syncthreads();
}

void naiveReducion(float *out, float *in_h, unsigned size)
{
    cudaError_t cuda_ret;    

    int i=0;
    float *out_h, *in_d, *out_d;

    unsigned out_size = size / (BLOCK_SIZE<<1);
    if(size % (BLOCK_SIZE<<1)) out_size++;

    // Allocate Host memory ---------------------------------------------------
    out_h = (float*)malloc(out_size * sizeof(float));
    if(out_h == NULL) FATAL("Unable to allocate host in naive reduction.");

    // Allocate device variables ----------------------------------------------
    cuda_ret = cudaMalloc((void**)&in_d, size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in naive reduction.");

    cuda_ret = cudaMalloc((void**)&out_d, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in naive reduction.");

    // Copy host variables to device ------------------------------------------
    cuda_ret = cudaMemcpy(in_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device in naive reduction.");

    cuda_ret = cudaMemset(out_d, 0, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory in naive reduction.");
    
    // Launch kernel ----------------------------------------------------------
    dim3 dim_grid, dim_block;


    /********************************************************************
    Change dim_block & dim_grid.
    ********************************************************************/
    dim_block.x = 512; 
    dim_block.y = 1;
    dim_block.z = 1;
    
    dim_grid.x = (size + 511)/512; 
    dim_grid.y = 1;
    dim_grid.z = 1;


    naiveReductionKernel<<<dim_grid, dim_block>>>(out_d, in_d, size);

    // Copy device variables from host ----------------------------------------
    cuda_ret = cudaMemcpy(out_h, out_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host in naive reduction.");


    /********************************************************************
    For bonus implement this part on GPU. (2.5pts)
    HINT: Think recursively!!
    ********************************************************************/
    /* Accumulate partial sums on host */
    out[0] = 0;
    for( i = 0; i < out_size; i++ ) {
        out[0] += out_h[i];
    }

    
    // Free memory ------------------------------------------------------------
    free(out_h);
    cudaFree(in_d);
    cudaFree(out_d);
}

__global__ void improvedReductionKernel(float *out, float *in, unsigned size)
{
    /********************************************************************
    Implement the better reduction you learned in class.
    ********************************************************************/
   __shared__ float partialSum[2*BLOCK_SIZE];
   unsigned int thread = threadIdx.x;
   unsigned start = blockIdx.x * (BLOCK_SIZE * 2) + thread;
   unsigned int GRID_SIZE = BLOCK_SIZE * 2 * gridDim.x;
   partialSum[thread] = 0;
   

   while (start < size){
   partialSum[thread] += in[start] + in[start + BLOCK_SIZE];
   start += GRID_SIZE;
   } 
   __syncthreads();
}

void improvedReducion(float *out, float *in_h, unsigned size)
{
    cudaError_t cuda_ret;

    int i=0;
    float *out_h, *in_d, *out_d;

    unsigned out_size = size / (BLOCK_SIZE<<1);
    if(size % (BLOCK_SIZE<<1)) out_size++;

    // Allocate Host memory ---------------------------------------------------
    out_h = (float*)malloc(out_size * sizeof(float));
    if(out_h == NULL) FATAL("Unable to allocate host in improved reduction.");

    // Allocate device variables ----------------------------------------------
    cuda_ret = cudaMalloc((void**)&in_d, size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in improved reduction.");

    cuda_ret = cudaMalloc((void**)&out_d, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory in improved reduction.");

    // Copy host variables to device ------------------------------------------
    cuda_ret = cudaMemcpy(in_d, in_h, size * sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device in improved reduction.");

    cuda_ret = cudaMemset(out_d, 0, out_size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory in improved reduction.");
    
    // Launch kernel ----------------------------------------------------------
    dim3 dim_grid, dim_block;


    /********************************************************************
    Change dim_block & dim_grid.
    ********************************************************************/
    dim_block.x = 512; 
    dim_block.y = 1;
    dim_block.z = 1;
    
    dim_grid.x = (size + 511)/512; 
    dim_grid.y = 1;
    dim_grid.z = 1;


    improvedReductionKernel<<<dim_grid, dim_block>>>(out_d, in_d, size);

    // Copy device variables from host ----------------------------------------
    cuda_ret = cudaMemcpy(out_h, out_d, out_size * sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host in improved reduction.");

    /********************************************************************
    For bonus implement this part on GPU. (2.5pts)
    HINT: This is exactly like naiveReduction so free bonus point!!
    ********************************************************************/
    /* Accumulate partial sums on host */
    out[0] = 0;
    for(i = 0; i < out_size; i++ )
        out[0] += out_h[i];


    // Free memory ------------------------------------------------------------
    free(out_h);
    cudaFree(in_d);
    cudaFree(out_d);
}
