#include <stdio.h>
#include <stdlib.h>
#include <lcutil.h>
#include <timestamp.h>

#define NXPROB      3600                 /* x dimension of problem grid */
#define NYPROB      3600  
#define STEPS       1000


void inidat(int nx, int ny, float *u) {
int ix, iy;

for (ix = 0; ix <= nx-1; ix++) 
  for (iy = 0; iy <= ny-1; iy++)
     *(u+ix*ny+iy) = (float)(ix * (nx - ix - 1) * iy * (ny - iy - 1));
}
__global__ void update(int, float*, float*);



extern "C" void updateGPU()
{
  	float *u1,*u2, *host_array;
	int it;
	timestamp t_start;	
	
	/* creating two 1d arrays for cuda */

	CUDA_SAFE_CALL(cudaMalloc((void**)&u1,(unsigned long)(NXPROB*NYPROB*sizeof(float))));
	CUDA_SAFE_CALL(cudaMalloc((void**)&u2,(unsigned long)(NXPROB*NYPROB*sizeof(float))));

	/* creating host_array to initialiaze */

	host_array=(float*)malloc(NXPROB*NYPROB*sizeof(float));
	memset(host_array,0,NXPROB*NYPROB*sizeof(float));
	
	/* transfering the host_array to device */

	CUDA_SAFE_CALL(cudaMemcpy(u2,host_array,NXPROB*NYPROB*sizeof(float),cudaMemcpyHostToDevice));

	inidat(NXPROB, NYPROB, host_array);
	
	CUDA_SAFE_CALL(cudaMemcpy(u1,host_array,NXPROB*NYPROB*sizeof(float),cudaMemcpyHostToDevice));	
	 
	dim3 NumberOfThreads(NXPROB-2);			
	dim3 NumberOfBlocks(NYPROB-2);	
	
	int iz = 0; 
	
	t_start = getTimestamp();
	
	for (it = 1; it <= STEPS; it++)
	{
		/* swapping between the two arrays */
       
		if(iz==0){
			update<<<NumberOfBlocks,NumberOfThreads>>>(NYPROB,u1,u2);
		}else{
			update<<<NumberOfBlocks,NumberOfThreads>>>(NYPROB,u2,u1);
		}		
        iz = 1 - iz;
        		
		CUDA_SAFE_CALL(cudaThreadSynchronize() );		
	}
			
	printf("Time elapsed = %6.4lf in ms\n",getElapsedtime(t_start));
	
	/* Copy results back to host memory */

	CUDA_SAFE_CALL( cudaMemcpy(host_array, u2, NXPROB*NYPROB*sizeof(float), cudaMemcpyDeviceToHost) );	
	CUDA_SAFE_CALL( cudaFree(u1) );	
	CUDA_SAFE_CALL( cudaFree(u2) );
	
	
	/*int i,j
	  for(i=0;i<NXPROB;i++){
		for(j=0;j<NYPROB;j++){
			printf("%5.1f ",*(host_array + i*NYPROB + j));
		}
		printf("\n");
	}*/	

    free(host_array);
	
	
}


__global__ void update( int ny, float *u1, float *u2)
{	
	struct Parms { 
		float cx;
		float cy;
	} parms = {0.1, 0.1};
	
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int ix = idx / (ny-2) + 1;
	int iy = idx % (ny-2) + 1;
 
  	
     *(u2+ix*ny+iy) =  *(u1+ix*ny+iy)  + 
                  	  parms.cx * (*(u1+(ix+1)*ny+iy) +
                      *(u1+(ix-1)*ny+iy) - 
                      2.0 * *(u1+ix*ny+iy)) +
                      parms.cy * (*(u1+ix*ny+iy+1) +
                     *(u1+ix*ny+iy-1) - 
                      2.0 * *(u1+ix*ny+iy));	
	
	__syncthreads();
}

	


int main(int argc,char *argv[])
{
	updateGPU();
}





