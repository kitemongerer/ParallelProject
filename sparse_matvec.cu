#include <stdio.h>
#include <assert.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

#define TBS 256

//Multiplies each element of sparse matrix by the correct vector element and puts the result back in the matrix
__global__ void sparseMatrixMul(int *d_indices, float *d_data, float *d_b, int *d_n) {
	int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < *d_n) {
    	d_data[idx] = d_data[idx] * d_b[d_indices[idx]];
    }
}

//Adds each element in a row of a matrix
//Used after sparseMatrixMul it gives the result of matrix multiplication by a vector
__global__ void addResults(int *d_ptr, float *d_data, float *d_result, int *d_nr) {
	int idx = blockIdx.x * 256 + threadIdx.x;
	if (idx < *d_nr) {
		for (int j = d_ptr[idx]; j < d_ptr[idx + 1]; j++) {
			d_result[idx] = d_result[idx] + d_data[j];
		}
	}
}

int main (int argc, char **argv) {
	//int TBS = 256;
	FILE *fp;
	char line[1024]; 
	int *ptr, *indices;
	float *data, *b, *t;
	int i,j;
	int n; // number of nonzero elements in data
	int nr; // number of rows in matrix
	int nc; // number of columns in matrix

	// Open input file and read to end of comments
	if (argc !=2) abort(); 

	if ((fp = fopen(argv[1], "r")) == NULL) {
		abort();
	}

	fgets(line, 128, fp);
	while (line[0] == '%') {
		fgets(line, 128, fp); 
	}

	// Read number of rows (nr), number of columns (nc) and
	// number of elements and allocate memory for ptr, indices, data, b and t.
	sscanf(line,"%d %d %d\n", &nr, &nc, &n);
	ptr = (int *) malloc ((nr + 1) * sizeof(int));
	indices = (int *) malloc(n * sizeof(int));
	data = (float *) malloc(n * sizeof(float));
	b = (float *) malloc(nc * sizeof(float));
	t = (float *) malloc(nr * sizeof(float));

	// Read data in coordinate format and initialize sparse matrix
	int lastr=0;
	for (i = 0; i < n; i++) {
		int r;
		fscanf(fp,"%d %d %f\n", &r, &(indices[i]), &(data[i]));  
		indices[i]--;  // start numbering at 0
		if (r != lastr) { 
			ptr[r-1] = i; 
			lastr = r; 
		}
	}
	ptr[nr] = n;

	// initialize t to 0 and b with random data  
	for (i = 0; i < nr; i++) {
		t[i] = 0.0;
	}

	for (i = 0; i < nc; i++) {
		b[i] = (float) rand()/1111111111;
	}

	int *d_indices, *d_n, *d_nc, *d_nr, *d_ptr;
	float *d_data, *d_b, *d_result; // device copies 
	float *gpu_result = (float *) malloc(nr * sizeof(float)); //local result

	// Allocate space for device copies of a, b, c
	cudaMalloc((void **)&d_indices, n * sizeof(int));
	cudaMalloc((void **)&d_ptr, (nr + 1) * sizeof(int));
	cudaMalloc((void **)&d_data, n * sizeof(float));
	cudaMalloc((void **)&d_b, nc * sizeof(float));
	cudaMalloc((void **)&d_result, nr * sizeof(float));
	cudaMalloc((void **)&d_n, sizeof(int));
	cudaMalloc((void **)&d_nc, sizeof(int));
	cudaMalloc((void **)&d_nr, sizeof(int));

	// Copy inputs to device
	cudaMemcpy(d_indices, indices, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ptr, ptr, (nr + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, nc * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, t, nr * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nc, &nc, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nr, &nr, sizeof(int), cudaMemcpyHostToDevice);


	cudaEvent_t start;
	cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

	// MAIN COMPUTATION, SEQUENTIAL VERSION
	for (i = 0; i < nr; i++) {                                                      
		for (j = ptr[i]; j < ptr[i + 1]; j++) {
			t[i] = t[i] + data[j] * b[indices[j]];
		}
	}

    // Record the start event
    cudaEventRecord(start, NULL);
	int gridSz = ceil(((float) n) / TBS);

	// Launch sparseMatrixMul() kernel on GPU
	sparseMatrixMul<<<gridSz, TBS>>>(d_indices, d_data, d_b, d_n);
	
	// Make sure multiplication is finished
	cudaDeviceSynchronize();
	
	//Change gridSz for addResults to go row by row	
	gridSz = ceil(((float) nr) / TBS);
	// Launch addResults() on GPU
	addResults<<<gridSz, TBS>>>(d_ptr, d_data, d_result, d_nr);

	cudaEventRecord(stop, NULL);

	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    printf("GPU Time= %.3f msec\n", msecTotal);

	// Copy result back to host
	cudaMemcpy(gpu_result, d_result, nr * sizeof(float), cudaMemcpyDeviceToHost);

	bool isCorrect = true;
	for (int i = 0; i < nr; i++) {

		//Print the result if it is wrong
		if (t[i] != gpu_result[i]) {
			printf("%i GPU: %.9f CPU: %.9f\n", i, gpu_result[i], t[i]);
			isCorrect = false;
		}
	}

	if (!isCorrect) {
		printf("The results do not match\n");
	} else {
		printf("The results match\n");
	}

	// Cleanup
	cudaFree(d_indices); 
	cudaFree(d_data); 
	cudaFree(d_b); 
	cudaFree(d_result);

	return 0;
}
