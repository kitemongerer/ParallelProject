#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>

// CUDA runtime
#include <cuda_runtime.h>

using namespace std;

// Thread block size
#define TBS 512

// Warp size
#define WS 32

class Node {
private:
	int value;
	Node* children;
	int numChildren;
	int explored;

public:
	Node();
	Node(int);
	int getValue();
	void addChild(Node);
	Node* getChildren();
	int getNumChildren();
	void printNode();
	void initializeChildren(int);
	int getExplored();
	void setExplored(int);
};

/*__global__ void addBase(int *d_array, int *d_size, int *d_base) {
	int idx = blockIdx.x * TBS + threadIdx.x;
	if (idx < *d_size && idx >= TBS) {
		d_array[idx] = d_array[idx] + d_base[blockIdx.x];
	}
}

__device__ int scan_warp(int *d_array, int idx) { 
	int lane = idx % WS;

	int i = 1;
	while(i < WS && lane >= i) {
		d_array[idx] = d_array[idx - i] + d_array[idx];
		i *= 2;
	}

	return (lane > 0) ? d_array[idx - 1] : 0; 
}

//Multiplies each element of sparse matrix by the correct vector element and puts the result back in the matrix
__global__ void allPrefixSums(int *d_array, int *d_size, int *d_base) {
	int idx = blockIdx.x * TBS + threadIdx.x;
	if (idx < *d_size) {
		int warpId = (idx / WS) % (TBS / WS);
		int lane = idx % WS;
		int arrayVal = d_array[idx];
		int val = scan_warp(d_array, idx);

		__syncthreads();
		if (lane == WS - 1){
			d_array[warpId + blockIdx.x * TBS] = d_array[idx];
		}
		__syncthreads();

		// If there are more than WS ^ 2 then more than one warp is needed
		//To calculate bases
		if (warpId == 0){
			scan_warp(d_array, idx);
		}
		__syncthreads();

		if (warpId > 0){
			val = val + d_array[warpId + blockIdx.x * TBS - 1];
		}
		__syncthreads();

		// Save block base
		if ((idx % TBS) == TBS - 1) {
			d_base[blockIdx.x] = val + arrayVal;
		}

		d_array[idx] = val;
	}
}*/

Node* generateGraph(int nNodes, int maxEdgesPerNode) {
	srand((unsigned)time(0)); 
	Node* nodes = new Node[nNodes];
	for (int i = 0; i < nNodes; i++) {
		Node* tmp = new Node(i);
		nodes[i] = *tmp;
	}

	for (int i = 0; i < nNodes; i++) {
		int numEdges = rand() % (maxEdgesPerNode + 1);
		nodes[i].initializeChildren(numEdges);
		for (int j = 0; j < numEdges; j++) {
			int child = rand() % nNodes;
			bool isChild = false;
			for (int k = 0; k < nodes[i].getNumChildren(); k++){
				if (child == nodes[i].getChildren()[k].getValue()){
					isChild = true;
					break;
				}
			}
			if (!isChild && child != nodes[i].getValue()){
				nodes[i].addChild(nodes[child]);
			}
		}
	}
	
	for (int i = 0; i < nNodes; i++) {
		nodes[i].printNode();
	}

	return nodes; 
}

void exploreChild(Node child, vector< vector<Node> > path, int depth) {
	printf("Explore Child%i\n", child.getValue());
	child.setExplored(1);
	vector<Node> currentPath;
	if (path.size() <= depth) {
		path.push_back(currentPath);
	}
	currentPath = path[depth];
	for (int i = 0; i < child.getNumChildren(); i++) {
		if (child.getChildren()[i].getExplored() == 0) {
			currentPath.push_back(child.getChildren()[i]);
			exploreChild(child.getChildren()[i], path, depth + 1);	
		}
	}

	child.setExplored(2);
	return;
}

vector< vector<Node> > bfs(Node* nodes, int size) {
	vector< vector<Node> > path;

	Node currentNode = nodes[0];
	vector<Node> firstPath;
	firstPath.push_back(currentNode);
	path.push_back(firstPath);

	exploreChild(currentNode, path, 1);

	return path;
}

int main (int argc, char **argv) {

	// Get command line argument
	int size = atoi(argv[1]);
	int maxEdgesPerNode = atoi(argv[2]);

	Node* nodes = generateGraph(size, maxEdgesPerNode);
	bfs(nodes, size);

	
	/*int *d_array, *d_base, *d_size;

	// Allocate space for device copies
	cudaMalloc((void **)&d_array, size * sizeof(int));
	cudaMalloc((void **)&d_base, ceil(((float) size) / TBS) * sizeof(int));
	cudaMalloc((void **)&d_size, sizeof(int));
	

	// Copy inputs to device
	cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start;
	cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

	// MAIN COMPUTATION, SEQUENTIAL VERSION
	result[0] = 0;
	for (int i = 1; i < size; i++) {
		result[i] = result[i-1] + array[i-1];
	}

    int gridSz = ceil(((float) size) / TBS);
    // Record the start event
    cudaEventRecord(start, NULL);

	// Launch sparseMatrixMul() kernel on GPU
	allPrefixSums<<<gridSz, TBS>>>(d_array, d_size, d_base);
	
	// Make sure result is finished
	cudaDeviceSynchronize();
	
	calcBase(d_base, size, d_array);

	// Record end event
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    printf("GPU Time= %.3f msec\n", msecTotal);

	// Copy result back to host
	int *gpu_result = (int *) malloc(size * sizeof(int));
	cudaMemcpy(gpu_result, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

	bool isCorrect = true;
	for (int i = 0; i < size; i++) {
		//printf("%i GPU: %i CPU: %i arr: %i\n", i, gpu_result[i], result[i], array[i]);
		//Print the result if it is wrong
		if (result[i] != gpu_result[i]) {
			printf("%i GPU: %i CPU: %i\n", i, gpu_result[i], result[i]);
			isCorrect = false;
		}

		if(i % TBS == TBS - 1 || i % TBS == 0) {
			//printf("%i GPU: %i CPU: %i\n", i, gpu_result[i], result[i]);
		} 
	}

	if (!isCorrect) {
		printf("The results do not match\n");
	} else {
		printf("The results match\n");
	}

	// Cleanup
	cudaFree(d_array); 
	cudaFree(d_size); */

	return 0;
}

Node::Node(int newValue) {
	value = newValue;
	explored = 0;
}

Node::Node() {
}

int Node::getValue() {
	return value;
}

Node* Node::getChildren() {
	return children;
}

int Node::getNumChildren() {
	return numChildren;
}

void Node::addChild(Node child) {
	children[numChildren] = child;
	numChildren++;

	return;
}

void Node::printNode() {
	printf("Value: %i Children: [", value);
	for (int i = 0; i < numChildren; i++) {
		printf("%i", children[i].getValue());
		if (i != numChildren - 1) {
			printf(", ");
		}
	}
	printf("]\n");
	return;
}

void Node::initializeChildren(int numEdges) {
	children = new Node[numEdges];
}

int Node::getExplored() {
	return explored;
}

void Node::setExplored(int newExplored) {
	explored = newExplored;
	return;
}



