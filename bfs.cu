#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <queue>
#include <ctime>

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
	int* children;
	int numChildren;
	int explored;

public:
	Node();
	Node(int);
	__host__ __device__ int getValue();
	void addChild(Node*);
	__host__ __device__ int* getChildren();
	__host__ __device__ int getNumChildren();
	void printNode();
	void initializeChildren(int);
	__host__ __device__ int getExplored();
	void setExplored(int);
	__device__ int parallelSetExplored(int);
};

__global__ void parentListBackwardsWave(int *d_waveMask, int *d_nextWaveMask, int *d_parent, int *d_parentPtr, int *d_cost, int *d_size) {
	int idx = blockIdx.x * TBS + threadIdx.x;

	if (idx < *d_size && d_waveMask[idx] == 0) {
		// Loop through all children
		for (int i = d_parentPtr[idx]; i < d_parentPtr[idx + 1]; i++) {
			if (d_waveMask[d_parent[i]] == 1) {
				atomicCAS(&d_nextWaveMask[idx], 0, 1);
				d_cost[idx] = d_cost[d_parent[i]] + 1;
				break;
			}
		}
	}
	if(idx < *d_size && d_waveMask[idx] == 2){
		d_nextWaveMask[idx] = 2;
	}
}

__global__ void backwardsWave(int *d_waveMask, int *d_nextWaveMask, int *d_children, int *d_numChildren, int *d_cost, int *d_size, int *d_maxChildren) {
	int idx = blockIdx.x * TBS + threadIdx.x;

	if (idx < *d_size && d_waveMask[idx] == 0) {
		// Loop through all children
		for (int i = 0; i < *d_size * *d_maxChildren; i++) {
			if (d_children[i] == idx) {
				int parent = i / *d_maxChildren;
				if (d_waveMask[parent] == 1) {
					atomicCAS(&d_nextWaveMask[idx], 0, 1);
					d_cost[idx] = d_cost[parent] + 1;
					break;
				}
			}
		}
	}
	if(idx < *d_size && d_waveMask[idx] == 2){
		d_nextWaveMask[idx] = 2;
	}
}

__global__ void childListExploreWave(int *d_waveMask, int *d_nextWaveMask, int *d_children, int *d_numChildren, int *d_cost, int *d_size, int *d_maxChildren) {
	int idx = blockIdx.x * TBS + threadIdx.x;

	if (idx < *d_size && d_waveMask[idx] == 1) {
		int numChildren = d_numChildren[idx];
		
		for (int i = 0; i < numChildren; i++) {
			int child = d_children[idx * *d_maxChildren + i];
			
			atomicCAS(&d_nextWaveMask[child],0,1);
					
			if (d_waveMask[child] == 0) {
				d_cost[child] = d_cost[idx] + 1;
			}
		}
	}
	if(idx < *d_size && d_waveMask[idx] == 2){
		d_nextWaveMask[idx] = 2;
	}
}

__global__ void exploreWave(int *d_waveMask, int *d_nextWaveMask, Node *d_graph, int *d_children, int *d_cost, int *d_size, int *d_maxChildren) {
	int idx = blockIdx.x * TBS + threadIdx.x;

	if (idx < *d_size && d_waveMask[idx] == 1) {
		Node currentNode = d_graph[idx];
		int numChildren = currentNode.getNumChildren();
		
		for (int i = 0; i < numChildren; i++) {
			int child = d_children[idx * *d_maxChildren + i];
			
			atomicCAS(&d_nextWaveMask[child],0,1);
					
			if (d_waveMask[child] == 0) {
				d_cost[child] = d_cost[idx] + 1;
			}
		}
	}
	if(idx < *d_size && d_waveMask[idx] == 2){
		d_nextWaveMask[idx] = 2;
	}
}

__global__ void setPreviousExplored(int *d_waveMask, int *d_nextWaveMask, int *d_size){
	int idx = blockIdx.x * TBS + threadIdx.x;

	if(idx < *d_size){
		if(d_waveMask[idx] == 1){
			d_nextWaveMask[idx] = 2;
		}
	}
}

int* generateChildren(Node *nodes, int nNodes, int maxEdgesPerNode) {
	int* children = new int[nNodes * maxEdgesPerNode];

	for (int i = 0; i < nNodes; i++) {
		int numEdges = (rand() % maxEdgesPerNode) + 1;
		nodes[i].initializeChildren(numEdges);
		for (int j = 0; j < numEdges; j++) {
			int child = rand() % nNodes;
			bool isChild = false;
			for (int k = 0; k < nodes[i].getNumChildren(); k++){
				if (child == nodes[i].getChildren()[k]){
					isChild = true;
					break;
				}
			}
			if (!isChild && child != nodes[i].getValue()){
				children[i * maxEdgesPerNode + nodes[i].getNumChildren()] = child;
				nodes[i].addChild(&nodes[child]);
			}
		}
	}
	/*for (int i = 0; i < nNodes; i++) {
		nodes[i].printNode();
	}*/

	return children;
}

Node* generateGraph(int nNodes) {
	srand((unsigned)time(0)); 
	Node* nodes = new Node[nNodes];
	
	for (int i = 0; i < nNodes; i++) {
		Node* tmp = new Node(i);
		nodes[i] = *tmp;
	}

	return nodes; 
}

void exploreChild(Node* child, vector< vector<Node*> >* path, int depth, Node* nodes) {
	int numChildren = child->getNumChildren();
	if (numChildren > 0) {
		bool *toExplore = new bool[numChildren];
		vector<Node*> newPath;
		if (path->size() <= depth) {
			path->push_back(newPath);
		}
		vector<Node*>* currentPath = &(path->at(depth));
		
		for (int i = 0; i < numChildren; i++) {
			Node* newChild = &nodes[child->getChildren()[i]];
			if (newChild->getExplored() == 0) {
				currentPath->push_back(newChild);
				newChild->setExplored(1);
				toExplore[i] = true;
			} else {
				toExplore[i] = false;
			}
		}

		// Explore loop after push loop so it is actually BFS
		for (int i = 0; i < numChildren; i++) {
			Node* newChild = &nodes[child->getChildren()[i]];
			if (toExplore[i]) {
				exploreChild(newChild, path, depth + 1, nodes);	
			}
		}
	}

	child->setExplored(2);
	return;
}

int* bfs(Node* nodes, int size) {
	int* cost = new int[size];
	for (int i = 0; i < size; i++) {
		cost[i] = -1;
	}

	Node* currentNode = &nodes[0];
	queue<Node*> wave;
	wave.push(currentNode);
	cost[0] = 0;

	int depth = 0;
	while (!wave.empty()) {
		depth = cost[wave.front()->getValue()];
		while (!wave.empty() && depth == cost[wave.front()->getValue()]) {
			currentNode = wave.front();
			wave.pop();
			currentNode->setExplored(1);
			if (currentNode->getNumChildren() > 0) {
				int *children = currentNode->getChildren();
				for (int i = 0; i < currentNode->getNumChildren(); i++) {
					if (nodes[children[i]].getExplored() == 0) {
						nodes[children[i]].setExplored(1);
						cost[children[i]] = depth + 1;
						wave.push(&nodes[children[i]]);
					}
				}
			}
		}
	}

	return cost;
}

int* transformBfs(vector< vector<Node*> > path, int size) {
	int *result = new int[size];
	for (int i = 0; i < size; i++) {
		result[i] = -1;
	}
	for (int i = 0; i < path.size(); i++) {
		//printf("%i - ", i);
		for (int j = 0; j < path[i].size(); j++) {
			//printf(" %i ", path[i][j]->getValue());
			result[path[i][j]->getValue()] = i;
		}
		//printf("\n");
	}
	return result;
}

int* transformNumChildren(Node* nodes, int size) {
	int *result = new int[size];
	for (int i = 0; i < size; i++) {
		result[i] = nodes[i].getNumChildren();
	}
	return result;
}

int* transformParentPtr(Node* nodes, int size) {
	int *result = new int[size + 1];
	for (int i = 0; i < size; i++) {
		result[i] = 0;
	}

	for (int i = 0; i < size; i++) {
		Node *node = &nodes[i];
		if (node->getNumChildren() > 0) {
			int *children = node->getChildren();
			for (int j = 0; j < node->getNumChildren(); j++) {
				int child = children[j];
				result[child + 1] += 1;
			}
		}
	}
	
	for (int i = 1; i < size + 1; i++) {
		result[i] = result[i] + result[i - 1];
	}
	return result;
}

int* transformParents(Node* nodes, int size, int* parentPtr) {
	int numEdges = parentPtr[size];
	int *result = new int[numEdges];
	int *curIdx = new int[size];
	for (int i = 0; i < size; i++) {
		curIdx[i] = parentPtr[i];
	}

	for (int i = 0; i < size; i++) {
		Node *node = &nodes[i];
		if (node->getNumChildren() > 0) {
			int *children = node->getChildren();
			for (int j = 0; j < node->getNumChildren(); j++) {
				int child = children[j];
				result[curIdx[child]] = i;
				curIdx[child] = curIdx[child] + 1;
			}
		}
	}

	return result;
}

void callFlipFlopParent(int *d_size, int *d_children, int *d_numChildren, int *d_maxChildren, int *d_parent, int *d_parentPtr, int size, int maxChildren, int *synchResult) {
	cudaEvent_t start;
	cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    int *d_cost, *d_waveMask, *d_nextWaveMask;

	// Allocate space for device copies
	cudaMalloc((void **)&d_cost, size * sizeof(int));
	cudaMalloc((void **)&d_waveMask, size * sizeof(int));
	cudaMalloc((void **)&d_nextWaveMask, size * sizeof(int));


    int gridSz = ceil(((float) size) / TBS);

    int *waveMask = new int[size];
    int *nextWaveMask = new int[size]; 

    int *cost = new int[size];
    cost[0] = 0;
    for (int i = 1; i < size; i++) {
    	cost[i] = -1;
    	waveMask[i] = 0;
		nextWaveMask[i] = 0;
    }

    waveMask[0] = 1;

    cudaMemcpy(d_cost, cost, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_waveMask, waveMask, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nextWaveMask, nextWaveMask, size * sizeof(int), cudaMemcpyHostToDevice);

	// Record the start event
    cudaEventRecord(start, NULL);
    
    bool complete = false;
    int completed = 0;
    while(!complete) {
    	// Launch kernel on GPU
    	if (completed < (maxChildren * maxChildren - 1) / (maxChildren * maxChildren) * size) {
    		childListExploreWave<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_children, d_numChildren, d_cost, d_size, d_maxChildren);
    	} else {
    		parentListBackwardsWave<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_parent, d_parentPtr, d_cost, d_size);
    	}
		
		cudaDeviceSynchronize();
		setPreviousExplored<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_size);		
		cudaDeviceSynchronize();
		cudaMemcpy(d_waveMask, d_nextWaveMask, size * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_nextWaveMask, nextWaveMask, size * sizeof(int), cudaMemcpyHostToDevice);

		complete = true;
		cudaMemcpy(waveMask, d_waveMask, size * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0 ; i < size; i++) {
			if(waveMask[i] == 1) {
				complete = false;
			} else if (waveMask[i] == 2) {
				completed += 1;
			}
		}
    }

	
	
	// Make sure result is finished
	cudaDeviceSynchronize();

	// Record end event
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    printf("GPU Parent Flip Flop Explore Time= %.3f msec\n", msecTotal);

	// Copy result back to host
	int *gpu_result = (int *) malloc(size * sizeof(int));
	cudaMemcpy(gpu_result, d_cost, size * sizeof(int), cudaMemcpyDeviceToHost);

	bool isCorrect = true;

	for (int i = 0; i < size; i++) {
		if (synchResult[i] != gpu_result[i]) {
			isCorrect = false;
			printf("%i CPU: %i GPU:%i\n", i, synchResult[i], gpu_result[i]);
		}
	}

	if (!isCorrect) {
		printf("The results do not match\n");
	} else {
		printf("The results match\n");
	}
}


void callFlipFlopWaveExplore(int *d_size, int *d_children, int *d_numChildren, int size, int *d_maxChildren, int maxChildren, int *synchResult) {
	cudaEvent_t start;
	cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    int *d_cost, *d_waveMask, *d_nextWaveMask;

	// Allocate space for device copies
	cudaMalloc((void **)&d_cost, size * sizeof(int));
	cudaMalloc((void **)&d_waveMask, size * sizeof(int));
	cudaMalloc((void **)&d_nextWaveMask, size * sizeof(int));


    int gridSz = ceil(((float) size) / TBS);

    int *waveMask = new int[size];
    int *nextWaveMask = new int[size]; 

    int *cost = new int[size];
    cost[0] = 0;
    for (int i = 1; i < size; i++) {
    	cost[i] = -1;
    	waveMask[i] = 0;
		nextWaveMask[i] = 0;
    }

    waveMask[0] = 1;

    cudaMemcpy(d_cost, cost, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_waveMask, waveMask, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nextWaveMask, nextWaveMask, size * sizeof(int), cudaMemcpyHostToDevice);

	// Record the start event
    cudaEventRecord(start, NULL);
    
    bool complete = false;
    int completed = 0;
    while(!complete) {
    	// Launch kernel on GPU
    	if (completed < (maxChildren * maxChildren - 1) / (maxChildren * maxChildren) * size) {
    		childListExploreWave<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_children, d_numChildren, d_cost, d_size, d_maxChildren);
    	} else {
    		backwardsWave<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_children, d_numChildren, d_cost, d_size, d_maxChildren);
    	}
		
		cudaDeviceSynchronize();
		setPreviousExplored<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_size);		
		cudaDeviceSynchronize();
		cudaMemcpy(d_waveMask, d_nextWaveMask, size * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_nextWaveMask, nextWaveMask, size * sizeof(int), cudaMemcpyHostToDevice);

		complete = true;
		cudaMemcpy(waveMask, d_waveMask, size * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0 ; i < size; i++) {
			if(waveMask[i] == 1) {
				complete = false;
			} else if (waveMask[i] == 2) {
				completed += 1;
			}
		}
    }

	
	
	// Make sure result is finished
	cudaDeviceSynchronize();

	// Record end event
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    printf("GPU Flip Flop Explore Time= %.3f msec\n", msecTotal);

	// Copy result back to host
	int *gpu_result = (int *) malloc(size * sizeof(int));
	cudaMemcpy(gpu_result, d_cost, size * sizeof(int), cudaMemcpyDeviceToHost);

	bool isCorrect = true;

	for (int i = 0; i < size; i++) {
		if (synchResult[i] != gpu_result[i]) {
			isCorrect = false;
			printf("%i CPU: %i GPU:%i\n", i, synchResult[i], gpu_result[i]);
		}
	}

	if (!isCorrect) {
		printf("The results do not match\n");
	} else {
		printf("The results match\n");
	}
}

void callChildListExploreWave(int *d_size, int *d_children, int *d_numChildren, int size, int *d_maxChildren, int *synchResult) {
	cudaEvent_t start;
	cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    int *d_cost, *d_waveMask, *d_nextWaveMask;

	// Allocate space for device copies
	cudaMalloc((void **)&d_cost, size * sizeof(int));
	cudaMalloc((void **)&d_waveMask, size * sizeof(int));
	cudaMalloc((void **)&d_nextWaveMask, size * sizeof(int));


    int gridSz = ceil(((float) size) / TBS);

    int *waveMask = new int[size];
    int *nextWaveMask = new int[size]; 

    int *cost = new int[size];
    cost[0] = 0;
    for (int i = 1; i < size; i++) {
    	cost[i] = -1;
    	waveMask[i] = 0;
		nextWaveMask[i] = 0;
    }

    waveMask[0] = 1;

    cudaMemcpy(d_cost, cost, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_waveMask, waveMask, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nextWaveMask, nextWaveMask, size * sizeof(int), cudaMemcpyHostToDevice);
    
	// Record the start event
    cudaEventRecord(start, NULL);

    bool complete = false;
    while(!complete) {

    	// Launch kernel on GPU
		childListExploreWave<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_children, d_numChildren, d_cost, d_size, d_maxChildren);
		cudaDeviceSynchronize();
		setPreviousExplored<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_size);		
		cudaDeviceSynchronize();
		cudaMemcpy(d_waveMask, d_nextWaveMask, size * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_nextWaveMask, nextWaveMask, size * sizeof(int), cudaMemcpyHostToDevice);

		complete = true;
		cudaMemcpy(waveMask, d_waveMask, size * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0 ; i < size; i++){
			if(waveMask[i] == 1){
				complete = false;
			}
		}
    }

	
	
	// Make sure result is finished
	cudaDeviceSynchronize();

	// Record end event
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    printf("GPU Child List Explore Time= %.3f msec\n", msecTotal);

	// Copy result back to host
	int *gpu_result = (int *) malloc(size * sizeof(int));
	cudaMemcpy(gpu_result, d_cost, size * sizeof(int), cudaMemcpyDeviceToHost);

	bool isCorrect = true;

	for (int i = 0; i < size; i++) {
		if (synchResult[i] != gpu_result[i]) {
			isCorrect = false;
			printf("%i CPU: %i GPU:%i\n", i, synchResult[i], gpu_result[i]);
		}
	}

	if (!isCorrect) {
		printf("The results do not match\n");
	} else {
		printf("The results match\n");
	}
}

void callDeviceCachedVisitBFS(Node *d_graph, int *d_size, int *d_children, int size, int *d_maxChildren, int *synchResult) {
	cudaEvent_t start;
	cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    int *d_cost, *d_waveMask, *d_nextWaveMask;

	// Allocate space for device copies
	cudaMalloc((void **)&d_cost, size * sizeof(int));
	cudaMalloc((void **)&d_waveMask, size * sizeof(int));
	cudaMalloc((void **)&d_nextWaveMask, size * sizeof(int));


    int gridSz = ceil(((float) size) / TBS);

    int *waveMask = new int[size];
    int *nextWaveMask = new int[size]; 

    int *cost = new int[size];
    cost[0] = 0;
    for (int i = 1; i < size; i++) {
    	cost[i] = -1;
    	waveMask[i] = 0;
	nextWaveMask[i] = 0;
    }

    waveMask[0] = 1;

    cudaMemcpy(d_cost, cost, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_waveMask, waveMask, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_nextWaveMask, nextWaveMask, size * sizeof(int), cudaMemcpyHostToDevice);
    
	// Record the start event
    cudaEventRecord(start, NULL);

    bool complete = false;
    while(!complete) {

    	// Launch kernel on GPU
		exploreWave<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_graph, d_children, d_cost, d_size, d_maxChildren);
		cudaDeviceSynchronize();
		setPreviousExplored<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_size);		
		cudaDeviceSynchronize();
		cudaMemcpy(d_waveMask, d_nextWaveMask, size * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_nextWaveMask, nextWaveMask, size * sizeof(int), cudaMemcpyHostToDevice);

		//exploreWave<<<gridSz, TBS>>>(d_waveMask, d_nextWaveMask, d_graph, d_children, d_cost, d_size, d_maxChildren);
		complete = true;
		cudaMemcpy(waveMask, d_waveMask, size * sizeof(int), cudaMemcpyDeviceToHost);
		for(int i = 0 ; i < size; i++){
			if(waveMask[i] == 1){
				complete = false;
			}
		}
    }

	
	
	// Make sure result is finished
	cudaDeviceSynchronize();

	// Record end event
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    printf("GPU Wave Time= %.3f msec\n", msecTotal);

	// Copy result back to host
	int *gpu_result = (int *) malloc(size * sizeof(int));
	cudaMemcpy(gpu_result, d_cost, size * sizeof(int), cudaMemcpyDeviceToHost);

	bool isCorrect = true;

	for (int i = 0; i < size; i++) {
		if (synchResult[i] != gpu_result[i]) {
			isCorrect = false;
			printf("%i CPU: %i GPU:%i\n", i, synchResult[i], gpu_result[i]);
		}
	}

	if (!isCorrect) {
		printf("The results do not match\n");
	} else {
		printf("The results match\n");
	}
}

int main (int argc, char **argv) {
	if (argc !=3) {
		printf("\nToo few arguments!\n");
		abort();
	}

	// Get command line argument
	int size = atoi(argv[1]);
	int maxEdgesPerNode = atoi(argv[2]);

	Node* nodes = generateGraph(size);
	int* children = generateChildren(nodes, size, maxEdgesPerNode);
	int* numChildren = transformNumChildren(nodes, size);
	int* parentPtr = transformParentPtr(nodes, size);
	int numEdges = parentPtr[size];
	int* parent = transformParents(nodes, size, parentPtr);

	/*for (int i = 0; i < size + 1; i++) {
		printf("%i parentPtr: %i\n", i, parentPtr[i]);
	}
	
	for (int i = 0; i < size; i++) {
		for (int j = parentPtr[i]; j < parentPtr[i + 1]; j++) {
			printf("%i child: %i parent: %i\n", j, i, parent[j]);
		}
	}*/

	Node* d_graph;
	int *d_children, *d_size, *d_maxChildren, *d_numChildren, *d_parent, *d_parentPtr;

	// Allocate space for device copies
	cudaMalloc((void **)&d_graph, size * sizeof(Node));
	cudaMalloc((void **)&d_size, sizeof(int));
	cudaMalloc((void **)&d_maxChildren, sizeof(int));
	cudaMalloc((void **)&d_children, size * maxEdgesPerNode * sizeof(int));
	cudaMalloc((void **)&d_numChildren, size * sizeof(int));
	cudaMalloc((void **)&d_parentPtr, (size + 1) * sizeof(int));
	cudaMalloc((void **)&d_parent, numEdges * sizeof(int));

	// Copy inputs to device
	cudaMemcpy(d_graph, nodes, size * sizeof(Node), cudaMemcpyHostToDevice);
	cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_maxChildren, &maxEdgesPerNode, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_children, children, size * maxEdgesPerNode * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_numChildren, numChildren, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parentPtr, parentPtr, (size + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_parent, parent, numEdges * sizeof(int), cudaMemcpyHostToDevice);

	//Synchronouse bfs
	//vector< vector<Node*> > path = bfs(nodes, size);
	clock_t start;
	clock_t end;
	start = clock();
	int *synchResult = bfs(nodes, size);
	end = clock();

	printf("CPU Time= %.3f msec\n", (end - start) / (double) (CLOCKS_PER_SEC / 1000));

	callDeviceCachedVisitBFS(d_graph, d_size, d_children, size, d_maxChildren, synchResult);

	callChildListExploreWave(d_size, d_children, d_numChildren, size, d_maxChildren, synchResult);

	//callFlipFlopWaveExplore(d_size, d_children, d_numChildren, size, d_maxChildren, maxEdgesPerNode, synchResult);

	callFlipFlopParent(d_size, d_children, d_numChildren, d_maxChildren, d_parent, d_parentPtr, size, maxEdgesPerNode, synchResult);

	// Cleanup
	cudaFree(d_graph); 
	cudaFree(d_size);
	cudaFree(d_children);
	cudaFree(d_numChildren);
	cudaFree(d_maxChildren);

	return 0;
}

Node::Node(int newValue) {
	value = newValue;
	explored = 0;
}

Node::Node() {
}

__host__ __device__ int Node::getValue() {
	return value;
}

__host__ __device__ int* Node::getChildren() {
	return children;
}

__host__ __device__ int Node::getNumChildren() {
	return numChildren;
}

void Node::addChild(Node* child) {
	children[numChildren] = child->getValue();
	numChildren++;

	return;
}

void Node::printNode() {
	printf("Value: %i Children: [", value);
	for (int i = 0; i < numChildren; i++) {
		printf("%i", children[i]);
		if (i != numChildren - 1) {
			printf(", ");
		}
	}
	printf("]\n");
	return;
}

void Node::initializeChildren(int numEdges) {
	children = new int[numEdges];
}

__host__ __device__ int Node::getExplored() {
	return explored;
}

__device__ int Node::parallelSetExplored(int newExplored) {
	return atomicExch(&explored, newExplored);
}

void Node::setExplored(int newExplored) {
	explored = newExplored;
	return;
}



