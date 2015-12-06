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

__global__ void exploreWave(int *d_currentWave, Node *d_graph, int *d_waveSize, int *d_cost, int *d_size) {
	int idx = blockIdx.x * TBS + threadIdx.x;

	if (idx < *d_waveSize) {
		printf("%i hey\n", idx);
		Node currentNode = d_graph[d_currentWave[idx]];
		int* children = currentNode.getChildren();
		int numChildren = currentNode.getNumChildren();
		printf("numChild: %i\n\n\n\n\n", numChildren);
		for (int i = 0; i < numChildren; i++) {
			//printf("child: %i\n", children[0]->getValue());
			if (d_graph[children[i]].getExplored() == 0) {
				d_cost[children[i]] = d_cost[currentNode.getValue()] + 1;
				d_graph[children[i]].parallelSetExplored(1);	
			}
		}
		
	}
	
}


//Multiplies each element of sparse matrix by the correct vector element and puts the result back in the matrix
/*__global__ void cachedVisitBFS(Node *d_graph, int *d_size) {
	int idx = blockIdx.x * TBS + threadIdx.x;
	if (idx == 0) {
		int waveSize = 1;
		int *currentWave = new int[waveSize];
		currentWave[0] = d_graph[0].getValue();

		int *nextWave = exploreWave(currentWave, d_graph, waveSize);
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
				nodes[i].addChild(&nodes[child]);
			}
		}
	}
	
	for (int i = 0; i < nNodes; i++) {
		nodes[i].printNode();
	}

	return nodes; 
}

void exploreChild(Node* child, vector< vector<Node*> >* path, int depth, Node* nodes) {
	//printf("Explore Child%i Depth: %i\n", child->getValue(), depth);
	child->setExplored(1);
	if (child->getNumChildren() > 0) {
		vector<Node*> newPath;
		if (path->size() <= depth) {
			path->push_back(newPath);
		}
		vector<Node*>* currentPath = &(path->at(depth));
		//printf("%i numChildren: %i\n", child->getValue(), child->getNumChildren());
		//child->printNode();
		for (int i = 0; i < child->getNumChildren(); i++) {
			Node* newChild = &nodes[child->getChildren()[i]];
			if (newChild->getExplored() == 0) {
				currentPath->push_back(newChild);
			}
		}

		// Explore loop after push loop so it is actually BFS
		for (int i = 0; i < child->getNumChildren(); i++) {
			Node* newChild = &nodes[child->getChildren()[i]];
			if (newChild->getExplored() == 0) {
				exploreChild(newChild, path, depth + 1, nodes);	
			}
		}
	}

	child->setExplored(2);
	return;
}

vector< vector<Node*> > bfs(Node* nodes, int size) {
	vector< vector<Node*> > path;

	Node* currentNode = &nodes[0];
	vector<Node*> firstPath;
	firstPath.push_back(currentNode);
	path.push_back(firstPath);

	exploreChild(currentNode, &path, 1, nodes);

	return path;
}

void callDeviceCachedVisitBFS(Node *d_graph, int *d_size, int size, vector< vector<Node*> > path) {
	cudaEvent_t start;
	cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);

    int *d_currentWave, *d_waveSize, *d_cost;

	// Allocate space for device copies
	cudaMalloc((void **)&d_currentWave, size * sizeof(int));
	cudaMalloc((void **)&d_waveSize, sizeof(int));
	cudaMalloc((void **)&d_cost, size * sizeof(int));


    int gridSz = ceil(((float) size) / TBS);
    // Record the start event
    cudaEventRecord(start, NULL);

    int waveSize = 1;
    int *currentWave = new int[waveSize];
    currentWave[0] = 0;

    int *cost = new int[size];
    cost[0] = 0;
    for (int i = 1; i < size; i++) {
    	cost[i] = -1;
    }

    cudaMemcpy(d_cost, cost, size * sizeof(int), cudaMemcpyHostToDevice);

    bool complete = false;
    while(!complete) {
    	// Copy inputs to device
		cudaMemcpy(d_waveSize, &waveSize, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_currentWave, currentWave, waveSize * sizeof(int), cudaMemcpyHostToDevice);

    	// Launch kernel on GPU
		exploreWave<<<gridSz, TBS>>>(d_currentWave, d_graph, d_waveSize, d_cost, d_size);

		complete = true;
    }

	
	
	// Make sure result is finished
	cudaDeviceSynchronize();

	// Record end event
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    printf("GPU Time= %.3f msec\n", msecTotal);

	// Copy result back to host
	int *gpu_result = (int *) malloc(size * sizeof(int));
	cudaMemcpy(gpu_result, d_cost, size * sizeof(int), cudaMemcpyDeviceToHost);

	bool isCorrect = true;

	for (int j = 0; j < size; j++) {
		printf("%i cost: %i\n", j, gpu_result[j]);
	}


	for (int i = 0; i < path.size(); i++) {
		printf("%i - ", i);
		for (int j = 0; j < path[i].size(); j++) {
			printf(" %i ", path[i][j]->getValue());
		}
		printf("\n");
	}

	if (!isCorrect) {
		printf("The results do not match\n");
	} else {
		printf("The results match\n");
	}

	/*for (int i = 0; i < size; i++) {
		//printf("%i GPU: %i CPU: %i arr: %i\n", i, gpu_result[i], result[i], array[i]);
		//Print the result if it is wrong
		if (result[i] != gpu_result[i]) {
			printf("%i GPU: %i CPU: %i\n", i, gpu_result[i], result[i]);
			isCorrect = false;
		}

		if(i % TBS == TBS - 1 || i % TBS == 0) {
			//printf("%i GPU: %i CPU: %i\n", i, gpu_result[i], result[i]);
		} 
	}*/
}

int main (int argc, char **argv) {
	if (argc !=3) {
		printf("\nToo few arguments!\n");
		abort();
	}

	// Get command line argument
	int size = atoi(argv[1]);
	int maxEdgesPerNode = atoi(argv[2]);

	Node* nodes = generateGraph(size, maxEdgesPerNode);

	Node* d_graph;
	int* d_size;

	// Allocate space for device copies
	cudaMalloc((void **)&d_graph, size * sizeof(Node));
	cudaMalloc((void **)&d_size, sizeof(int));

	// Copy inputs to device
	cudaMemcpy(d_graph, nodes, size * sizeof(Node), cudaMemcpyHostToDevice);
	cudaMemcpy(d_size, &size, sizeof(int), cudaMemcpyHostToDevice);

	//Synchronouse bfs
	vector< vector<Node*> > path = bfs(nodes, size);

	callDeviceCachedVisitBFS(d_graph, d_size, size, path);

	// Cleanup
	cudaFree(d_graph); 
	cudaFree(d_size); 

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



