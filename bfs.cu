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
	Node** children;
	int numChildren;
	int explored;

public:
	Node();
	Node(int);
	int getValue();
	void addChild(Node*);
	Node** getChildren();
	int getNumChildren();
	void printNode();
	void initializeChildren(int);
	int getExplored();
	void setExplored(int);
	__device__ int parallelSetExplored(int);
};

__device__ int* exploreWave(int *currentWave, Node* d_graph, int waveSize) {
	int idx = blockIdx.x * TBS + threadIdx.x;
	if (idx < waveSize) {
		printf("%i\n", idx);
		return new int[1];
	}
	return null;
}


//Multiplies each element of sparse matrix by the correct vector element and puts the result back in the matrix
__global__ void cachedVisitBFS(Node *d_graph, int *d_size) {
	int idx = blockIdx.x * TBS + threadIdx.x;
	if (idx == 0) {
		int waveSize = 1;
		int *currentWave = new int[waveSize];
		currentWave[0] = d_graph[0].getValue();

		int *nextWave = exploreWave(currentWave, d_graph, waveSize);
	}
}

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
				if (child == nodes[i].getChildren()[k]->getValue()){
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

void exploreChild(Node* child, vector< vector<Node*> >* path, int depth) {
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
			Node* newChild = child->getChildren()[i];
			if (newChild->getExplored() == 0) {
				currentPath->push_back(newChild);
			}
		}

		// Explore loop after push loop so it is actually BFS
		for (int i = 0; i < child->getNumChildren(); i++) {
			Node* newChild = child->getChildren()[i];
			if (newChild->getExplored() == 0) {
				exploreChild(newChild, path, depth + 1);	
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

	exploreChild(currentNode, &path, 1);

	return path;
}

void callDeviceCachedVisitBFS(Node *d_graph, int *d_size, int size, vector< vector<Node*> > path) {
	cudaEvent_t start;
	cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);


    int gridSz = ceil(((float) size) / TBS);
    // Record the start event
    cudaEventRecord(start, NULL);

	// Launch sparseMatrixMul() kernel on GPU
	cachedVisitBFS<<<gridSz, TBS>>>(d_graph, d_size);
	
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
	cudaMemcpy(gpu_result, d_graph, size * sizeof(int), cudaMemcpyDeviceToHost);

	bool isCorrect = true;


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

int Node::getValue() {
	return value;
}

Node** Node::getChildren() {
	return children;
}

int Node::getNumChildren() {
	return numChildren;
}

void Node::addChild(Node* child) {
	children[numChildren] = child;
	numChildren++;

	return;
}

void Node::printNode() {
	printf("Value: %i Children: [", value);
	for (int i = 0; i < numChildren; i++) {
		printf("%i", children[i]->getValue());
		if (i != numChildren - 1) {
			printf(", ");
		}
	}
	printf("]\n");
	return;
}

void Node::initializeChildren(int numEdges) {
	children = new Node*[numEdges];
}

int Node::getExplored() {
	return explored;
}

__device__ int Node::parallelSetExplored(int newExplored) {
	return atomicExch(&explored, newExplored);
}

void Node::setExplored(int newExplored) {
	explored = newExplored;
	return;
}



