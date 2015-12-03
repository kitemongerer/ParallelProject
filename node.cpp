#include "node.h"

Node::Node(int value) {
	this.value = value;
}

int Node::getValue() {
	return this.value;
}

Node* Node::getChildren() {
	return this.children;
}

int Node::getNumChildren() {
	return this.numChildren;
}

void Node::addChild(Node child) {
	children[numChildren] = child;
	numChildren++;

	return;
}

void Node::printNode() {
	printf("Value: %i Children: [", value);
	for (int i = 0; i < numChildren; i++) {
		printf("child: %i is %i, ", i, children[i].getValue());
	}
	printf("]\n");
	return;
}
