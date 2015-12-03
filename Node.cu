#include "Node.h"

Node::Node(int value) {
	this.value = value;
	return this;
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
		printf("child: %i is %i, ", i, children[i].getValue());
	}
	printf("]\n");
	return;
}

