class Node {
private:
	int value;
	Node* children;
	int numChildren;

public:
	Node();
	Node(int);
	int getValue();
	void addChild(Node);
	Node* getChildren();
	int getNumChildren();
	void printNode();
};
