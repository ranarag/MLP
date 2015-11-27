#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>
#include "nodes.cpp"
using namespace std;
class LAYER{
private:
    double Net;
public:
    double * Input;
    NODE *Node;
    int NodeLength;
    int InputLength;
    void FeedForward();
    double Sigmoid(double Net);
    double *OutputVector();
    LAYER(int NoOfNodes,int NoOfInputs);
    NODE * getNodes();

};
LAYER::LAYER(int NoOfNodes,int NoOfInputs){
    Node=(NODE *)malloc(NoOfNodes*sizeof(NODE));
    for(int i=0;i<NoOfNodes;i++){
        Node[i]=NODE(NoOfInputs);
    }
    Input = new double[NoOfInputs];
    NodeLength = NoOfNodes;
    InputLength = NoOfInputs;
}
NODE * LAYER::getNodes(){

return Node;
}
double LAYER::Sigmoid(double Nit){
return 1.0/(1.0+exp(-Nit));

}
void LAYER::FeedForward(){
for(int i=0;i<NodeLength;i++){
    Net = Node[i].Threshold;
    for(int j=0;j<Node[i].WeightSize;j++){
        Net += Node[i].Weight[j]*Input[j];
    }
    Node[i].Output = Sigmoid(Net);
}

}
double* LAYER::OutputVector(){
    double * v = new double[NodeLength];
    for(int i=0;i<NodeLength;i++)
        v[i]=Node[i].Output;
    return v;
}

