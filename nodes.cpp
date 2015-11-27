#include<iostream>
#include<vector>
#include<cstdlib>
using namespace std;
class NODE{
private:
    void InitializeWeights(){
        Threshold = -1+2.0*((double)rand()/(double)RAND_MAX);
        cout << Threshold << endl;
        ThresholdDiff = 0;

        for(int i=0;i<WeightSize;i++){


            Weight[i] = -1+2.0*((double)rand()/(double)RAND_MAX);
            cout << Weight[i] << endl;
            WeightDiff[i]=0;}

    }
public:
    double Output;
    double *Weight;
    double *WeightDiff;
    double Threshold;
    double ThresholdDiff;
    double Delta;
    //NODE(int NoOfNodes);
    //double * get_weights();
   // double get_output();
    int WeightSize;
    NODE(int NoOfNodes){
    Weight = new double[NoOfNodes];
    WeightDiff = new double[NoOfNodes];
    WeightSize=NoOfNodes;
    InitializeWeights();
}
double* get_weights(){
    return Weight;}
double get_output(){
    return Output;
}
};




