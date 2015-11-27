#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>

#include "layer.cpp"
using namespace std;
class BACKPROP{
private:
    double OverAllError;
    double MinimumError;
    double **ExpectedOutput;
    double **Input;
    double LearningRate;
    double Momentum;
    int NumberOfLayers;
    int NumberOfSamples;
    int SampleNumber;
    long int MaximumNumberOfIterations;
    void CalculateDelta();
    void BackPropError();
    void CalculateOverallError();
public:
    LAYER *Layer;
    double ** ActualOutput;
    void FeedForward();
    void UpdateWeights();
    BACKPROP(int *NumberOfNodes,double **InputSamples,double **OutputSamples,double LearnRate,double Moment,double MinError,long int MaxIter,int nos,int nol);
    void TrainNetwork();
    double Test(double input[]);
    double get_Error(){
        CalculateOverallError();
        return OverAllError;
    }

    double dsigmoid(double x){

    return x*(1.0 - x);}


};
void BACKPROP::FeedForward(){
    int i,j;
    for (i = 0; i < Layer[0].NodeLength; i++)
			Layer[0].Node[i].Output = Layer[0].Input[i];
            //Layer[0].FeedForward();
   /* double *xxx=Layer[0].OutputVector();
    Layer[1].Input=new double[Layer[0].NodeLength];
    for(i=0;i<Layer[0].NodeLength;i++)
    {
        Layer[1].Input[i]=xxx[i];
    }*/
    Layer[1].Input = Layer[0].OutputVector();
		for (i = 1; i < NumberOfLayers; i++) {
			Layer[i].FeedForward();
                if (i != NumberOfLayers-1)
                {
                    /*xxx=Layer[i].OutputVector();
                    Layer[i+1].Input = new double[Layer[i].NodeLength];
                    for(int f=0;f<Layer[i].NodeLength;f++)
                    {
                        Layer[i+1].Input[f]=xxx[f];
                    }*/
                    Layer[i+1].Input = Layer[i].OutputVector();
                }
                    //Layer[i+1].Input = Layer[i].OutputVector();
		}


}
void BACKPROP::UpdateWeights(){

		CalculateDelta();
		BackPropError();
}
void BACKPROP::CalculateDelta(){
    int i,j,k;
    for(i=0;i<Layer[NumberOfLayers-1].NodeLength;i++){
        Layer[NumberOfLayers-1].Node[i].Delta=(ExpectedOutput[SampleNumber][i] - Layer[NumberOfLayers-1].Node[i].Output)*dsigmoid(Layer[NumberOfLayers-1].Node[i].Output);
        //cout << ExpectedOutput[SampleNumber][i] << endl;
    }
    for(i = NumberOfLayers -2;i>0;i--){

        for(j=0;j<Layer[i].NodeLength;j++){
            double sum = 0.0;
            for(k=0;k<Layer[i+1].NodeLength;k++){
                sum += Layer[i+1].Node[k].Weight[j]*Layer[i+1].Node[k].Delta;

            }
             double temp;
            Layer[i].Node[j].Delta = dsigmoid(Layer[i].Node[j].Output)*sum;
            /*for(int x = 0;x<Layer[i-1].NodeLength;x++){
               temp = Layer[i].Node[j].Weight[x] + LearningRate*Layer[i].Node[j].SignalError*Layer[i-1].Node[x].Output;
               Layer[i].Node[j].Weight[x]=temp;
                temp = Layer[i].Node[j].Threshold + LearningRate*Layer[i].Node[j].SignalError;
                Layer[i].Node[j].Threshold = temp;
            }*/
        }
    }
}
void BACKPROP::BackPropError(){
    int i,j,k;
    for(i = NumberOfLayers-1;i>0;i--){
        for(j=0;j<Layer[i].NodeLength;j++){
            Layer[i].Node[j].ThresholdDiff = LearningRate*Layer[i].Node[j].Delta +Momentum*Layer[i].Node[j].ThresholdDiff;
            Layer[i].Node[j].Threshold+=Layer[i].Node[j].ThresholdDiff;

            for(k=0;k<Layer[i].InputLength;k++){
                Layer[i].Node[j].WeightDiff[k] = LearningRate*Layer[i].Node[j].Delta*Layer[i-1].Node[k].Output + Momentum*Layer[i].Node[j].WeightDiff[k];
                Layer[i].Node[j].Weight[k] += Layer[i].Node[j].WeightDiff[k];

            }
        }
    }


}

void BACKPROP::CalculateOverallError(){
    int i,j;
    OverAllError = 0;
        i = SampleNumber-1;
        for(j=0;j<Layer[NumberOfLayers-1].NodeLength;j++)
            OverAllError+=pow(ExpectedOutput[i][j]-ActualOutput[i][j],2);
            //cout<<" i ="<<i<<" j = "<<j<<;


    OverAllError/=2.0;

}

 BACKPROP::BACKPROP(int NumberOfNodes[],double **InputSamples,double ** OutputSamples,double LearnRate,double Moment,double MinError,long int MaxIter,int nos,int nol){
    int i,j;
    NumberOfSamples=nos;//sizeof(InputSamples)/sizeof(InputSamples[0]);
    MinimumError = MinError;
    LearningRate = LearnRate;
    Momentum = Moment;
    NumberOfLayers = nol;//sizeof(NumberOfNodes)/sizeof(NumberOfNodes[0]);
    cout<<NumberOfLayers<<endl;
    MaximumNumberOfIterations = MaxIter;
    Layer = (LAYER *)malloc(NumberOfLayers*sizeof(LAYER));
    Layer[0]=LAYER(NumberOfNodes[0],NumberOfNodes[0]);
    for (i = 1; i < NumberOfLayers; i++)
			Layer[i] =LAYER(NumberOfNodes[i],NumberOfNodes[i-1]);

    Input = new double *[NumberOfSamples];
    for(i=0;i<NumberOfSamples;i++)
        Input[i]=new double [Layer[0].NodeLength];
		ExpectedOutput = new double*[NumberOfSamples];
		ActualOutput = new double*[NumberOfSamples];
    for(int i=0;i<NumberOfSamples;i++){
        ExpectedOutput[i] = new double[Layer[NumberOfLayers-1].NodeLength];
		ActualOutput[i] = new double[Layer[NumberOfLayers-1].NodeLength];
    }
		for (i = 0; i < NumberOfSamples; i++)
			for (j = 0; j < Layer[0].NodeLength; j++)
				Input[i][j] = InputSamples[i][j];

        for (i = 0; i < NumberOfSamples; i++)
			for (j = 0; j < Layer[NumberOfLayers-1].NodeLength; j++)
				ExpectedOutput[i][j] = OutputSamples[i][j];
    TrainNetwork();
}
void BACKPROP::TrainNetwork(){
    int i,j;
    long int k=0;
    do{
        for (SampleNumber = 0; SampleNumber < NumberOfSamples; SampleNumber++) {
           			for (i = 0; i < Layer[0].NodeLength; i++)
					{
					    Layer[0].Input[i] = Input[SampleNumber][i];
                        //cout << Layer[0].Input[i] << " ";
					}

				FeedForward();

				for (i = 0; i < Layer[NumberOfLayers-1].NodeLength; i++)
           				ActualOutput[SampleNumber][i] =
						Layer[NumberOfLayers-1].Node[i].Output;
				UpdateWeights();
    }
        k++;
        CalculateOverallError();
//cout<<OverAllError<<endl;
}while(k < MaximumNumberOfIterations);
}
double BACKPROP::Test(double input[]){
    int winner=0;
    double *output_nodes;
    int nl=NumberOfLayers;
    for (int j = 0; j < Layer[0].NodeLength; j++){
			Layer[0].Input[j] = input[j];
			//cout<<"fine"<<endl;
			}

		FeedForward();
		//cout<<"OK1"<<endl;
		output_nodes = Layer[nl-1].OutputVector();

		//cout<<"OK2  "<<nl<<endl;
		//int n=0;
		double nu=0;
		//for(int i=0;i<Layer[nl-1].NodeLength;i++){
          //  nu+=Layer[nl-1].Node[i].Output;}

            //cout<<Layer[nl-1].Node[i].Output<<endl;}
		//cout<<"OK3  "<<n<<endl;

			//cout<<"OK4"<<endl;

    return output_nodes[0];
}
