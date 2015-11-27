    #include<bits/stdc++.h>
#include "newbpnn.cpp"

using namespace std;
int main(void){
int nodes[]={2,2,1};
double x,y,z;
double **Input = new double *[4];
double **Output = new double *[4];
for(int i=0;i<4;i++){
    Input[i]=new double[2];
    Output[i]=new double[1];


    /*cin>>x>>y>>z;
    Input[i][0]=x;
    Input[i][1]=y;
    Output[i][0]=z;*/
}
Input[0][0]=0;
Input[0][1]=0;
Output[0][0]=0;
Input[1][0]=0;
Input[1][1]=1;
Output[1][0]=1;
Input[2][0]=1;
Input[2][1]=0;
Output[2][0]=1;
Input[3][0]=1;
Input[3][1]=1;
Output[3][0]=0;

double learn=0.5;
double moment=0.1;
double minerror=0.005;
long int maxiter=1000000;
    BACKPROP p(nodes,Input,Output,learn,moment,minerror,maxiter,4,3);

    //for(int i=0;i<10;i++)
            //p.TrainNetwork();
    double inp[2];
    for(int i=0;i<4;i++){
            //p.TrainNetwork();
    cout<<"done training\n";
        cin>>inp[0]>>inp[1];
    cout<<p.Test(inp)<<endl;}

}
