#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<vector>
#include<cmath>
#include"NMF.hpp"

using namespace std;
using namespace Eigen;

int main(){
  const int freqbins=100;//1024;
  const int time=20;
  const int coms=3;
  const int Iteration=100;
  // data = W*H
  cout<<"data setting"<<endl;
  MatrixXd data=MatrixXd::Random(freqbins,time).cwiseAbs();
  MatrixXd W=MatrixXd::Random(freqbins,coms).cwiseAbs();
  MatrixXd H=MatrixXd::Random(coms,time).cwiseAbs();
  cout<<"### object func ###"<<endl;
  nmf::NMF nmfsolve;
  nmfsolve.set(0); // Euclid
  nmfsolve.W=W;
  nmfsolve.D=H;
  for(int i=0;i<Iteration;i++){
    nmfsolve.update_D(&data);
    nmfsolve.update_W(&data);
    cout<<(data-nmfsolve.W*nmfsolve.D).cwiseAbs2().sum()<<endl;
  }
  return 0;
}
