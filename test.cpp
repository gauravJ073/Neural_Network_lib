#include <iostream>
#include <vector>

#include "lib/CSVdata.h"
#include "lib/neuralnet.h"

int main(){
    CSVdata data;
    // data.loadData("..\\dataset\\mnist_train.csv");

    vector<unsigned> topology;
    topology.push_back(4);
    // topology.push_back(392);
    // topology.push_back(196);
    // topology.push_back(98);
    topology.push_back(8);
    topology.push_back(8);
    topology.push_back(3);

    NNModel model(1000, 50, topology, ".\\dataset\\norm_iris.csv");
    model.train();
    // vector<int> actual;
    // for(int i = 0; i<data.output_vec.size();i++){
    //     actual.push_back(model.getIdx(data.output_vec[i]));
    //     cout<<(data.output_vec[i][i]);

    // }

    model.calcConfusionMatrix(model.test(".\\dataset\\norm_iris_test.csv"));
    model.printConfusionMatrix();
}