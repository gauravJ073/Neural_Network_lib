#include <iostream>
#include <vector>

#include "lib/CSVdata.h"
#include "lib/neuralnet.h"

int main(){
    CSVdata data;

    vector<unsigned> topology;
    topology.push_back(32);
    topology.push_back(16);
    topology.push_back(16);
    // topology.push_back(4);
    // topology.push_back(4);
    topology.push_back(2);

    NNModel model(100, 10, topology, ".\\dataset\\ionosphere_train.csv");
    model.train();

    model.calcConfusionMatrix(model.test(".\\dataset\\ionosphere_test.csv"));
    model.printConfusionMatrix();
    model.printAccuracy();
}