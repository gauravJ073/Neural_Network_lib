#include <iostream>
#include <vector>

#include "lib/CSVdata.h"
#include "lib/neuralnet.h"

int main(){
    CSVdata data;

    vector<unsigned> topology;
    topology.push_back(8);
    topology.push_back(4);
    topology.push_back(4);
    topology.push_back(4);
    topology.push_back(4);
    topology.push_back(2);

    NNModel model(100, 10, topology, ".\\dataset\\Churn_train.csv");
    model.train();

    model.calcConfusionMatrix(model.test(".\\dataset\\Churn_test.csv"));
    model.printConfusionMatrix();
    model.printAccuracy();
}