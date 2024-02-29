#include <iostream>
#include <vector>

#include "lib/trainingdata.h"
#include "lib/neuralnet.h"

int main(){
    TrainingData dataset;
    dataset.loadData("dataset\\trainingdata.csv");
    std::cout<<"all ok"<<std::endl<<dataset.input_size;

    vector<unsigned> topology;
    topology.push_back(dataset.feature_size);
    topology.push_back(7);
    topology.push_back(7);
    topology.push_back(7);
    topology.push_back(dataset.output_vec[0].size());
    cout<<dataset.output_vec[0].size()<<endl;
    Network net(topology);
    cout<<"Netowrk ready"<<endl;

    TrainNet train(6000, 7, &net, dataset);
    train.training(dataset);
}