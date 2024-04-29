#include <iostream>
#include <vector>

#include "lib/CSVdata.h"
#include "lib/neuralnet.h"

int main(){
    CSVdata data;

    cout<<"Enter the file name for training data: ";
    string training_file;
    cin>>training_file;

    cout<<"Enter the file name for training data: ";
    string testing_file;
    cin>>testing_file;

    cout<<"Enter the number of features in data: ";
    int input_layer;
    cin>>input_layer;
    cout<<"Enter the number of output classes in data: ";
    int output_layer;
    cin>>output_layer;

    vector<unsigned> topology;
    topology.push_back(input_layer);
    cout<<"Enter the number of layers you want in the neural network: ";
    int n_layer;
    cin>>n_layer;
    for(int i=0;i<n_layer;i++){
        cout<<"Enter the number of neurons you want in the "<<i+1<<" layer: ";
        int n_neurons;
        cin>>n_neurons;
        topology.push_back(n_neurons);
    }
    topology.push_back(output_layer);


    NNModel model(100, 10, topology, ".\\dataset\\"+training_file);
    model.train();
    model.calcConfusionMatrix(model.test(".\\dataset\\"+testing_file));
    model.printConfusionMatrix();
    model.printAccuracy();
}