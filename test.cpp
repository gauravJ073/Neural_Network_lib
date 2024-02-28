#include <iostream>
#include "lib/trainingdata.h"

int main(){
    TrainingData dataset;
    dataset.loadData("dataset\\trainingdata.csv");
    std::cout<<"all ok"<<std::endl<<dataset.input_size;
}