//neuralNet.cpp
#include <vector>
#include <set>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <sstream>
#include <cmath>
#include "CSVdata.h"
#include <iomanip>

using namespace std;

class Neuron; //declaring neuron first so that we can create a layer
typedef vector<Neuron> Layer;

class Connection{
    public:
        double weight;
        double delta_weight;//delta weight
        Connection();
    private:
        static double randomWeight(){return 2*(rand()/double(RAND_MAX)) -1;}
};
Connection::Connection(){
    weight=randomWeight();
}


//----------------------------Neuron-----------------------------
// neuron definition is after Layer because we need Layer in its definition
class Neuron{
    public:
        Neuron(unsigned num_output, unsigned index);
        void set_neuron_output(double input){ neuron_output = input; }
        double get_neuron_output() const { return neuron_output ; }
        void calculateOutput(const Layer &prevLayer);
        void calculateOutputGradient(double target_val);
        void calculateHiddenGradient(const Layer &next_layer);
        void updateInputWeights(Layer &prev_layer);
    private:
        static double eta; //[0....1]
        static double alpha; //[0....1]
        static double activation_function(double x){return tanh(x);}//tanh function
        static double activation_function_derivative(double x){return 1.0 - x * x;}//Derivative of tanh function
        double sumDerivativeOfWeight(const Layer &next_layer);
        double neuron_output;
        unsigned neuron_index;
        double gradient;
        vector<Connection> neuron_outputweight;
};

double Neuron::alpha=0.15;
double Neuron::eta=0.5;

Neuron::Neuron(unsigned num_output, unsigned index){
    for (unsigned connection=0;connection<num_output;connection++){
        neuron_outputweight.push_back(Connection());
    }

    neuron_index=index;
}

void Neuron::calculateOutput(const Layer &prevLayer){

    double sum=0.0;
    for(unsigned prev_neuron=0;prev_neuron<prevLayer.size();prev_neuron++){
        sum+=prevLayer[prev_neuron].get_neuron_output() * prevLayer[prev_neuron].neuron_outputweight[neuron_index].weight;
    }
    neuron_output=activation_function(sum);
}

void Neuron::calculateOutputGradient(double target_val){
    double delta=target_val-neuron_output;
    gradient=delta*Neuron::activation_function_derivative(neuron_output);
}

void Neuron::calculateHiddenGradient(const Layer &next_layer){
    double derivative_of_weight = sumDerivativeOfWeight(next_layer);
    gradient=derivative_of_weight*Neuron::activation_function_derivative(neuron_output);
}

double Neuron::sumDerivativeOfWeight(const Layer &next_layer){
    double sum=0.0;
    for(int neuron=0;neuron<next_layer.size()-1;neuron++){
        sum+=neuron_outputweight[neuron].weight*next_layer[neuron].gradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer &prev_layer){
    for(int neuron=0;neuron<prev_layer.size();neuron++){
        Neuron &curr_neuron=prev_layer[neuron];
        double oldDeltaWeight = curr_neuron.neuron_outputweight[neuron_index].delta_weight;
        double newDeltaWeight=alpha*curr_neuron.get_neuron_output()*gradient;//+eta*oldDeltaWeight;
        curr_neuron.neuron_outputweight[neuron_index].delta_weight=newDeltaWeight;
        curr_neuron.neuron_outputweight[neuron_index].weight+=newDeltaWeight;
    }

}

//---------------------------------Network-----------------------------------------

class Network{
    public:
        Network(vector<unsigned> &topology);
        Network(Network &n);
        void forwardpropagation(const vector<double> &input);
        void backpropagation(const vector<double> &target);
        void getResult(vector<double> &result) const;
        vector<unsigned> gettopology(){return tp;}

    private: 
        vector<Layer> layers;
        vector<unsigned> tp;
        double error;
        double recent_average_error;
        double recent_average_smoothing_factor;
};

Network::Network(vector<unsigned> &topology){
    tp=topology;
    unsigned num_layers=topology.size();
    for (unsigned layer=0;layer<num_layers;layer++){
        layers.push_back(Layer());
        unsigned num_output= layer==num_layers-1?0:tp[layer+1];

        //have a bias neuron
        for(unsigned neuron=0;neuron<=tp[layer];neuron++){
            layers.back().push_back(Neuron(num_output, neuron));
        }
    }
}
Network::Network(Network &n){
    tp=n.tp;
    unsigned num_layers=tp.size();
    for (unsigned layer=0;layer<num_layers;layer++){
        layers.push_back(Layer());
        unsigned num_output= layer==num_layers-1?0:tp[layer+1];

        //have a bias neuron
        for(unsigned neuron=0;neuron<=tp[layer];neuron++){
            layers.back().push_back(Neuron(num_output, neuron));
        }
    }
}
void Network::forwardpropagation(const vector<double> &input){
    assert(input.size()==(layers[0].size()-1));//ensuring size of input is same as size of input layer

    for(unsigned i=0;i<input.size();i++){
        layers[0][i].set_neuron_output(input[i]);//setting the output of neurons in input layer = respective element in input
    }

    for(unsigned layer=1;layer<layers.size();++layer){//iterate through each layer
    Layer &prevLayer = layers[layer-1];
        for(unsigned neuron=0;neuron<layers[layer].size()-1;++neuron){//iterate through each neuron of the layer except the bias
            layers[layer][neuron].calculateOutput(prevLayer);
        }
    }
}

void Network::backpropagation(const vector<double> &target){
    Layer &outputlayer=layers.back();

    //calculate RMS error
    error=0.0;
    for(int neuron =0;neuron<outputlayer.size();neuron++){
        double delta = target[neuron]-outputlayer[neuron].get_neuron_output();
        error+=delta*delta;
    }
    error/=outputlayer.size()-1;
    error=sqrt(error);
    //get running error
    recent_average_error=(recent_average_error*recent_average_smoothing_factor+error)/(recent_average_smoothing_factor+1);

    //calculate output layer gradient
    for(int neuron=0;neuron<outputlayer.size();neuron++){
        outputlayer[neuron].calculateOutputGradient(target[neuron]);
    }

    //calculate gradient of hidden layer
    for(int layer=layers.size()-2;layer>0;layer--){
        Layer &hidden_layer=layers[layer];
        Layer &next_hidden_layer=layers[layer+1];

        for(int neuron=0;neuron<hidden_layer.size();neuron++){
            hidden_layer[neuron].calculateHiddenGradient(next_hidden_layer);
        }
    }
    //update weights
    for(int layer=layers.size()-1;layer>0;layer--){
        Layer &curr_layer=layers[layer];
        Layer &prev_layer=layers[layer-1];

        for(int neuron=0;neuron<layers.size()-1;neuron++){
            curr_layer[neuron].updateInputWeights(prev_layer);
        }
    }
}

void Network::getResult(vector<double> &result) const {
    result.clear();
    for(int neuron=0;neuron<layers.back().size()-1;neuron++){
        result.push_back(layers.back()[neuron].get_neuron_output());
    }
}

//-------------------------------------------Train Network-----------------------------------------
class NNModel{
    public:
        NNModel(){};
        NNModel(int ep, int b_s, vector<unsigned> topology, string traindatapath);
        NNModel(int ep, int b_s, vector<unsigned> topology, string traindatapath, string testdatapath);
        void train();
        int predict(vector<double>& input);
        vector<int> test(string testdatapath);
        vector<int> predict(vector<vector<double> >& input);
        void calcConfusionMatrix(vector<int> predictedclass);
        void printConfusionMatrix();
        void printAccuracy();
        int getIdx(vector<double> result);
    private:
        int epoch;
        int batch_size;
        int iteration;
        int output_classes;
        vector<string> output_class_labels;
        CSVdata traindata;
        CSVdata *testdata;
        Network *net;
        struct evaluation{
            vector<vector<int> > confusion_matrix;
            vector<double> tp, tn, fp, fn;
        }eval;

        
};
NNModel::NNModel(int ep, int b_s, vector<unsigned> topology, string traindatapath){
    epoch=ep;
    batch_size=b_s;
    Network n(topology);
    net=new Network(n);

    traindata.loadData(traindatapath);
    testdata=new CSVdata(traindata);
    assert(batch_size<=traindata.input_size);
    iteration = traindata.input_size/float(batch_size);

    for(int i =0;i<traindata.target_classes.size();i++){
        output_class_labels.push_back(traindata.target_classes[i]);
    }
    output_classes=traindata.output_vec[0].size();

    for( int i=0;i<output_classes;i++){
        vector<int> temp;
        for( int j=0;j<output_classes;j++){
            temp.push_back(0);
        }
        eval.tp.push_back(0);
        eval.fp.push_back(0);
        eval.tn.push_back(0);   
        eval.fn.push_back(0);
        eval.confusion_matrix.push_back(temp);
    }
}

int NNModel::getIdx(vector<double> output_vec){
    int maxidx=0;
    for(int j=0;j<output_vec.size();j++){
        if(output_vec[j]>output_vec[maxidx]){
            maxidx=j;
        }
    }
    return maxidx;
}
void NNModel::train(){
    while(epoch--){
        int itr=0;
        while(itr<iteration){
            for(int i=itr*batch_size;i<(itr+1)*batch_size;i++){
                vector<double> input, target, result;
                input=traindata.Data[i];
                net->forwardpropagation(input);
                net->getResult(result);
                int maxidx=getIdx(result);
                
                maxidx=getIdx(traindata.output_vec[i]);
                target=traindata.output_vec[i];
                net->backpropagation(target);

            }
            itr++;
        }
        cout<<"Epochs left: "<<epoch<<endl;
    }
}
void NNModel::calcConfusionMatrix(vector<int> predictedclass){
    vector<int> actualclass;
    for(int i = 0; i<testdata->output_vec.size();i++){
        actualclass.push_back(getIdx(testdata->output_vec[i]));
    }
    assert(actualclass.size()==predictedclass.size());
    for(int i=0;i<actualclass.size();i++){
        eval.confusion_matrix[actualclass[i]][predictedclass[i]]+=1;
    }
    
    for(int i=0;i<eval.tp.size();i++){
        eval.tp[i]=eval.confusion_matrix[i][i];
    }
    for(int i=0;i<eval.fp.size();i++){
        double fp=0;
        for(int j=0;j<eval.fp.size();j++){
            fp+=eval.confusion_matrix[j][i];
        }
        fp=fp-eval.tp[i];
        eval.fp[i]=fp;
    }
    for(int i=0;i<eval.fn.size();i++){
        double fn=0;
        for(int j=0;j<eval.fn.size();j++){
            fn+=eval.confusion_matrix[i][j];
        }
        fn=fn-eval.tp[i];
        eval.fn[i]=fn;
    }
    for(int i=0;i<eval.tn.size();i++){
        double tn=0;
        for(int j=0;j<eval.tn.size();j++){
            for(int k=0;k<eval.tn.size();k++){
                tn+=eval.confusion_matrix[j][k];
            }
        }
        tn=tn-eval.tp[i]-eval.fp[i]-eval.fn[i];
        eval.tn[i]=tn;
    }
}

void NNModel::printConfusionMatrix(){
    cout<<endl<<"act\\pred";
    for(int i=0;i<output_classes;i++){
        cout<<setw(8)<<"["<<i<<"]";
    }
    cout<<endl;
    for(int i=0;i<output_classes;i++){
        cout<<"["<<i<<"]    ";
        for( int j=0;j<output_classes;j++){
            cout<<setw(10)<<eval.confusion_matrix[i][j]<<"";
        }
        cout<<endl;
    }
    cout<<"Labels: "<<endl;
    for(int i=0;i<output_classes;i++){
        cout<<"["<<i<<"]    "<<output_class_labels[i]<<endl;
    }

}

void NNModel::printAccuracy(){
    cout<<"Accuracy: "<<endl;
    double numerator=0, denomenator=0;
    for(int i =0;i<output_classes;i++){
        numerator+=eval.tp[i]+eval.tn[i];
    }
    for(int i =0;i<output_classes;i++){
        denomenator+=eval.tp[i]+eval.tn[i]+eval.fp[i]+eval.fn[i];
    }
    cout<<numerator/denomenator<<endl;
}

int NNModel::predict(vector<double>& input){
    assert(input.size()==net->gettopology()[0]);
    net->forwardpropagation(input);
    vector<double> result;
    net->getResult(result);
    return getIdx(result);
}

vector<int> NNModel::test(string testdatapath){
    
    testdata->loadData(testdatapath);
    vector<int> pred=predict(testdata->Data);
    return pred;
}

vector<int> NNModel::predict(vector<vector<double> >& input){
    vector<int> pred;
    
    for(int i=0;i<input.size();i++){
        pred.push_back(predict(input[i]));
    }
    return pred;
}