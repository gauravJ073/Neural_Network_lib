//neuralNet.cpp
#include <vector>
#include <set>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>


using namespace std;

class Neuron; //declaring neuron first so that we can create a layer

typedef vector<Neuron> Layer;

class Connection{
    public:
        double weight;
        double delta_weight;//delta weight
        Connection();
    private:
        static double randomWeight(){return rand()/double(RAND_MAX);}
};
Connection::Connection(){
    weight=randomWeight();
}

//-------------------------------------------training-data----------------------------------------

class TrainingData{
    public:
        int input_size;
        int feature_size;
        vector<vector<double> > Data;
        vector<string> target;
        vector<string> target_classes;
        vector<vector<double> > output_vec;

        bool isEof(){return training_file.eof();}
        void loadData(string filename);
        void calculateOutput();
    private:
        ifstream training_file;


};
void TrainingData::calculateOutput(){
    for(int i=0;i<input_size;i++){
        vector<double> temp;
        for(int j=0;j<target_classes.size();j++){
            if(target[i]==target_classes[j]){
                temp.push_back(1.0);
            }
            else{
                temp.push_back(0);
            }
        }
        output_vec.push_back(temp);
    }
}
void TrainingData::loadData(string filename){
    training_file.open(filename.c_str());
    string line;
    int i=0;
    for(i=0;!training_file.eof();i++){
        getline(training_file, line);
        int start=0, end=0;
        vector<double> row;
        while(end<(line.size())){
            if(line[end]==','){

                string temp=line.substr(start, end-start);

                stringstream ss;
                ss<<temp;
                double x;
                ss>>x;
                row.push_back(x);
                start=end+1;
            }
            end++;
        }
        target.push_back(line.substr(start, end-start));
        Data.push_back(row);
        
        // cout<<target[i]<<endl;
        int flag=1;
        for(int j=0;j<target_classes.size();j++){
            if(target_classes[j]==target[i]){
                flag=0;
            }
        }
        if(flag){
            target_classes.push_back(line.substr(start, end-start));
        }
        

        input_size=i+1;
        feature_size=Data[0].size();
        cout<<"Output classes: "<<target_classes.size()<<endl;
        cout<<"input size: "<<input_size<<endl;
        cout<<"features: "<<feature_size<<endl;
        
    
    }
    // cout<<"Before sorting";
    // for(int i=0;i<target_classes.size();i++){
    //     cout<<target_classes[i]<<", ";
    // }
    // cout<<endl;
    sort(target_classes.begin(), target_classes.end());
    // cout<<"After sorting";
    // for(int i=0;i<target_classes.size();i++){
    //     cout<<target_classes[i]<<", ";
    // }
    // cout<<endl;
    calculateOutput();
    
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
        static double activation_function(double x){return tanh(x);}//ReLU function
        static double activation_function_derivative(double x){return 1.0 - x * x;}//Derivative of ReLU function
        double sumDerivativeOfWeight(const Layer &next_layer);
        double neuron_output;
        unsigned neuron_index;
        double gradient;
        vector<Connection> neuron_outputweight;
};

double Neuron::eta=0.05;
double Neuron::alpha=0.1;

Neuron::Neuron(unsigned num_output, unsigned index){
    for (unsigned connection=0;connection<num_output;connection++){
        neuron_outputweight.push_back(Connection());
        // cout<<" Value ["<<connection<<"]: "<<neuron_outputweight.back().weight<<".";
    }

    neuron_index=index;
}

void Neuron::calculateOutput(const Layer &prevLayer){
    // activation a = g(sum(weight_prevlayer*output_prevlayer))

    double sum=0.0;
    for(unsigned prev_neuron=0;prev_neuron<prevLayer.size();prev_neuron++){
        sum+=prevLayer[prev_neuron].get_neuron_output() * prevLayer[prev_neuron].neuron_outputweight[neuron_index].weight;
    }
    neuron_output=activation_function(sum);
}

void Neuron::calculateOutputGradient(double target_val){
    // cout<<"output gradient"<<endl;
    double delta=target_val-neuron_output;
    // cout<<"delta: "<<delta<<endl;
    // cout<<"activation_function_derivative(neuron_output): "<<activation_function_derivative(neuron_output)<<endl;
    gradient=delta*Neuron::activation_function_derivative(neuron_output);
}

void Neuron::calculateHiddenGradient(const Layer &next_layer){
    // cout<<"Hidden gradient"<<endl;
    double derivative_of_weight = sumDerivativeOfWeight(next_layer);
    // cout<<"derivative_of_weight: "<<derivative_of_weight<<endl;
    // cout<<"activation_function_derivative(neuron_output): "<<activation_function_derivative(neuron_output)<<endl;
    gradient=derivative_of_weight*Neuron::activation_function_derivative(neuron_output);
}

double Neuron::sumDerivativeOfWeight(const Layer &next_layer){
    double sum=0.0;
    // cout<<"Summing"<<endl;
    for(int neuron=0;neuron<next_layer.size()-1;neuron++){
        // cout<<"neuron_outputweight[neuron].weight"<<neuron_outputweight[neuron].weight<<", ";
        sum+=neuron_outputweight[neuron].weight*next_layer[neuron].gradient;
    }
    return sum;
}

void Neuron::updateInputWeights(Layer &prev_layer){
    for(int neuron=0;neuron<prev_layer.size();neuron++){
        Neuron &curr_neuron=prev_layer[neuron];
        double oldDeltaWeight = curr_neuron.neuron_outputweight[neuron_index].delta_weight;
        double newDeltaWeight=eta*curr_neuron.get_neuron_output()*gradient+alpha*oldDeltaWeight;
        curr_neuron.neuron_outputweight[neuron_index].delta_weight=newDeltaWeight;
        curr_neuron.neuron_outputweight[neuron_index].weight+=newDeltaWeight;
    }

}

//---------------------------------Network-----------------------------------------

class Network{
    public:
        Network(vector<unsigned> &topology);
        void forwardpropagation(const vector<double> &input);
        void backpropagation(const vector<double> &target);
        void getResult(vector<double> &result) const;

    private: 
        vector<Layer> layers;
        double error;
        double recent_average_error;
        double recent_average_smoothing_factor;
};

Network::Network(vector<unsigned> &topology){
    unsigned num_layers=topology.size();
    for (unsigned layer=0;layer<num_layers;layer++){
        layers.push_back(Layer());
        unsigned num_output= layer==num_layers-1?0:topology[layer+1];

        //have a bias neuron
        for(unsigned neuron=0;neuron<=topology[layer];neuron++){
            layers.back().push_back(Neuron(num_output, neuron));
            // cout<<"Neuron create....neuron_num: "<<neuron<<",";
        }
        cout<<endl;
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
            // cout<<"-------layer["<<layer<<"]-------"<<endl;
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
    // cout<<"getresult";
    for(int neuron=0;neuron<layers.back().size()-1;neuron++){
        // cout<<layers.back()[neuron].get_neuron_output()<<endl;
        result.push_back(layers.back()[neuron].get_neuron_output());
    }
}

//-------------------------------------------Train Network-----------------------------------------
class TrainNet{
    private:
        int epoch;
        int batch_size;
        int iteration;
        Network *net;
    public:
        TrainNet(){};
        TrainNet(int ep, int b_s, Network *model, TrainingData &data);
        void training(TrainingData &data);

        
};
TrainNet::TrainNet(int ep, int b_s, Network *model, TrainingData &data){
    epoch=ep;
    batch_size=b_s;
    net=model;
    assert(batch_size<=data.input_size);
    iteration = data.input_size/float(batch_size);
}
void TrainNet::training(TrainingData &data){
    while(epoch--){
        int itr=0;
        while(itr<iteration){
            for(int i=itr*batch_size;i<(itr+1)*batch_size;i++){
                vector<double> input, target, result;
                input=data.Data[i];
                for(int j=0;j<input.size();j++){
                    cout<<"input ="<<input[j]<<", ";
                }
                net->forwardpropagation(input);
                net->getResult(result);
                int maxidx=0;
                for(int j=0;j<result.size();j++){
                    if(result[j]>result[maxidx]){
                        maxidx=j;
                    }
                }
                cout<<endl;
                for(int j=0;j<result.size();j++){
                    cout<<"result ["<<j<<"]="<<result[j]<<endl;
                }
                // cout<<"result ="<<result[1]<<endl;
                cout<<"Output: "<<data.target_classes[maxidx]<<endl;
                maxidx=0;
                for(int j=0;j<data.output_vec[i].size();j++){
                    if(data.output_vec[i][j]>data.output_vec[i][maxidx]){
                        maxidx=j;
                    }
                    // cout<<"-"<<data.output_vec[i][j]<<" ,";
                }
                target=data.output_vec[i];
                // cout<<"Index: "<<maxidx<<endl;
                cout<<"Actual: "<<data.target_classes[maxidx]<<endl;
                net->backpropagation(target);
                cout<<"<------------------------>"<<endl;

            }
            itr++;
        }
    }
}


int main(){
    
    // eg {3, 2, 1}
    // vector<unsigned> topology;
    // topology.push_back(3);
    // topology.push_back(2);
    // topology.push_back(1);
    // Network net(topology);

    // vector<double> input;
    // net.forwardpropagation(input);

    // vector<double> target;
    // net.backpropagation(target);

    // vector<double> results;
    // net.getResult(results);

    TrainingData data;
    data.loadData("dataset\\trainingdata.csv");

    vector<unsigned> topology;
    topology.push_back(data.feature_size);
    topology.push_back(8);
    topology.push_back(8);
    topology.push_back(8);
    topology.push_back(data.output_vec[0].size());
    cout<<data.output_vec[0].size()<<endl;
    Network net(topology);
    cout<<"Netowrk ready"<<endl;

    TrainNet train(2000, 7, &net, data);
    train.training(data);
}