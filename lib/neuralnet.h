#ifndef neuralnet
#define neuralnet

#include <vector>
#include "trainingData.h"


using namespace std;

class Neuron; //declaring neuron first so that we can create a layer
typedef vector<Neuron> Layer;

class Connection{
    public:
        double weight;
        double delta_weight;//delta weight
        Connection();
    private:
        static double randomWeight();
};


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
        static double activation_function(double x);//tanh function
        static double activation_function_derivative(double x);//Derivative of tanh function
        double sumDerivativeOfWeight(const Layer &next_layer);
        double neuron_output;
        unsigned neuron_index;
        double gradient;
        vector<Connection> neuron_outputweight;
};

double Neuron::eta=0.5;
double Neuron::alpha=0.15;

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

#endif