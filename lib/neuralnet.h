#ifndef neuralnet
#define neuralnet

#include <vector>
#include "csvdata.h"


using namespace std;

class Neuron; //declaring neuron first so that we can create a layer
typedef vector<Neuron> Layer;

class Connection{
    public:
        double weight;
        double delta_weight;
        Connection();
    private:
        static double randomWeight();
};


//----------------------------Neuron-----------------------------
// neuron definition is after Layer because we need Layer in its definition
class Neuron{
    public:
        public:
        Neuron(unsigned num_output, unsigned index);
        void set_neuron_output(double input);
        double get_neuron_output() const;
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

#endif