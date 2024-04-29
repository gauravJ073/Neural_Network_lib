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

//-------------------------------------------train-data----------------------------------------

class CSVdata{
    public:
        int input_size;
        int feature_size;
        vector<vector<double> > Data;
        vector<string> target;
        vector<string> target_classes;//class labels in target
        vector<vector<double> > output_vec;//target class labels as output vector (eg if there are 7 output and our target is at index 3, [0][0][0][1][0][0][0][0])

        bool isEof(){return training_file.eof();}
        void loadData(string filename);
        void calculateOutput();
        CSVdata(){};
        CSVdata(CSVdata &data);
    private:
        ifstream training_file;


};
CSVdata::CSVdata(CSVdata &data){
    for(int i=0;i<data.target_classes.size();i++){
        target_classes.push_back(data.target_classes[i]);
    }
}

void CSVdata::calculateOutput(){
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
        // for(int i=0;i<temp.size();i++){
        //     cout<<temp[i]<<" ";
        // }
        // cout<<endl;
        output_vec.push_back(temp);
    }
}
void CSVdata::loadData(string filename){
    // cout<<"In load"<<endl;
    training_file.open(filename.c_str());
    string line;
    int i=0;
    for(i=0;!training_file.eof();i++){
        // cout<<i<<endl;
        getline(training_file, line);
        if(line==""){break;}
        int start=0, end=0;
        vector<double> row;
        while(end<(line.size())){
            if(line[end]==','){

                string temp=line.substr(start, end-start);

                stringstream ss;
                ss<<temp;
                double x;
                // int x;
                ss>>x;
                row.push_back(x);
                start=end+1;
            }
            end++;
        }
        target.push_back(line.substr(start, end-start));
        Data.push_back(row);
        // cout<<Data.size()<<endl;
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
        // cout<<"Output classes: "<<target_classes.size()<<endl;
        // cout<<"input size: "<<input_size<<endl;
        // cout<<"features: "<<feature_size<<endl;
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
        sum+=(prevLayer[prev_neuron].get_neuron_output() * prevLayer[prev_neuron].neuron_outputweight[neuron_index].weight);
    }
    
    // cout<<"sum "<<sum<<" ";
    // cout<<"activation_function(sum) "<<activation_function(sum)<<endl;
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
        // cout<<"next_layer[neuron].gradient"<<next_layer[neuron].gradient<<", ";
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
    unsigned num_layers=tp.size();
    for (unsigned layer=0;layer<num_layers;layer++){
        layers.push_back(Layer());
        unsigned num_output= layer==num_layers-1?0:tp[layer+1];

        //have a bias neuron
        for(unsigned neuron=0;neuron<=tp[layer];neuron++){
            layers.back().push_back(Neuron(num_output, neuron));
            // cout<<"Neuron create....neuron_num: "<<neuron<<",";
        }
        // cout<<endl;
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
            // cout<<"Neuron create....neuron_num: "<<neuron<<",";
        }
        cout<<endl;
    }
}

void Network::forwardpropagation(const vector<double> &input){
    // cout<<input.size()<<"  "<<layers[0].size()-1<<endl;
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
class NNModel{
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
                int act, pred;
                input=traindata.Data[i];
                // for(int j=0;j<input.size();j++){
                //     cout<<"input ="<<input[j]<<", ";
                // }
                net->forwardpropagation(input);
                net->getResult(result);
                int maxidx=getIdx(result);
                
                // cout<<endl;
                // for(int j=0;j<result.size();j++){
                //     cout<<"result ["<<j<<"]="<<result[j]<<endl;
                // }
                // cout<<"result ="<<result[1]<<endl;
                // cout<<"Output: "<<traindata.target_classes[maxidx]<<endl;
                pred=maxidx;
                maxidx=getIdx(traindata.output_vec[i]);

                target=traindata.output_vec[i];
                // cout<<"Index: "<<maxidx<<endl;
                // cout<<"Actual: "<<traindata.target_classes[maxidx]<<endl;
                act=maxidx;
                // cout<<"act: "<<act<<" pred: "<<pred<<endl;
                net->backpropagation(target);
                // cout<<"<----------------------->";

            }
            itr++;
        }
        // cout<<"<----------------------->"<<endl;
        cout<<"Epochs left: "<<epoch<<endl;
        // cout<<"<----------------------->"<<endl;
    }
}
void NNModel::calcConfusionMatrix(vector<int> predictedclass){
    vector<int> actualclass;
    for(int i = 0; i<testdata->output_vec.size();i++){
        actualclass.push_back(getIdx(testdata->output_vec[i]));
        // cout<<(traindata.output_vec[i][i]);
    }
    // cout<<endl;

    // cout<<actualclass.size()<<" "<<predictedclass.size();
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
    // for(int i=0;i<output_classes;i++){
    //     cout<<"["<<i<<"]    "<<output_class_labels[i]<<" : "<<eval.tp[i]<<" "<<eval.fp[i]<<" "<<eval.fn[i]<<" "<<eval.tn[i]<<endl;
    // }
    double numerator=0, denomenator=0;
    for(int i =0;i<output_classes;i++){
        numerator+=eval.tp[i]+eval.tn[i];
    }
    for(int i =0;i<output_classes;i++){
        denomenator+=eval.tp[i]+eval.tn[i]+eval.fp[i]+eval.fn[i];
    }
    cout<<numerator/denomenator<<endl;
    // cout<<(eval.tp[1]+eval.tn[1])/(eval.tp[1]+eval.tn[1]+eval.fp[1]+eval.fn[1])<<endl;
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
        // cout<<"pred idx: "<<pred.back()<<endl;
        // cout<<"pred label: "<<output_class_labels[pred.back()]<<endl;
    }
    return pred;
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

    CSVdata data;
    // data.loadData("..\\dataset\\mnist_train.csv");

    vector<unsigned> topology;
    topology.push_back(32);
    // topology.push_back(392);
    // topology.push_back(196);
    // topology.push_back(98);
    topology.push_back(16);
    topology.push_back(8);
    topology.push_back(2);

    NNModel model(500, 50, topology, "..\\dataset\\ionosphere_train.csv");
    model.train();
    // vector<int> actual;
    // for(int i = 0; i<data.output_vec.size();i++){
    //     actual.push_back(model.getIdx(data.output_vec[i]));
    //     cout<<(data.output_vec[i][i]);

    // }

    model.calcConfusionMatrix(model.test("..\\dataset\\ionosphere_test.csv"));
    model.printConfusionMatrix();
    model.printAccuracy();
}