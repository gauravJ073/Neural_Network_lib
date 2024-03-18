#include <vector>
#include <sstream>
#include <fstream>
#include <algorithm>
// #include <iostream>

using namespace std;
//-------------------------------------------training-data----------------------------------------

class CSVdata{
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
        CSVdata();
        CSVdata(CSVdata &data);
    private:
        ifstream training_file;
};
CSVdata::CSVdata(){;}
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
        output_vec.push_back(temp);
    }
}
void CSVdata::loadData(string filename){
    training_file.open(filename.c_str());
    string line;
    int i=0;
    for(i=0;!training_file.eof();i++){
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
                ss>>x;
                row.push_back(x);
                start=end+1;
            }
            end++;
        }
        target.push_back(line.substr(start, end-start));
        Data.push_back(row);
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
    sort(target_classes.begin(), target_classes.end());
    calculateOutput();
    
}