#ifndef trainingdata
#define trainingdata

#include <vector>
#include <fstream>
//-------------------------------------------training-data----------------------------------------

class TrainingData{
    public:
        int input_size;
        int feature_size;
        std::vector<std::vector<double> > Data;
        std::vector<std::string> target;
        std::vector<std::string> target_classes;
        std::vector<std::vector<double> > output_vec;

        bool isEof();
        void loadData(std::string filename);
        void calculateOutput();
    private:
        std::ifstream training_file;
};
#endif