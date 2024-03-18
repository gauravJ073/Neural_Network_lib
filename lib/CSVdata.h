#ifndef csvdata
#define csvdata

#include <vector>
#include <fstream>
//-------------------------------------------training-data----------------------------------------

class CSVdata{
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
        CSVdata();
        CSVdata(CSVdata &data);
    private:
        std::ifstream training_file;
};
#endif