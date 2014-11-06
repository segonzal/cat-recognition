#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main (int argc, const char** argv) 
{
   
    int K_values[4] = { 100, 500, 1000, 1500 };

    FileStorage f("descriptors.yml", FileStorage::READ);
    Mat descriptors;
    f["descriptors"] >> descriptors;
    f.release();

    FileStorage kmeans("kmeans.yml", FileStorage::WRITE);
    
    for (int i = 0; i < 4; i++) 
    {
        
        int K = K_values[i];
        BOWKMeansTrainer bowtrainer(K); //num clusters
        bowtrainer.add(descriptors);
        Mat vocabulary = bowtrainer.cluster();
        
        stringstream name;
        name << "kmeans" << K;

        cout << name.str() <<  endl;
        kmeans << name.str() << vocabulary;
        name.str(std::string());
        
    }
    kmeans.release();

    return 0;
}
