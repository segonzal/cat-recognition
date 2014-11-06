#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

String dog_folder = "./dogs/train/";
String cat_folder = "./cats/train/";

String out_name = "train";

int dog_size= 1;
int cat_size = 1;

// aweonao eran doubles no ints :3 por eso todo se iba a la mierda, saludos
// cordiales
void save_sparse_descriptors(ostream& stream, vector<int> labels,
        vector< vector<Mat> > data) 
{

    for (int i = 0; i < data.size(); ++i)
    {
        vector<Mat> histograms = data[i];
        stream << labels[i];

        for (int k = 0; k < histograms.size(); ++k)
        {
            Mat descriptor = histograms[k];
            int len = descriptor.cols;
            double val;
    
            for(int j = 0; j < len; ++j)
            {
                val = descriptor.at<float>(0, j);

                if (val == 0) continue;

                stream << " " << (len * k + j + 1) << ":" << val;
            }

        }       
        stream << endl;
    }

}

int main (int argc, const char** argv) 
{

    int K_values[4] = { 100, 500, 1000, 1500 };

    stringstream filename;

    Ptr<FeatureDetector > detector(new SiftFeatureDetector());
    Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<float> >());
    Ptr<DescriptorExtractor> extractor(new
            SiftDescriptorExtractor());
    Ptr<BOWImgDescriptorExtractor> bowide(new BOWImgDescriptorExtractor(extractor,matcher));

    FileStorage f("kmeans.yml", FileStorage::READ);

    stringstream param_name;
    vector<KeyPoint> keypoints;

    vector<int> labels;

    Mat histogram;
    Mat img, grayimg;

    for (int k_idx = 0; k_idx < 4; k_idx++) 
    {
        Mat vocabulary;

        param_name << "kmeans" << K_values[k_idx];
        f[param_name.str()] >> vocabulary;
        param_name.str(std::string());

        bowide->setVocabulary(vocabulary);

        vector< vector<Mat> > histograms;

        for (int i = 0; i < dog_size; i++) 
        {
            vector<Mat> image_histograms;
            Mat histogram;

            filename << dog_folder << i << ".jpg";

            img = imread(filename.str(), CV_LOAD_IMAGE_COLOR);
            
            cvtColor(img,grayimg,CV_BGR2GRAY);
            
            filename.str(std::string());
            
            Mat mask, bgdModel, fgdModel;

            grabCut(img, mask, Rect(img.cols/10, img.rows/10, 
                        (8 * img.cols)/10, (8 * img.rows)/10), 
                        bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
            

            detector->detect(grayimg,keypoints, mask);
            bowide->compute(grayimg, keypoints, histogram);

            image_histograms.push_back(histogram);

            int hstep = grayimg.cols/2;
            int vstep = grayimg.rows/2;

            for (int m = 0; m < 2; m++) {
                for (int n = 0; n < 2; n++) {
                    Mat subimg(grayimg, cv::Rect(hstep * m, vstep * n, hstep, vstep));
                    Mat submask(mask, cv::Rect(hstep * m, vstep * n, hstep, vstep));

                    detector->detect(subimg,keypoints, submask);
                    bowide->compute(subimg, keypoints, histogram);
                    image_histograms.push_back(histogram);
                
                }
            }

            labels.push_back(1);
            histograms.push_back(image_histograms);

        }
        
        for (int i = 0; i < cat_size; i++) 
        {
            vector<Mat> image_histograms;
            Mat histogram;

            filename << dog_folder << i << ".jpg";

            img = imread(filename.str(), CV_LOAD_IMAGE_COLOR);
            cvtColor(img,grayimg,CV_BGR2GRAY);

            filename.str(std::string());
             
            Mat mask, bgdModel, fgdModel;

            grabCut(img, mask, Rect(img.cols/10, img.rows/10, 
                        (8 * img.cols)/10, (8 * img.rows)/10), 
                        bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
            

            detector->detect(grayimg,keypoints, mask);
            bowide->compute(grayimg, keypoints, histogram);

            image_histograms.push_back(histogram);

            int hstep = img.cols/2;
            int vstep = img.rows/2;

            for (int m = 0; m < 2; m++) {
                for (int n = 0; n < 2; n++) {
                    Mat subimg(grayimg, cv::Rect(hstep * m, vstep * n, hstep, vstep));
                    Mat submask(mask, cv::Rect(hstep * m, vstep * n, hstep, vstep));

                    detector->detect(subimg,keypoints, submask);
                    bowide->compute(subimg, keypoints, histogram);
                    image_histograms.push_back(histogram);
                
                }
            }

            labels.push_back(0);
            histograms.push_back(image_histograms);

        
        }
        
        param_name << out_name << "_" << K_values[k_idx];

        ofstream histograms_file;
        histograms_file.open(param_name.str().c_str());
        //save_sparse_descriptors(cout, labels, histograms);
        save_sparse_descriptors(histograms_file, labels, histograms);
        histograms_file.close();
        param_name.str(std::string());

    }


    f.release();

}
