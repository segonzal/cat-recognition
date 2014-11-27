#include <iostream>
#include <fstream>
#include <iterator>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int n_ks = 1;
int K_values[1] = { 1000 };
String out_name = "train";

double w = 0.6;

String cats_file = "./cat_train_final.txt";
String dogs_file = "./dog_train_final.txt";

void save_sparse_descriptors(ostream& stream, vector<int> labels,
        vector< vector<Mat> > data) 
{

    for (int i = 0; i < data.size(); ++i)
    {
        vector<Mat> histograms = data[i];
        stream << labels[i];
        cout << labels[i];
        for (int k = 0; k < histograms.size(); ++k)
        {
            Mat descriptor = histograms[k];
            int len = descriptor.cols;
            double val;
    
            if (descriptor.rows == 0) continue;

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

    Ptr<FeatureDetector > detector(new SiftFeatureDetector());
    Ptr<FeatureDetector > detector2(new SurfFeatureDetector(400));
    Ptr<BOWImgDescriptorExtractor> bowides[n_ks];

    FileStorage f("kmeans.yml", FileStorage::READ);

    stringstream param_name;
    vector<KeyPoint> keypoints, keypoints2;
    vector<KeyPoint> skeypoints, skeypoints2;

    vector<int> labels;

    Mat img, grayimg;

    for (int k_idx = 0; k_idx < n_ks; k_idx++) 
    {
        Mat vocabulary;
        Ptr<DescriptorExtractor> extractor(new
                SiftDescriptorExtractor());
        Ptr<DescriptorMatcher > matcher(new BruteForceMatcher<L2<float> >());
        Ptr<BOWImgDescriptorExtractor> bowide(new BOWImgDescriptorExtractor(extractor,matcher));

        param_name << "kmeans" << K_values[k_idx];
        f[param_name.str()] >> vocabulary;
        param_name.str(std::string());
        bowide->setVocabulary(vocabulary);

        bowides[k_idx] = bowide;

    }

    ifstream dogfile(dogs_file.c_str());
    vector< vector<Mat> > histograms[n_ks];
    string line;
    Mat histogram, histogram2;
 
    ifstream catfile(cats_file.c_str());

    while (getline(catfile, line))
    {
        cout << line << endl;
        vector<Mat> image_histograms;
        img = imread(line, CV_LOAD_IMAGE_COLOR);
        cvtColor(img,grayimg,CV_BGR2GRAY);
        Mat mask, antimask, bgdModel, fgdModel;

        grabCut(img, mask, Rect(img.cols/10, img.rows/10, 
                    (8 * img.cols)/10, (8 * img.rows)/10), 
                    bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);

        bitwise_not(mask, antimask);

        detector->detect(grayimg,keypoints,mask);
        detector->detect(grayimg,keypoints2,antimask);

        detector2->detect(grayimg,skeypoints, mask);
        detector2->detect(grayimg,skeypoints2, antimask);

        keypoints.insert(
                keypoints.end(),
                skeypoints.begin(),
                skeypoints.end()
                );

        keypoints2.insert(
                keypoints2.end(),
                skeypoints2.begin(),
                skeypoints2.end()
                );

        for (int k_idx = 0; k_idx < n_ks; k_idx++)
        {

            bowides[k_idx]->compute(grayimg, keypoints, histogram);
            bowides[k_idx]->compute(grayimg, keypoints2, histogram2);
            image_histograms.push_back(w * histogram + (1 - w) * histogram2);

            int hstep = grayimg.cols/2;
            int vstep = grayimg.rows/2;

        /*    for (int m = 0; m < 2; m++) {
                for (int n = 0; n < 2; n++) {
                    Mat subimg(grayimg, cv::Rect(hstep * m, vstep * n, hstep, vstep));
                //    Mat submask(mask, cv::Rect(hstep * m, vstep * n, hstep, vstep));


                    detector->detect(subimg,keypoints);
                    bowides[k_idx]->compute(subimg, keypoints, histogram);
                    image_histograms.push_back(histogram);
                
                }
            }*/

            labels.push_back(1);
            cout << "fef" << endl;
            histograms[k_idx].push_back(image_histograms);
        }
    
    }
    
    cout << "Finished Cats" << endl;

    while (getline(dogfile, line))
    {
        cout << line << endl;
        vector<Mat> image_histograms;
        img = imread(line, CV_LOAD_IMAGE_COLOR);
        cvtColor(img,grayimg,CV_BGR2GRAY);
        Mat mask, antimask, bgdModel, fgdModel;

        grabCut(img, mask, Rect(img.cols/10, img.rows/10, 
                    (8 * img.cols)/10, (8 * img.rows)/10), 
                   bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);

        bitwise_not(mask, antimask);

        detector->detect(grayimg,keypoints, mask);
        detector->detect(grayimg,keypoints2, antimask);

        detector2->detect(grayimg,skeypoints, mask);
        detector2->detect(grayimg,skeypoints2, antimask);

        keypoints.insert(
                keypoints.end(),
                skeypoints.begin(),
                skeypoints.end()
                );

        keypoints2.insert(
                keypoints2.end(),
                skeypoints2.begin(),
                skeypoints2.end()
                );

        for (int k_idx = 0; k_idx < n_ks; k_idx++)
        {

            bowides[k_idx]->compute(grayimg, keypoints, histogram);
            bowides[k_idx]->compute(grayimg, keypoints2, histogram2);
            image_histograms.push_back(w * histogram + (1 - w) * histogram2);

            int hstep = grayimg.cols/2;
            int vstep = grayimg.rows/2;

        /*    for (int m = 0; m < 2; m++) {
                for (int n = 0; n < 2; n++) {
                    Mat subimg(grayimg, cv::Rect(hstep * m, vstep * n, hstep, vstep));
                  //  Mat submask(mask, cv::Rect(hstep * m, vstep * n, hstep, vstep));

                    detector->detect(subimg, keypoints);
                    bowides[k_idx]->compute(subimg, keypoints, histogram);

                    image_histograms.push_back(histogram);
                
                }
            }*/

            labels.push_back(0);
            histograms[k_idx].push_back(image_histograms);
        }
    
    }

    cout << "Finished dogs" << endl;
   
    for (int k_idx = 0; k_idx < n_ks; k_idx++)
    {
        param_name << out_name << "_" << K_values[k_idx];

        ofstream histograms_file;
        histograms_file.open(param_name.str().c_str());
        //save_sparse_descriptors(cout, labels, histograms);
        save_sparse_descriptors(histograms_file, labels, histograms[k_idx]);
        histograms_file.close();
        param_name.str(std::string());
        f.release();
    }

}
