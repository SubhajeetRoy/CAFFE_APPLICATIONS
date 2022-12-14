#ifndef DEMO_H
#define DEMO_H
#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "classify.h"
#include <getopt.h>
using namespace caffe;
using std::string;
typedef std::pair<string, float> Prediction;
class Classifier
{
public:
    Classifier(const string &model_file,
               const string &trained_file,
               const string &mean_file,
               const string &label_file);

    std::vector<Prediction> Classify(const cv::Mat &img, int N = 5);
    float *GetoutputLogits(const cv::Mat &img, string &label_file);

private:
    void SetMean(const string &mean_file);

    std::vector<float> Predict(const cv::Mat &img);

    void WrapInputLayer(std::vector<cv::Mat> *input_channels);

    void Preprocess(const cv::Mat &img,
                    std::vector<cv::Mat> *input_channels);

private:
    boost::shared_ptr<Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    std::vector<string> labels_;
    float fbuf[1000];
};
class Utils
{
public:
    void Usage(char **argv);
    int WriteToFile( float *f,const char *name);
    void makefileName(string token, string &newName);
    char imagePath[1024];
};
#endif