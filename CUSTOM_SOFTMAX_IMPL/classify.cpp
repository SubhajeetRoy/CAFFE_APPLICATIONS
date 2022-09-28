
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iomanip>
#include "classify.h"
#include "demo.h"
extern int verbose_flag;
extern Utils utils;
extern int savebin_flag ;
static bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs)
{
    return lhs.first > rhs.first;
}
void NS_Classify::softmax(float *input, size_t input_len)
{
    assert(input);
    // assert(input_len >= 0);  Not needed

    float m = -INFINITY;
    for (size_t i = 0; i < input_len; i++)
    {
        if (input[i] > m)
        {
            m = input[i];
        }
    }

    float sum = 0.0;
    for (size_t i = 0; i < input_len; i++)
    {
        sum += expf(input[i] - m);
    }

    float offset = m + logf(sum);
    for (size_t i = 0; i < input_len; i++)
    {
        input[i] = expf(input[i] - offset);
    }
}

/*Function : Argmax
*Return the indices of the top N values of vector v. 
Argmax is most commonly used in machine learning for finding the class with the largest predicted probability
*/
static std::vector<int> Argmax(const std::vector<float> &v, int N)
{
    std::vector<std::pair<float, int>> pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare); //https://www.geeksforgeeks.org/stdpartial_sort-in-cpp/

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}
/**Function :Classify
 * Args: float Niz--F32 output of the network
 *       label_file: Path of the file where labels are stored
 */

int NS_Classify::Classify(float *Niz, const string &label_file)
{
    std::vector<float> out_softmax;

    std::ifstream labels(label_file.c_str());
    //read the labels file synset_words.txt and store each line in a vector
    string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));

    float sumary = 0;
    softmax(Niz, 1000); //run softmax algoritm on the F32 output array
    if (verbose_flag == 1)
        printf("OUTPUT FROM CAFFE CUSTOM SOFTMAX\n");
    for (int j = 0; j < 1000; j++)
    {
        if (verbose_flag == 1)
            printf("%f ", Niz[j]);
        sumary += Niz[j];              //add the array elements to check if they add upto 1
        out_softmax.push_back(Niz[j]); //save the output of softmax in the vector
    }
    if (verbose_flag == 1)
        printf("\n\nSum of Softmax Probilities=%f\n", sumary); //check if the sum adds up to 1 in total

    if (savebin_flag)
    {
        string name;
        utils.makefileName("customsoftmax_output", name);
        cout << endl
              << " << " << name << endl;
        utils.WriteToFile(Niz, name.c_str());
    }
    int N = 5; //number of classes we want to print as result.
    //Argmax is most commonly used in machine learning for finding the class with the largest predicted probability
    std::vector<int> maxN = Argmax(out_softmax, N);

    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i)
    {
        // printf("\nSum of Softmax =%f\n", sumary);
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], out_softmax[idx]));
    }
    if (verbose_flag == 1)
        cout << "\n############ predictions ##################\n"
             << endl;
    //print the probability of top 2 classes
    for (size_t i = 0; i < predictions.size(); ++i)
    {
        Prediction p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                  << p.first << "\"" << std::endl;
    }
    return 0;
}
