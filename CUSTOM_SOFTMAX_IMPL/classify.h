
#ifndef NNA_DEMO_H
#define NNA_DEMO_H
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>
#include <memory.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <sys/ioctl.h>
#include <sys/mman.h>

using namespace std;
class NS_Classify
{

public:
    int Classify(float *Niz,
                 const string &label_file);

private:
    void softmax(float *input, size_t input_len);

    typedef std::pair<string, float> Prediction;
    std::vector<string> labels_;
};

#endif