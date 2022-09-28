# Example BVLC C++ classifier application using caffe

This application uses an already trained model from Caffe model Zoo called
https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
Which is licensed for unrestricted use .
The model is trained to classify 1000 objects in an image .
Description :
1) This sample first uses caffe_example/models/deploy.prototxt
Which has a Softmax layer defined as output layer and prints the probabilities
2) Then the sample uses/caffe_example/models/deploy_no_softmax.prototxt which has its
softmax layer commented out and uses custom softmax function to print the probabilities
.

## To Compile and run the application

1. Go to folder caffe_example_catsAnddogs

2. Change the path in CMakeList.txt to your respective caffe install directory

  ![]()

  ● Run cmake CMakeList.txt
  ● makeTo classify a local image and compare probabities
  Type  ./DEMO --image <image.jpg>
  Below is a sample output from a sample image prediction

3. First application prints the output probability using Softmax layer within Caffe

4. Then the application will print the probability using FC8 output and then performing a
  custom Softmax on the outputOptional Arguments :
  --verbose:
  Use command  ./DEMO --image <image.jpg> --verbose
  This command will print every step and also print the binary outputs of below float
  buffers :
  When using using Softmax layer of Caffe
  ● prob Layer output
  When using using FC8 layer of Caffe and performing custom Softmax
  ● Fc8 layer output
  ● Softmax function output
  Below command can also be used to save the logs in a text file and for later viewing.
  ./DEMO --image <image.jpg>  2>&1 | tee ./log.txt
  --savebin:
  ./DEMO --image <image.jpg> --savebin
  This option is optional and will save the output as a binary file .
  Below are the files it will generate .
  ● <imagename>prob_output.bin:prob Layer output When using using Softmax
  layer of Caffe
  ● <imagename>fc8_output.bin:Fc8 layer output
  ● <imagename>customsoftmax_output.bin:Custom Softmax function outputResults :
  There was no major difference observed between the output of the Caffe Softmax layer
  and Custom Softmax function .