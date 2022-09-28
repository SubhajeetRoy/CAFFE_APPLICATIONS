#include "demo.h"

Utils utils;
int verbose_flag = 0;
int savebin_flag = 0;

Classifier::Classifier(const string &model_file,
                       const string &trained_file,
                       const string &mean_file,
                       const string &label_file)
{
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  for (size_t i = 0; i < 1000; ++i)
  {
    fbuf[i] = 0.0;
  }

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float> *input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
      << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float> *output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
      << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int> &lhs,
                        const std::pair<float, int> &rhs)
{
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float> &v, int N)
{
  std::vector<std::pair<float, int>> pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat &img, int N)
{
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i)
  {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string &mean_file)
{
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float *data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i)
  {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat &img)
{
  Blob<float> *input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float> *output_layer = net_->output_blobs()[0];
  const float *begin = output_layer->cpu_data();
  const float *end = begin + output_layer->channels();
  float *start;
  start = (float *)begin;
  std::vector<string> name = net_->layer_names();
  for (int layer_index = 0; layer_index < name.size(); layer_index++)
    printf("Layer [%d] =%s \n", layer_index, name[layer_index].c_str());
  const vector<int> output_index = net_->output_blob_indices();
  printf("\nOUTPUT LAYER NAME = %s\n\n", net_->blob_names()[output_index[0]].c_str());

  if (verbose_flag == 1)
    printf("\n##############################  OUTPUT FROM CAFFE SOFTMAX %s Layer############################## \n", net_->blob_names()[output_index[0]].c_str());
  int i = 0;
  while (start != end)
  {
    if (verbose_flag == 1)
      printf("%f ", *start);
    if (savebin_flag)
    {
      fbuf[i] = *start;
    }
    start++;
    i++;
  }
  printf("\n");
  if (savebin_flag)
  {

    string name;
    utils.makefileName("prob_output", name);
    cout << endl
         << " << " << name << endl;
    utils.WriteToFile(fbuf, name.c_str());
  }

  return std::vector<float>(begin, end);
}

float *Classifier::GetoutputLogits(const cv::Mat &img, string &label_file)
{
  Blob<float> *input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float> *output_layer = net_->output_blobs()[0];
  const float *begin = output_layer->cpu_data();
  const float *end = begin + output_layer->channels();
  float *start;
  start = (float *)begin;

  std::vector<string> name = net_->layer_names();
  //vector<int> output_layer_inidices = net_->output_blob_indices() ;
  for (int layer_index = 0; layer_index < name.size(); layer_index++)
    printf("Layer [%d] =%s \n", layer_index, name[layer_index].c_str());
  const vector<int> output_index = net_->output_blob_indices();
  printf("\nOUTPUT LAYER NAME = %s\n\n", net_->blob_names()[output_index[0]].c_str());

  int i = 0;
  if (verbose_flag == 1)
    printf("\n############################## OUTPUT FROM CAFFE %s Layer #################################\n", net_->blob_names()[output_index[0]].c_str());
  while (start != end)
  {
    if (verbose_flag == 1)
      printf("%f ", *start);
    fbuf[i] = *start;
    start++;
    i++;
  }
  printf("\n");
  if (savebin_flag)
  {
    string name;
    utils.makefileName("fc8_output", name);
    cout << endl
         << " << " << name << endl;
    utils.WriteToFile(fbuf, name.c_str());
  }
  return fbuf;
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat> *input_channels)
{
  Blob<float> *input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float *input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i)
  {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat &img,
                            std::vector<cv::Mat> *input_channels)
{
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float *>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
      << "Input channels are not wrapping the input layer of the network.";
}
void Utils::Usage(char **argv)
{
  printf("Usage: \n");
  printf("For  classification Use: %s  --image <imagepath> \n", argv[0]);
  printf("optional arguments :\n--vebose : Enable verbose mode (NOT enabled by default)\n");
  printf("--savebin : Will save the output of prob layer ,fc8 Layer and custom softmax Layer in .bin files \n");
}
int Utils::WriteToFile(float *f, const char *name)
{
  
  std::ofstream out;
  //printf("\n\n After precision \n\n ");
  for (int i = 0; i < 1000; i++){
    f[i] = floor(pow(10,6)*f[i])/pow(10,6);  //set precision to 6 digits after decimal
  // printf(" %f",f[i]);
  }
  printf("\n");
  out.open(name, std::ios::out | std::ios::binary);
  out.write(reinterpret_cast<const char *>(f), 1000*sizeof(float));
  out.close();
  return 0;
}
void Utils::makefileName(string token, string &newName)
{

  string s = imagePath;
  string::size_type i = s.rfind('.', s.length());

  if (i != string::npos)
  {
    s.replace(i, token.length(), token);
    s.append(".bin");
  }
  newName = s;
}
int main(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0]
              << " ImageFile.jpg"
              << std::endl;
    return 1;
  }

  int cc;

  if (argc < 3) //there shsould be atleast 3 arguments or else exit the app
  {
    utils.Usage(argv);

    return 1;
  }

  while (1)
  {
    static struct option long_options[] =
        {
            {"verbose", no_argument, &verbose_flag, 1},
            {"savebin", no_argument, &savebin_flag, 1},
            {"image", required_argument, 0, 'i'},

            {0, 0, 0, 0}};
    /* getopt_long stores the option index here. */
    int option_index = 0;

    cc = getopt_long(argc, argv, "i:",
                     long_options, &option_index);

    /* Detect the end of the options. */
    if (cc == -1)
      break;

    switch (cc)
    {
    case 0:
      if (long_options[option_index].flag != 0)
      {
        printf("break here\n");
        break;
      }
      printf("-option %s", long_options[option_index].name);
      if (optarg)
        printf("- with arg %s", optarg);
      printf("\n");
      break;

    case 'i':
      printf("option -i with value `%s'\n", optarg);
      strcpy(utils.imagePath, optarg);
      break;

    case '?':
      printf("unrecognized option....\n");
      /* getopt_long already printed an error message. */
      break;

    default:
      printf("ABORT....\n");
      abort();
    }
  }

  printf("verbose =%d imagePath=%s savebin_flag=%d \n", verbose_flag, utils.imagePath, savebin_flag);
  if (verbose_flag == 0)
    ::google::InitGoogleLogging(argv[0]);
  {
    string model_file = "./models/deploy.prototxt";
    string trained_file = "./models/bvlc_reference_caffenet.caffemodel";
    string mean_file = "./data/imagenet_mean.binaryproto";
    string label_file = "./data/synset_words.txt";
    Classifier classifier(model_file, trained_file, mean_file, label_file);

    string file = utils.imagePath;

    std::cout << "---------- Prediction of Softmax using Caffe for image= "
              << file << " ----------" << std::endl;

    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;

    std::vector<Prediction> predictions = classifier.Classify(img);

    /* Print the top N predictions. */
    for (size_t i = 0; i < predictions.size(); ++i)
    {
      Prediction p = predictions[i];
      std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                << p.first << "\"" << std::endl;
    }
  }
  {

    string file = utils.imagePath;
    std::cout << std::endl
              << "\n\n---------- Prediction of Softmax using Custom softmax implementation for image="
              << file << " ----------" << std::endl;
    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;

    string model_file = "./models/deploy_no_softmax.prototxt";
    string trained_file = "./models/bvlc_reference_caffenet.caffemodel";
    string mean_file = "./data/imagenet_mean.binaryproto";
    string label_file = "./data/synset_words.txt";
    Classifier classifier(model_file, trained_file, mean_file, label_file);
    float *fc8_layer;
    fc8_layer = classifier.GetoutputLogits(img, label_file);
    NS_Classify c;
    c.Classify(fc8_layer, label_file);
  }
}
