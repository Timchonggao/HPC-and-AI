#include "glog/logging.h"
#include <queue>
#include <string>
#include <sstream>
#include <thread> // NOLINT
#include <utility>
#include <vector>
#include <iomanip>
#ifdef USE_MLU
#include "cnrt.h" // NOLINT
#include "yolov3_processor.hpp"
#include "runner.hpp"
#include "command_option.hpp"
#include "common_functions.hpp"

using std::pair;
using std::vector;
using std::string;
using std::pair;

std::mutex PostProcessor::post_mutex_;

void YoloV3Processor::WriteVisualizeBBox_offline(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>> detections,
    const vector<string>& labelToDisplayName, const vector<string>& imageNames,
    int input_dim, const int from, const int to) {
  // Retrieve detections.
  for (int i = from; i < to; ++i) {
    if (imageNames[i] == "null") continue;
    cv::Mat image;
    image = images[i];
    vector<vector<float>> result = detections[i];
    std::string name = imageNames[i];
    int positionMap = imageNames[i].rfind("/");
    if (positionMap > 0 && positionMap < imageNames[i].size()) {
      name = name.substr(positionMap + 1);
    }
    positionMap = name.find(".");
    if (positionMap > 0 && positionMap < name.size()) {
      name = name.substr(0, positionMap);
    }
    string filename = name + ".txt";
    std::ofstream fileMap(FLAGS_outputdir + "/" + filename);
    float img_max_shape = (image.cols < image.rows)?image.rows:image.cols;
    float pad_x = (image.rows > image.cols)?(image.rows - image.cols):0;
    float pad_y = (image.rows < image.cols)?(image.cols - image.rows):1;
    float img_size = 416;
    pad_x = pad_x * (img_size/img_max_shape);
    pad_y = pad_y * (img_size/img_max_shape);
    float unpad_h = img_size - pad_y;
    float unpad_w = img_size - pad_x;
    for (int j = 0; j < result.size(); j++) {
      float box_h = ((result[j][3] - result[j][1])/unpad_h) * image.rows;
      float box_w = ((result[j][2] - result[j][0])/unpad_w) * image.cols;
      float x0 =((result[j][0] - std::floor(pad_x/2))/unpad_w) * image.cols;
      float y0 =((result[j][1] - std::floor(pad_y/2))/unpad_h) * image.rows;

      float x1 = x0 + box_w;
      float y1 = y0 + box_h;
      cv::Point p1(x0, y0);
      cv::Point p2(x1, y1);
      cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 2);
      stringstream ss;
      ss << round(result[j][5] * 1000) / 1000.0;
      std::string str =
          labelToDisplayName[static_cast<int>(result[j][6])] + ":"+ss.str();
      cv::Point p5(x0, y0 + 10);
      cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(240, 240, 0), 2);

      fileMap << labelToDisplayName[static_cast<int>(result[j][6])]
              << " " << ss.str()
              << " " << x0 / image.cols
              << " " << y0 / image.rows
              << " " << x1 / image.cols
              << " " << y1 / image.rows
              << " " << image.cols
              << " " << image.rows << std::endl;
    }
    fileMap.close();
    if (FLAGS_dump) {
      stringstream ss;
      string outFile;
      ss << FLAGS_outputdir << "/yolov3_offline_" << name << ".jpg";
      ss >> outFile;
      cv::imwrite((outFile.c_str()), image);
    }
  }
}

void YoloV3Processor::readLabels(std::vector<string>& label_name) {
  if (!FLAGS_labels.empty()) {
    std::ifstream labels(FLAGS_labels);
    string line;
    while (std::getline(labels, line)) {
      label_name.push_back(line);
    }
    labels.close();
  }
}

void YoloV3Processor::runParallel() {
  setDeviceId(deviceId_);

  this->readLabels(this->label_name);

  int dim_order[4] = {0, 3, 1, 2};
  // TODO: czr-it's confusing that a micro is introduced to calculate tensor size. why?
  cnrtDataType_t* output_data_type = this->offlineDescripter()->getOutputDataType();
  int outputNum = this->offlineDescripter()->outputNum();
  auto outCount = this->offlineDescripter()->outCount();
  int temp_output_size = GET_OUT_TENSOR_SIZE(output_data_type[0], outCount);
  // yolov3 has single output, prepare data and shape outside of for loop.
  auto output_shape = this->offlineDescripter()->getOutputShape();
  int dim_shape[4] = {static_cast<int>(output_shape[0]), /*n*/
                      static_cast<int>(output_shape[2]), /*h*/
                      static_cast<int>(output_shape[3]), /*w*/
                      static_cast<int>(output_shape[1])}; /*c*/

  outCpuPtrs_ = MakeArray<float>(outCount);
  outTrans_ = MakeArray<float>(outCount);
  outCpuTempPtrs_ = MakeArray<char>(temp_output_size);

  // TODO: Hardcode thread_num to 4 here. Needs to wrap it in a class
  int TASK_NUM = 4;
  zl::ThreadPool tp(TASK_NUM);
  while (true) {
    Timer postProcess;
    std::unique_lock<std::mutex> lock(post_mutex_);
    auto output_boxing_data = this->offlineDescripter()->popValidOutputBoxingData(this->deviceId_);
    if (output_boxing_data == nullptr) {
      lock.unlock();
      break;  // no more data to process
    }
    auto mluOutData = output_boxing_data->getBuf();
    auto origin_img = output_boxing_data->getImageAndName();
    lock.unlock();

    TimePoint t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < outputNum; i++) {
      cnrtMemcpy(outCpuTempPtrs_.get(),
                 mluOutData.get()[i],
                 this->offlineDescripter()->outputSizeS[i],
                 CNRT_MEM_TRANS_DIR_DEV2HOST);

      if (output_data_type[i] != CNRT_FLOAT32) {
        cnrtCastDataType(outCpuTempPtrs_.get(),
                         output_data_type[i],
                         outTrans_.get(),
                         CNRT_FLOAT32,
                         outCount,
                         nullptr);
      } else {
        memcpy(outTrans_.get(),
               outCpuTempPtrs_.get(),
               this->offlineDescripter()->outputSizeS[i]);
      }

      cnrtTransDataOrder(outTrans_.get(),
                         CNRT_FLOAT32,
                         outCpuPtrs_.get(),
                         4,
                         dim_shape,
                         dim_order);
    }
    TimePoint t2 = std::chrono::high_resolution_clock::now();
    auto time_stamp = output_boxing_data->getStamp();
    time_stamp->out_start = t1;
    time_stamp->out_end = t2;
    this->appendTimeTrace(*time_stamp);
    this->offlineDescripter()->pushFreeOutputBoxingData(output_boxing_data, this->deviceId_);
		
    if (FLAGS_perf_mode) {
      postProcess.log("post process time ...");
      continue;
    }
   
    auto final_boxes = getResults();

    vector<cv::Mat> imgs;
    vector<string> img_names;
    getImages(&imgs, &img_names, origin_img);

    int input_dim = FLAGS_yuv? 416: this->offlineDescripter()->inH();
    const int size = imgs.size();
    if (TASK_NUM > size) {
        TASK_NUM = size;
    }
    const int delta = size / TASK_NUM;
    int from = 0;
    int to = delta;
    for (int i = 0; i < TASK_NUM; i++) {
        from = delta * i;
        if (i == TASK_NUM - 1) {
            to = size;
        } else {
            to = delta * (i + 1);
        }
        tp.add([](const vector<cv::Mat>& imgs,
                    const vector<vector<vector<float>>>& final_boxes,
                    const vector<string>& label_name,
                    const vector<string>& img_names,
                    const int input_dim, const int& from, const int& to,
                    YoloV3Processor* object) {
                object->WriteVisualizeBBox_offline(imgs, final_boxes,
                        label_name, img_names, input_dim, from, to);
                }, imgs, final_boxes, this->label_name, img_names, input_dim, from, to, this);
    }
    postProcess.log("post process time ...");
  }
}

vector<vector<vector<float>>> YoloV3Processor::getResults() {

    std::shared_ptr<float> data = outCpuPtrs_;
    vector<vector<vector<float>>> final_boxes(this->offlineDescripter()->inN());

    int batch_size = this->offlineDescripter()->outN();
    int per_batch_size = this->offlineDescripter()->outCount() / batch_size;
    int numBoxFinal = 0;
    int imgSize = 416;
    float max_limit = 1;
    float min_limit = 0;

    for (int i = 0; i < batch_size; i++) {
      numBoxFinal = static_cast<int>(data.get()[i*per_batch_size]);
      for (int k = 0; k < numBoxFinal ;  k++) {
        vector<float> single_box;
        int batchNum = data.get()[i * per_batch_size + 64 + k * 7];
        if ((batchNum < 0) || (batchNum >= batch_size)) {
          continue;
        }
        float bl = std::max(min_limit,
                            std::min(max_limit,
                                     data.get()[i * per_batch_size + 64 + k * 7 + 3]) * imgSize);
        float br = std::max(min_limit,
                            std::min(max_limit,
                                     data.get()[i * per_batch_size + 64 + k * 7 + 4]) * imgSize);
        float bt = std::max(min_limit,
                            std::min(max_limit,
                                     data.get()[i * per_batch_size + 64 + k * 7 + 5]) * imgSize);
        float bb = std::max(min_limit,
                            std::min(max_limit,
                                     data.get()[i * per_batch_size + 64 + k * 7 + 6]) * imgSize);
        single_box.push_back(bl);
        single_box.push_back(br);
        single_box.push_back(bt);
        single_box.push_back(bb);
        single_box.push_back(data.get()[i * per_batch_size + 64 + k * 7 + 2]);
        single_box.push_back(data.get()[i * per_batch_size + 64 + k * 7 + 2]);
        single_box.push_back(data.get()[i * per_batch_size + 64 + k * 7 + 1]);
        if ((bt - bl) > 0 && (bb - br) > 0) {
          final_boxes[batchNum].push_back(single_box);
        }
      }
    }
    return final_boxes;
}

void YoloV3Processor::getImages(
    vector<cv::Mat>*imgs,
    vector<string>*img_names,
    vector<pair<string, cv::Mat>> origin_img) {
    for (auto img_name : origin_img) {
      if (img_name.first != "null") {
        cv::Mat img;
        if (FLAGS_yuv) {
          cv::Size size = cv::Size(this->offlineDescripter()->inW(),
                                   this->offlineDescripter()->inH());
          img = yuv420sp2Bgr24(convertYuv2Mat(img_name.first, size));
        } else {
          img = img_name.second;;
        }
        imgs->push_back(img);
        img_names->push_back(img_name.first);
      }
    }
}

#endif
