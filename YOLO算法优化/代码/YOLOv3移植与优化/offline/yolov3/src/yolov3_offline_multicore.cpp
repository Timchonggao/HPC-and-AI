/*
All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#if defined(USE_MLU) && defined(USE_OPENCV)
#include <algorithm>
#include <condition_variable>  // NOLINT
#include <iomanip>
#include <iosfwd>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <thread>  // NOLINT
#include <utility>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "pipeline.hpp"
#include "data_provider.hpp"
#include "runner.hpp"
#include "yolov3_processor.hpp"
#include "common_functions.hpp"

using std::map;
using std::pair;
using std::queue;
using std::string;
using std::stringstream;
using std::thread;
using std::vector;

DEFINE_int32(dump, 1, "0 or 1, dump output images or not.");
DEFINE_string(outputdir, ".", "The directoy used to save output images");

class YoloV3DataProvider : public DataProvider {
public:
  explicit YoloV3DataProvider(const std::string& meanvalue,
                        const std::queue<std::string>& images): DataProvider(meanvalue, images) {}
  void resizeMat(const cv::Mat& sample, cv::Mat& sample_resized) {
    cv::Mat sample_temp ;
    int input_dim = this->inGeometry_.height;
    /* cv::Mat sample_resized(input_dim, input_dim, CV_8UC3, cv::Scalar(128, 128, 128)); */
    if (sample.size() != inGeometry_) {
      cv::Mat sample_init(input_dim, input_dim, CV_8UC3, cv::Scalar(128, 128, 128));
      sample_init.copyTo(sample_resized);
      // resize the raw picture and copyTo the center of a 416*416 backgroud feature map
      float img_w = sample.cols;
      float img_h = sample.rows;
      int new_w = static_cast<int>(
          img_w * std::min(static_cast<float>(input_dim) / img_w,
                           static_cast<float>(input_dim) / img_h));
      int new_h = static_cast<int>(
          img_h * std::min(static_cast<float>(input_dim) / img_w,
                           static_cast<float>(input_dim) / img_h));
      cv::resize(sample, sample_temp, cv::Size(new_w, new_h), CV_INTER_CUBIC);
      sample_temp.copyTo(sample_resized(
          cv::Range((static_cast<float>(input_dim) - new_h) / 2,
                    (static_cast<float>(input_dim) - new_h) / 2 + new_h),
          cv::Range((static_cast<float>(input_dim) - new_w) / 2,
                    (static_cast<float>(input_dim) - new_w) / 2 + new_w)));
    } else {
      sample_resized = sample;
    }
  }
};

// TODO: czr- do not typedef here
typedef DataProvider DataProviderT;
typedef YoloV3DataProvider YoloV3DataProviderT;
typedef Runner RunnerT;
typedef PostProcessor PostProcessorT;
typedef Pipeline PipelineT;

int main(int argc, char* argv[]) {
  {
    const char* env = getenv("log_prefix");
    if (!env || strcmp(env, "true") != 0) FLAGS_log_prefix = false;
  }
  ::google::InitGoogleLogging(argv[0]);
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  gflags::SetUsageMessage(
      "Do detection using yolov3 mode.\n"
      "Usage:\n"
      "    yolov3_offline_multicore [FLAGS] model_file list_file\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (argc == 0) {
    gflags::ShowUsageWithFlagsRestrict(
        argv[0], "examples/yolo_v3/yolov3_offline_multicore");
    return 1;
  }

  int provider_num = 4, postprocessor_num = 2;

  if (FLAGS_logdir != "") {
    FLAGS_log_dir = FLAGS_logdir;
  } else {
    //  log to terminal's stderr if no log path specified
    FLAGS_alsologtostderr = 1;
  }

  // get device ids
  std::stringstream sdevice(FLAGS_mludevice);
  vector<int> deviceIds_;
  std::string item;
  while (getline(sdevice, item, ',')) {
    int device = std::atoi(item.c_str());
    deviceIds_.push_back(device);
  }
  int totalThreads = FLAGS_threads * deviceIds_.size();

  cnrtInit(0);
  OfflineDescripter* offline_descripter = new OfflineDescripter(deviceIds_);
  offline_descripter->loadOfflinemodel(FLAGS_offlinemodel,
                                   FLAGS_channel_dup,
                                   FLAGS_threads);

  ImageReader img_reader(FLAGS_dataset_path, FLAGS_images, totalThreads * provider_num);
  auto&& imageList = img_reader.getImageList();
  int imageNum = img_reader.getImageNum();
  if (imageNum < totalThreads)
    totalThreads = imageNum;
  
  if (FLAGS_perf_mode) {
    LOG(INFO) << "[INFO] Using performance mode";
    // calculate number of fake image per thread
    FLAGS_perf_mode_img_num = FLAGS_perf_mode_img_num / (totalThreads * provider_num);
  }

  std::vector<std::thread*> stageThreads;
  std::vector<PipelineT*> pipelines;
  std::vector<DataProviderT*> providers;
  std::vector<PostProcessorT*> postprocessors;
  for (int i = 0; i < totalThreads; i++) {
    DataProviderT* provider;
    RunnerT* runner;
    PipelineT* pipeline;
    PostProcessorT* postprocessor;

    providers.clear();  // clear vector of last thread
    postprocessors.clear();
    for (int j = 0; j < provider_num; j++) {
      provider = new YoloV3DataProviderT(FLAGS_meanvalue,
                                      imageList[provider_num * i + j]);
      providers.push_back(provider);
    }

    for (int j = 0; j < postprocessor_num; j++) {
      postprocessor = new YoloV3Processor();
      postprocessors.push_back(postprocessor);
    }

    runner = new RunnerT(offline_descripter, i);
    pipeline = new PipelineT(providers, runner, postprocessors);

    stageThreads.push_back(new std::thread(&PipelineT::runParallel, pipeline));
    pipelines.push_back(pipeline);
  }

  Timer timer;
  for (int i = 0; i < stageThreads.size(); i++) {
    pipelines[i]->notifyAll();
  }
  
  for (int i = 0; i < stageThreads.size(); i++) {
    stageThreads[i]->join();
    delete stageThreads[i];
  }
  timer.log("Total execution time");

  float mluTime = 0;
  for (auto pipeline : pipelines) {
    mluTime += pipeline->runner()->runTime();
  }
  int batch_size = pipelines[0]->runner()->offlineDescripter()->inN();
  std::vector<InferenceTimeTrace> timetraces;
  for (auto iter : pipelines) {
    for (auto pP : iter->postProcessors()) {
      for (auto tc : pP->timeTraces()) {
        timetraces.push_back(tc);
      }
    }
  }
  printPerfTimeTraces(timetraces, batch_size, mluTime);
  saveResultTimeTrace(timetraces, (-1), (-1), (-1), imageNum, batch_size, mluTime);

  for (auto pipeline : pipelines)
    delete pipeline;

  cnrtDestroy();
}

#else
// TODO: remove caffe here
#include "caffe/common.hpp"
int main(int argc, char* argv[]) {
  LOG(FATAL) << "This program should be compiled with the defintion"
             << " of both USE_MLU and USE_OPENCV!";
  return 0;
}
#endif  // USE_MLU
