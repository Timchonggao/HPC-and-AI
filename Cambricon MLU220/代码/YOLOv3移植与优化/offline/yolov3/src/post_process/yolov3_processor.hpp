#ifndef CATCH_EXAMPLES_OFFLINE_SSD_POST_PROCESS_SSD_PROCESSOR_HPP_
#define CATCH_EXAMPLES_OFFLINE_SSD_POST_PROCESS_SSD_PROCESSOR_HPP_
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "glog/logging.h"
#include "post_processor.hpp"
#include "threadPool.h"

using std::map;
using std::max;
using std::min;
using std::queue;
using std::stringstream;
using std::vector;
using std::string;

class YoloV3Processor : public PostProcessor {
  public:
  YoloV3Processor() {}
  virtual ~YoloV3Processor() {}
  virtual void runParallel();
  void WriteVisualizeBBox_offline(const vector<cv::Mat>& images,
      const vector<vector<vector<float>>> detections,
      const vector<string>& labelToDisplayName, const vector<string>& imageNames,
      int input_dim, const int from, const int to);
  void readLabels(std::vector<string>& label_name);

  protected:
  std::vector<string> label_name;

  private:
    std::vector<std::vector<std::vector<float>>> getResults();
    void getImages(std::vector<cv::Mat>*imgs,
                   std::vector<std::string>*img_names,
                   std::vector<std::pair<std::string, cv::Mat>> origin_img);

    template<typename T>
    inline std::shared_ptr<T> MakeArray(int size){
        return std::shared_ptr<T>( new T[size],
            [](T *p ) {
                delete[] p;
            }
        );
    }
  private:
   std::shared_ptr<float> outCpuPtrs_;
   std::shared_ptr<float> outTrans_;
   std::shared_ptr<char> outCpuTempPtrs_;
};

#endif  // CATCH_EXAMPLES_OFFLINE_SSD_POST_PROCESS_SSD_PROCESSOR_HPP_
