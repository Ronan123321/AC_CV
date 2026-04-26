#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

// TODO transfer preprocesssing to seperate class

class ocr {
// Abstract class
public:
  virtual std::string readText(cv::Mat) = 0;

  virtual std::string readNum(cv::Mat) = 0;

  virtual std::string readFrac(cv::Mat) = 0;

  virtual cv::Mat preprocessWhite(const cv::Mat&) = 0;

  virtual cv::Mat preprocessBlue(const cv::Mat&) = 0;
  
  virtual ~ocr() = default;
};
