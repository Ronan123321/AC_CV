#pragma once
#include "ImageRegion.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

class ChampSlots {
public:

  std::vector<ImageRegion> champSlots;

  void updateImageRegion(const cv::Mat& fullImg);

  cv::Rect getChampSlot(int index);

  std::vector<cv::Rect> getAllChampSlot();

  void locate(ImageRegion rightShop);

  void testIRVals(cv::Mat fullImg);
  
private:
  //const cv::Scalar champBorderMin = cv::Scalar(85, 64, 20);
  //const cv::Scalar champBorderMax = cv::Scalar(105, 155, 40);
  const cv::Scalar champBorderMin = cv::Scalar(85, 64, 20);
  const cv::Scalar champBorderMax = cv::Scalar(104, 153, 40);
  const double boundRatio = 0.731; // height to width ratio
  const double ratioOffset = 0.1;
  const double minArea = 0.02; // multiplier of full area
  const double maxArea = 0.3;
  const double epsilon = 0.5;
  
  cv::Mat createMask(cv::Mat);

  cv::Mat approxMask(std::vector<cv::Point>, cv::Mat);

  std::vector<cv::Rect> createContours(cv::Mat);
  
};
