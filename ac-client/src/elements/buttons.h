#pragma once
#include <opencv2/core.hpp>
#include <ImageRegion.h>

class Buttons { 
public:
  Buttons();
  
  ImageRegion rerollRegion;
  ImageRegion levelRegion;

  void updateImageRegion(const cv::Mat& fullImg);

  std::pair<cv::Rect, cv::Rect> getBounds(); // TODO has to be changed to acoomodate two bounds

  std::pair<cv::Scalar, cv::Scalar> getScalarBounds(); // TODO has to be changed to accomodate two bounds

  void locate(ImageRegion leftShop);

  cv::Mat drawOnImg(cv::Mat fullImg, bool save);

private:
  const cv::Scalar borderMin = cv::Scalar(87, 100, 19);
  const cv::Scalar borderMax = cv::Scalar(103, 180, 40);
  const float minSize = 0.2; // 1 is full image
  const float maxSize = 0.8;
  const int defaultPadding = 10; // padding for the default bounding rects

  cv::Mat createInverseMask(cv::Mat);

  std::vector<cv::Rect> createBoundingRects(cv::Mat);

  std::vector<cv::Rect> defaultBoundingRects(cv::Mat);
};
