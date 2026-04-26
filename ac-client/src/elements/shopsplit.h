#pragma once
#include <ImageRegion.h>
#include <opencv2/opencv.hpp>

class ShopSplit {
public:
  ShopSplit() = default;

  ImageRegion left;
  ImageRegion right;

  void updateImageRegion(const cv::Mat& fullImg);

  cv::Rect getLeft();

  cv::Rect getRight();

  std::pair<cv::Rect, cv::Rect> getBoth();

  void locate(ImageRegion fullShop);

  void testIRVals(cv::Mat);

private:
  const cv::Scalar shopBorderMin = cv::Scalar(83, 81, 27);
  const cv::Scalar shopBorderMax = cv::Scalar(100, 171, 50);
  //	const cv::Scalar shopBorderMin = cv::Scalar(85, 100, 25);
  //	const cv::Scalar shopBorderMax = cv::Scalar(100, 185, 90);
  const double shopBoundMin = 0.5;
  const double epsilon = 2.0;
  const int borderSliceDistance = 50; // in pixels

  cv::Mat joinCrop(cv::Mat);

  std::vector<std::vector<cv::Point>> findBorderMask(const cv::Mat&);

  cv::Mat vertInclude(std::vector<cv::Point>, cv::Mat);

  cv::Mat findFlatMask(const cv::Mat&, std::vector<std::vector<cv::Point>>);

  std::vector<int> findSliceCords(const cv::Mat&);

  std::vector<cv::Rect> shopLeftRight(std::vector<int>, std::vector<std::vector<cv::Point>>);

  std::vector<cv::Rect> twoBiggest(std::vector<cv::Rect>);

  std::vector<int> sliceFallback(std::vector<int>);
  
};
