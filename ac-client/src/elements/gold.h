#pragma once
#include <ImageRegion.h>
#include <ocr.h>
#include <opencv2/core.hpp>

class Gold {
public:
  Gold(ocr&);

  ImageRegion goldRegion;

  void updateImageRegion(const cv::Mat& fullImg);

  cv::Rect getBounds();

  std::string getText();
  
  void locate(ImageRegion fullShop, ImageRegion middleSlot);

  void testIRVitals(cv::Mat);

private:
  const cv::Scalar borderMin = cv::Scalar(17, 15, 180);
  const cv::Scalar borderMax = cv::Scalar(26, 35, 250);
  const double numAreaMin = 0.0002;
  const double numAreaMax = 0.13;
  const double increaseBorder = 10;

  ocr& readNum;

  cv::Mat whiteNumMask(const cv::Mat&);

  std::vector<cv::Rect> createNumBox(cv::Mat);

  cv::Rect findBoxesPerim(std::vector<cv::Rect>, int padding = 5);
  
};
