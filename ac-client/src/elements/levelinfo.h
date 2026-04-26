#pragma once
#include <ocr.h>
#include <ImageRegion.h>
#include <opencv2/core/mat.hpp>

class LevelInfo {
public:
  LevelInfo(ocr& ocr);

  ImageRegion levelRegion;
  ImageRegion progressRegion;

  void updateImageRegion(const cv::Mat& fullImg);

  cv::Rect getLevelBounds();

  cv::Rect getProgBounds();

  std::string getLevelStr();

  std::string getProgStr();

  std::pair<cv::Rect, cv::Rect> getAllBounds();

  void locate(ImageRegion fullShop, ImageRegion leftShop);

  void testIRVitals(cv::Mat);

private:
  const cv::Scalar infoBorderMin = cv::Scalar(83, 60, 17);
  const cv::Scalar infoBorderMax = cv::Scalar(100, 153, 27);

  const cv::Scalar whiteTextMin = cv::Scalar(13, 11, 165);
  const cv::Scalar whiteTextMax = cv::Scalar(45, 35, 250);

  const cv::Scalar blueTextMin = cv::Scalar(85, 20, 82);
  const cv::Scalar blueTextMax = cv::Scalar(93, 60, 220);
  const double infoWidthMin = 0.75;
  const double infoHeightMin = 0.25;
  const double textMinArea = 0.001;

  ocr& readNum;
  
  ImageRegion cropTextArea(const ImageRegion&);

  cv::Mat createMask(const cv::Mat&, cv::Scalar, cv::Scalar);

  std::vector<cv::Rect> createTextBoxes(const cv::Mat&);

  cv::Rect findBoxesPerim(std::vector<cv::Rect>, int padding = 5);

  void blankPixelDistro(cv::Rect&, cv::Rect&);
};
