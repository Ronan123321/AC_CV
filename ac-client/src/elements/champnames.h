#pragma once
#include <opencv2/core.hpp>
#include "ImageRegion.h"
#include <ocr.h>
#include <opencv2/core/mat.hpp>
#include <string>

class ChampNames {
public:  
  ChampNames(ocr& ocr);

  std::vector<ImageRegion> champNames;

  void updateImageRegion(const cv::Mat& fullImg);

  std::string getNameStr(int index);

  std::vector<std::string> getAllNames();

  cv::Rect getBound(int index);

  std::vector<cv::Rect> getAllBounds();

  void locate(std::vector<ImageRegion> champSlots);

  void testIRVals(cv::Mat);

private:
  const std::string goldTemplatePath = "D:/linux_img_port/Gold_Icon.png";
  const cv::Scalar traitMin = cv::Scalar(0, 0, 31);
  const cv::Scalar traitMax = cv::Scalar(5, 5, 60);
  const cv::Scalar goldMin = cv::Scalar(18, 110, 195); // Is acutlly the gold icon(which is gold color)
  const cv::Scalar goldMax = cv::Scalar(21, 117, 205);
  const int rowSimKernel = 4;
  const double rowMaxDiff = 3.0f;
  const double groupMaxDiff = 3.5f;
  const double minNameLength = 0.75; // percent of champ slot width

  const double traitMinArea = 0.001; // percent of full image
  const double traitMaxArea = 0.3;
  const double goldMinArea = 0.003;
  const double goldMaxArea = 0.3;
  
  ocr& readName;

  struct Interval {
	  int start;
	  int end;
  };

  std::vector<int> changePerRow(cv::Mat);

  std::vector<Interval> combineRows(std::vector<int>);

  Interval bestRows(std::vector<Interval>);

  ImageRegion sliceGoodRows(std::vector<Interval>, ImageRegion);

  cv::Rect fMatchGold(const cv::Mat&);

  std::string runOCR(cv::Mat);

  ImageRegion chopGold(ImageRegion);

  ImageRegion bottomTraitIcon(ImageRegion);

};
