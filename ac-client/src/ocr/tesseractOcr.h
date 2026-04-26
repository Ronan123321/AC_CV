#pragma once
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "ocr.h"

class tesseractOcr : public ocr {
  tesseract::TessBaseAPI api;
public:
  
  std::string readText(cv::Mat) override;

  std::string readNum(cv::Mat) override;

  std::string readFrac(cv::Mat) override;

  cv::Mat preprocessWhite(const cv::Mat&) override;

  cv::Mat preprocessBlue(const cv::Mat&) override;
  
  int levenshteinDistance(const std::string&, const std::string&);

private:
	const std::vector<int> levelDenom = { 0, 0, 2, 6, 10, 20, 36, 48, 72, 84 };

  std::string verifySlash(const std::string&);

  std::string insertSlash(const std::string&);

  cv::Mat dilateMask(const cv::Mat&);

  void verifyGrayscale(cv::Mat&);
};

