#pragma once
#include "champnames.h"
#include <opencv2/core/types.hpp>
#include <shopsplit.h>
#include <buttons.h>
#include <champslots.h>
#include <gold.h>
#include <levelinfo.h>
#include <ImageRegion.h>
#include <tesseractOcr.h>

class Shop {

  ShopSplit shopSplit;       // needs shop
  Buttons findButtons;       // Needs left side of shop
  ChampSlots findChampslots; // Needs right side of shop
  ChampNames findChampNames; // Needs champ slots
  Gold findGold;             // Needs champ slots
  LevelInfo findLevelinfo;   // Needs left side of shop

  tesseractOcr tesseractOCR;
public:
  
  Shop();

  ImageRegion region;

  void locate(ImageRegion fullScreen, cv::Mat shopTemplate); // only locates shop

  void locateShopELements(ImageRegion fullScreen, cv::Mat shopTemplate); // Locates shop and elements

  void relocateElements(); // locates elements based on the current region

  void relocateElements(ImageRegion shop); // relocates elements based on a given shopregion

  void updateFrame(const cv::Mat&);

  std::vector<cv::Rect> getAllBounds();

  std::vector<cv::Rect> getMajorBounds();

  std::vector<std::string> getChampNames();

  std::string getLevel();

  std::string getLevelProgress();

  std::string getGold();

  std::vector<std::string> testAllString();

  void testIRVitals(const cv::Mat& fullImg);

  void testElementIRVitals(cv::Mat fullImg);

  ~Shop () {}

private:
  const int minHessian = 400;
  const double Lowe_ratio_thresh = 0.85f;
  const int padding = 10;
  const int champMiddleSlotIndex = 2; 

  void locateElements();

  void updateElementFrames(const cv::Mat&);

  std::vector<cv::DMatch> findMatches(const cv::Mat&,
                                      const cv::Mat&);

  cv::Mat findHomo(std::vector<cv::DMatch>, const std::vector<cv::KeyPoint>&, const std::vector<cv::KeyPoint>&);

  cv::Rect findBounds(const cv::Mat&, const cv::Mat&);

};