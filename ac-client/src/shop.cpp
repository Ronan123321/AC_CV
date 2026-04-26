#include "ImageRegion.h"
#include "ocr.h"
#include "shopsplit.h"
#include <allheaders.h>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <shop.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>


Shop::Shop() : findGold(tesseractOCR), findLevelinfo(tesseractOCR), findChampNames(tesseractOCR) {}

void Shop::locate(ImageRegion fullScreen, cv::Mat shopTemplate) {
  cv::Mat grayShopTemplate, grayFullScreen;

  if(fullScreen.roi.empty() || shopTemplate.empty()) {
    std::cerr << "Empty image provided for locating shop.\n";
    return;
  }

  if(fullScreen.roi.channels() == 3) {
    cv::cvtColor(fullScreen.roi, grayFullScreen, cv::COLOR_BGR2GRAY);
  } else {
    grayFullScreen = fullScreen.roi.clone();
  }

  if(shopTemplate.channels() == 3) {
    cv::cvtColor(shopTemplate, grayShopTemplate, cv::COLOR_BGR2GRAY);
  } else {
    grayShopTemplate = shopTemplate.clone();
  }
  
  cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );
  std::vector<cv::KeyPoint> keyPointsObject, keyPointsScene;
  cv::Mat descriptorsObject, descriptorsScene;

  detector->detectAndCompute(grayShopTemplate, cv::noArray(), keyPointsObject, descriptorsObject);
  detector->detectAndCompute(grayFullScreen, cv::noArray(), keyPointsScene, descriptorsScene);

  std::vector<cv::DMatch> bestMatches = findMatches(descriptorsObject, descriptorsScene);

  cv::Mat homograph = findHomo(bestMatches, keyPointsObject, keyPointsScene);

  cv::Rect shopBounds = findBounds(homograph, grayShopTemplate);

  int maxLeftPad = std::min(padding, shopBounds.x);
  int maxRightPad = std::min(padding, fullScreen.roi.cols - (shopBounds.x + shopBounds.width));
  int maxTopPad = std::min(padding, shopBounds.y);
  int maxBottomPad = std::min(padding, fullScreen.roi.rows - (shopBounds.y + shopBounds.height));

  shopBounds.x -= maxLeftPad;
  shopBounds.width += maxLeftPad + maxRightPad;
  shopBounds.y -= maxTopPad;
  shopBounds.height += maxTopPad + maxBottomPad;


  std::cout << "Shop bounds: " << shopBounds << std::endl;

  region = ImageRegion(fullScreen.roi(shopBounds).clone(), cv::Rect(fullScreen.bounds.x + shopBounds.x,
                                                                    fullScreen.bounds.y + shopBounds.y,
                                                                    shopBounds.width,
                                                                    shopBounds.height));
}

cv::Mat tempFullImage; // Used to store the full image for later use, only for TESTING purposes
void Shop::locateShopELements(ImageRegion fullScreeen, cv::Mat shopTemplates) {
  tempFullImage = fullScreeen.roi.clone();
  locate(fullScreeen, shopTemplates);
  locateElements();
}

void Shop::relocateElements() {
  locateElements();
}

void Shop::relocateElements(ImageRegion shop) {
  region = shop;
  locateElements();

}

void Shop::updateFrame(const cv::Mat& fullImg) {
    // Update the region with the new full image
    region.roi = fullImg(region.bounds);
    // Relocate elements based on the updated region
	updateElementFrames(fullImg);
}

std::vector<cv::Rect> Shop::getAllBounds() {
    std::vector<cv::Rect> allBounds;
    std::pair<cv::Rect, cv::Rect> loadingPair = shopSplit.getBoth();
    allBounds.push_back(loadingPair.first);
    allBounds.push_back(loadingPair.second);
    loadingPair = findButtons.getBounds();
    allBounds.push_back(loadingPair.first);
    allBounds.push_back(loadingPair.second);
    loadingPair = findLevelinfo.getAllBounds();
    allBounds.push_back(loadingPair.first);
    allBounds.push_back(loadingPair.second);
    std::vector<cv::Rect> loadingVec = findChampslots.getAllChampSlot();
    allBounds.insert(allBounds.end(), loadingVec.begin(), loadingVec.end());
    loadingVec = findChampNames.getAllBounds();
    allBounds.insert(allBounds.end(), loadingVec.begin(), loadingVec.end());
    allBounds.push_back(findGold.getBounds());

    return allBounds;
}

std::vector<cv::Rect> Shop::getMajorBounds() {
    std::vector<cv::Rect> majorBounds;

    std::pair<cv::Rect, cv::Rect> loadingPair;
    std::vector<cv::Rect> loadingVec;
    loadingPair = findButtons.getBounds();
    majorBounds.push_back(loadingPair.first);
    majorBounds.push_back(loadingPair.second);
    loadingPair = findLevelinfo.getAllBounds();
    majorBounds.push_back(loadingPair.first);
    majorBounds.push_back(loadingPair.second);
    loadingVec = findChampNames.getAllBounds();
    majorBounds.insert(majorBounds.end(), loadingVec.begin(), loadingVec.end());
    majorBounds.push_back(findGold.getBounds());

    return majorBounds;
}

std::vector<std::string> Shop::getChampNames() {
	return findChampNames.getAllNames();
}

std::string Shop::getLevel() {
    return findLevelinfo.getLevelStr();
}

std::string Shop::getLevelProgress() {
    return findLevelinfo.getProgStr();
}

std::string Shop::getGold() {
    return findGold.getText();
}

std::vector<std::string> Shop::testAllString() {
    std::vector<std::string> allStrings;
    // Add champ names at the start
    std::vector<std::string> champNames = getChampNames();
    allStrings.insert(allStrings.end(), champNames.begin(), champNames.end());
    allStrings.push_back(getLevel());
    allStrings.push_back(getLevelProgress());
    allStrings.push_back(getGold());
    return allStrings;
}

void Shop::testIRVitals(const cv::Mat& fullImg) {

  cv::imwrite("D:/linux_img_port/Shop_Class/Shop/shop_roi.png", region.roi);

  cv::Mat regionDrawn = fullImg.clone();
  cv::rectangle(regionDrawn, region.bounds, cv::Scalar(0, 255, 0), 1);
  cv::imwrite("D:/linux_img_port/Shop_Class/Shop/borders_drawn.png", regionDrawn);
  
}

void Shop::testElementIRVitals(cv::Mat fullImg) {
  testIRVitals(fullImg);

  shopSplit.testIRVals(fullImg);
  findButtons.drawOnImg(fullImg, true);
  findLevelinfo.testIRVitals(fullImg);
  findChampslots.testIRVals(fullImg);
  findChampNames.testIRVals(fullImg);
  findGold.testIRVitals(fullImg);
}

// private:

void Shop::locateElements() {
  testIRVitals(tempFullImage);

  shopSplit.locate(region);
  std::cout << "Shopsplit: done\n";
  shopSplit.testIRVals(tempFullImage.clone());

  findButtons.locate(shopSplit.left);
  std::cout << "Buttons: done\n";
  findButtons.drawOnImg(tempFullImage.clone(), true);

  findLevelinfo.locate(region, shopSplit.left);
  std::cout << "Levelinfo: done\n";
  findLevelinfo.testIRVitals(tempFullImage.clone());

  findChampslots.locate(shopSplit.right);
  std::cout << "Champslots: done\n";
  findChampslots.testIRVals(tempFullImage.clone());

  findChampNames.locate(findChampslots.champSlots);
  std::cout << "Champnames: done\n";
  findChampNames.testIRVals(tempFullImage.clone());

  findGold.locate(region, findChampslots.champSlots[champMiddleSlotIndex]);
  std::cout << "Gold: done\n";
  findGold.testIRVitals(tempFullImage.clone());
  
}

void Shop::updateElementFrames(const cv::Mat& fullImg) {
  shopSplit.updateImageRegion(fullImg);
  findButtons.updateImageRegion(fullImg);
  findLevelinfo.updateImageRegion(fullImg);
  findChampslots.updateImageRegion(fullImg);
  findChampNames.updateImageRegion(fullImg);
  findGold.updateImageRegion(fullImg);

}

std::vector<cv::DMatch> Shop::findMatches(const cv::Mat& descriptorObject,
                                          const cv::Mat& descriptorScene) {
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector<std::vector<cv::DMatch>> knnMatches;
  
  matcher->knnMatch(descriptorObject, descriptorScene, knnMatches, 2);

  std::vector<cv::DMatch> bestMatches;
  for(size_t matchIt = 0; matchIt < knnMatches.size(); matchIt++) {
    if(knnMatches[matchIt][0].distance < Lowe_ratio_thresh * knnMatches[matchIt][1].distance)
      bestMatches.push_back(knnMatches[matchIt][0]);
    
  }

  return bestMatches;
}

cv::Mat Shop::findHomo(std::vector<cv::DMatch> knnMatches, const std::vector<cv::KeyPoint>& keypointsObject,
                                                           const std::vector<cv::KeyPoint>& keypointsScene) {
  std::vector<cv::Point2f> objectMatches, sceneMatches;

  for(size_t i = 0; i < knnMatches.size(); i++) {
    objectMatches.push_back(keypointsObject[knnMatches[i].queryIdx].pt);
    sceneMatches.push_back(keypointsScene[knnMatches[i].trainIdx].pt);
  }

  cv::Mat homography = cv::findHomography(objectMatches, sceneMatches, cv::RANSAC);

  return homography;
}


cv::Rect Shop::findBounds(const cv::Mat& homo, const cv::Mat& shop) {
  std::vector<cv::Point2f> objectCorners(4);
  std::vector<cv::Point2f> sceneCorners(4);

  objectCorners[0] = cv::Point2f(0, 0);
  objectCorners[1] = cv::Point2f((float)shop.cols, 0);
  objectCorners[2] = cv::Point2f((float)shop.cols, (float)shop.rows);
  objectCorners[3] = cv::Point2f(0, (float)shop.rows);

  cv::perspectiveTransform(objectCorners, sceneCorners, homo);

  return cv::boundingRect(sceneCorners);
}
