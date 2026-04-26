#include <buttons.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
// TEMP: imwrite capabilities for testing
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <ImageRegion.h>
#include <iostream>

Buttons::Buttons() {

  
}

void Buttons::updateImageRegion(const cv::Mat& fullImg) {
  if (fullImg.empty()) {
    std::cout << "None or empty image given.\n";
    return;
  }
  
  rerollRegion.roi = fullImg(rerollRegion.bounds).clone();
  levelRegion.roi = fullImg(levelRegion.bounds).clone();
}

std::pair<cv::Rect, cv::Rect> Buttons::getBounds() {
  return std::pair<cv::Rect, cv::Rect>(levelRegion.bounds, rerollRegion.bounds);
}

std::pair<cv::Scalar, cv::Scalar> Buttons::getScalarBounds() {
  return std::pair<cv::Scalar, cv::Scalar>(borderMin, borderMax);
}

void Buttons::locate(ImageRegion leftShopReg) {
  std::vector<cv::Rect> buttonsBounds;
  cv::Mat inverseMask;

  inverseMask = createInverseMask(leftShopReg.roi.clone());
  
  buttonsBounds = createBoundingRects(inverseMask);

  if(buttonsBounds.size() != 2) {
    std::cout << "No buttons found, using default bounding rects\n";
    buttonsBounds = defaultBoundingRects(leftShopReg.roi.clone());
  }

  cv::Rect rerollBounds;
  cv::Rect levelBounds = cv::Rect(leftShopReg.roi.rows, leftShopReg.roi.cols, 0, 0); // Sentinel value

  for(auto& rect : buttonsBounds) { // Retrieves highest and lowest with the biggest bounds
    if(rect.y > rerollBounds.y) {
      rerollBounds = rect;
    }
    if(rect.y < levelBounds.y) {
      levelBounds = rect;
    }
  }

  rerollRegion.roi = leftShopReg.roi(rerollBounds).clone();
  levelRegion.roi = leftShopReg.roi(levelBounds).clone();

  rerollRegion.bounds = cv::Rect(leftShopReg.bounds.x + rerollBounds.x,
                                 leftShopReg.bounds.y + rerollBounds.y,
                                 rerollBounds.width,
                                 rerollBounds.height);

  levelRegion.bounds = cv::Rect(leftShopReg.bounds.x + levelBounds.x,
                                leftShopReg.bounds.y + levelBounds.y,
                                levelBounds.width,
                                levelBounds.height);
  
}

cv::Mat Buttons::drawOnImg(cv::Mat fullImg, bool save) {
  cv::Mat buttonsDrawn = fullImg.clone();

  cv::rectangle(buttonsDrawn, rerollRegion.bounds, cv::Scalar(0, 255, 0), 1);
  cv::rectangle(buttonsDrawn, levelRegion.bounds, cv::Scalar(0, 255, 0), 1);

  if(save) {
    cv::imwrite("D:/linux_img_port/Shop_Class/Shop_buttons/buttons_drawn.png", buttonsDrawn);
    cv::imwrite("D:/linux_img_port/Shop_Class/Shop_buttons/reroll_drawn.png", rerollRegion.roi);
    cv::imwrite("D:/linux_img_port/Shop_Class/Shop_buttons/level_drawn.png", levelRegion.roi);
  }
  return buttonsDrawn;
}

// Private methods:


cv::Mat Buttons::createInverseMask(cv::Mat leftShopImg) { // Could change this so it only uses one Mat
  cv::Mat hsv;
  cv::cvtColor(leftShopImg, hsv, cv::COLOR_BGR2HSV);

  cv::Mat borderMask;
  cv::inRange(hsv,borderMin, borderMax, borderMask);

  cv::Mat invertedMask; // Have to inverse because contours wants buttons mask not border
  cv::bitwise_not(borderMask, invertedMask);

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::erode(invertedMask, invertedMask, kernel, cv::Point(-1, -1), 1); // maybe decrease to 1 iteration

  return invertedMask; // This returns the buttons highlighted in white
}

std::vector<cv::Rect> Buttons::createBoundingRects(cv::Mat mask) {
  std::vector<std::vector<cv::Point>> contours;

  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Rect> boundRects;
  double fullArea = double(mask.rows) * double(mask.cols);
  for(std::vector<cv::Point> cont : contours) {
    int area = cv::contourArea(cont);
    cv::Rect rect = cv::boundingRect(cont);

    if(area > fullArea * minSize && area < fullArea * maxSize) 
      boundRects.push_back(rect);
  }

  return boundRects;
}


std::vector<cv::Rect> Buttons::defaultBoundingRects(cv::Mat leftShopImg) {
  std::vector<cv::Rect> defaultRects;
  defaultRects.push_back(cv::Rect(0 + defaultPadding, 0 + defaultPadding, leftShopImg.cols - (defaultPadding * 2), (leftShopImg.rows * 0.5) - defaultPadding * 2));
  defaultRects.push_back(cv::Rect(0 + defaultPadding, leftShopImg.rows * 0.5 + defaultPadding, leftShopImg.cols - (defaultPadding * 2), (leftShopImg.rows * 0.5) - defaultPadding * 2)); 
  return defaultRects;
}