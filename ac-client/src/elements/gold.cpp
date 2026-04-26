#include "ImageRegion.h"
#include "ocr.h"
#include <gold.h>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

Gold::Gold(ocr& ocr) : readNum(ocr) {}

void Gold::updateImageRegion(const cv::Mat& fullImg) {
  if (fullImg.empty()) {
    std::cout << "None or empty image given.\n";
    return;
  }
  
  goldRegion.roi = fullImg(goldRegion.bounds).clone();
}

cv::Rect Gold::getBounds() {

  return goldRegion.bounds;
}

std::string Gold::getText() {
  cv::Mat ocrMask = readNum.preprocessWhite(goldRegion.roi);
  return readNum.readNum(ocrMask);
}

void Gold::locate(ImageRegion fullShop, ImageRegion middleSlot) {
  ImageRegion numRegion;

  cv::Rect localMidSlot = fullShop.localBounds(middleSlot.bounds); 
  cv::Rect aboveSlot = cv::Rect(localMidSlot.x,
                                0,
                                localMidSlot.width,
                                localMidSlot.y);
  
  cv::Rect absAboveSlot = cv::Rect(fullShop.bounds.x + localMidSlot.x,
                                   fullShop.bounds.y + 0,
                                   localMidSlot.width,
                                   localMidSlot.y); 
  
  numRegion = ImageRegion(fullShop.roi(aboveSlot), absAboveSlot);
 
  
  cv::Mat numMask = whiteNumMask(numRegion.roi);
  std::vector<cv::Rect> maskBounds = createNumBox(numMask);
  cv::Rect numBounds = findBoxesPerim(maskBounds);

  cv::Rect expanded = numBounds;
  expanded.x -= increaseBorder;
  expanded.y -= increaseBorder;
  expanded.width += increaseBorder * 2;
  expanded.height += increaseBorder * 2;

  // Clamp to the bounds of numRegion
  expanded.x = std::max(0, expanded.x);
  expanded.y = std::max(0, expanded.y);
  if (expanded.x + expanded.width > numRegion.bounds.width)
      expanded.width = numRegion.bounds.width - expanded.x;
  if (expanded.y + expanded.height > numRegion.bounds.height)
      expanded.height = numRegion.bounds.height - expanded.y;

  numBounds = expanded;

  /*
    numBounds = cv::Rect(numBounds.x - increaseBorder, // adds on increaseBorder to numBounds
        numBounds.y - increaseBorder, // which is based on numBounds
        numBounds.width + increaseBorder * 2,
        numBounds.height + increaseBorder * 2);
  */

  goldRegion = ImageRegion(numRegion.roi(numBounds).clone(), cv::Rect(numRegion.bounds.x + numBounds.x, // adds on numBounds to numRegion
                                                                      numRegion.bounds.y + numBounds.y, // which numBounds is based on
                                                                      numBounds.width,
                                                                      numBounds.height));
}

void Gold::testIRVitals(cv::Mat img) {

  cv::imwrite("D:/linux_img_port/Shop_Class/Gold/gold_area.png", goldRegion.roi);

  cv::Mat drawnOn = img.clone();

  cv::rectangle(drawnOn, goldRegion.bounds, cv::Scalar(0, 255, 0), 1);

  cv::imwrite("D:/linux_img_port/Shop_Class/Gold/gold_area_drawn.png", drawnOn);
}

// private:


cv::Mat Gold::whiteNumMask(const cv::Mat& aboveChampSlot) {
  cv::Mat hsv;
  cv::cvtColor(aboveChampSlot, hsv, cv::COLOR_BGR2HSV);

  cv::Mat whiteNumMask;
  cv::inRange(hsv, borderMin, borderMax, whiteNumMask);

  std::vector<int> x = { };
  std::vector<int> y = { };
  for (int i = 0; i < x.size(); i++) {
      cv::Vec3b pixel = hsv.at<cv::Vec3b>(y[i], x[i]);
      std::cout << "Pixel at (" << x[i] << ", " << y[i] << "): " << pixel << std::endl;
  }

  return whiteNumMask;
}

std::vector<cv::Rect> Gold::createNumBox(cv::Mat goldMask) { // called gold mask because its for the gold num. Mask is white
  std::vector<std::vector<cv::Point>> contours;
  
  cv::findContours(goldMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Rect> numBorders;
  const double fullSize = double(goldMask.cols) * double(goldMask.rows);
  for(auto& cont : contours) {
    const cv::Rect contRect = cv::boundingRect(cont);
    const int contArea = cv::contourArea(cont);

    if(contArea < fullSize * numAreaMax && contArea > fullSize * numAreaMin)
      numBorders.push_back(contRect);
  }

  return numBorders;
}

cv::Rect Gold::findBoxesPerim(std::vector<cv::Rect> allBoxes, int padding) {

  int maxX = 0, maxY = 0, minX = 99999, minY = 99999; // 99999 sentinel value 

  for(cv::Rect rect : allBoxes) {
    int endX = rect.width + rect.x;
    int endY = rect.height + rect.y;

    if(rect.x < minX)
      minX = rect.x;

    if(rect.y < minY)
      minY = rect.y;

    if(endX > maxX)
      maxX = endX;

    if(endY > maxY)
      maxY = endY;
  }

  int xCord = minX - padding;
  int yCord = minY - padding;
  int xSize = maxX + padding * 2 - minX;
  int ySize = maxY + padding * 2 - minY;

  return cv::Rect(xCord, yCord, xSize, ySize);
}
