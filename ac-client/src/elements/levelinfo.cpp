#include "ImageRegion.h"
#include "ocr.h"
#include <levelinfo.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

LevelInfo::LevelInfo(ocr& ocr) : readNum(ocr) {} // initializes the polymorphed ocr

void LevelInfo::updateImageRegion(const cv::Mat& fullImg) {
  if (fullImg.empty()) {
    std::cout << "None or empty image given.\n";
    return;
  }
  
  levelRegion.roi = fullImg(levelRegion.bounds).clone();
  progressRegion.roi = fullImg(progressRegion.bounds).clone();
}

cv::Rect LevelInfo::getLevelBounds() {
  return levelRegion.bounds; // Just returns bounds for level area
}

cv::Rect LevelInfo::getProgBounds() {
  return progressRegion.bounds; // return area were progress(x/x) is
}

std::string LevelInfo::getLevelStr() {
  cv::Mat ocrMask = readNum.preprocessWhite(levelRegion.roi);
  return readNum.readNum(ocrMask);
}

std::string LevelInfo::getProgStr() {
  cv::Mat ocrMask = readNum.preprocessBlue(progressRegion.roi);
  return readNum.readFrac(ocrMask);
}

std::pair<cv::Rect, cv::Rect> LevelInfo::getAllBounds() {
  return std::pair<cv::Rect, cv::Rect>(levelRegion.bounds, progressRegion.bounds); // returns both bounds left to right
}

void printCords(cv::Mat hsv, std::vector<int> x, std::vector<int> y) {

    for (int i = 0; i < x.size(); i++) {
        cv::Vec3b pixel = hsv.at<cv::Vec3b>(y[i], x[i]);
        std::cout << "Pixel at (" << x[i] << ", " << y[i] << "): " << pixel << "         \n";
        /*
        bool inRange = true;
        for (size_t c = 0; c < 3; c++) {
            if (pixel[c] < shopBorderMin[c] || pixel[c] > shopBorderMax[c]) {
                inRange = false;
                break;
            }
        }
        if (inRange)
            std::cout << "-Val Accepted\n";
        else
            std::cout << "\n";
            */
    }
}

void LevelInfo::locate(ImageRegion fullShop, ImageRegion leftShop) { // This relies on the fullShop and left side of the shop
  ImageRegion textRegion;
  
  cv::Rect localButtons = fullShop.localBounds(leftShop.bounds);

  if (localButtons.x < 0 || localButtons.y < 0 || localButtons.x + localButtons.width > fullShop.roi.cols || localButtons.y + localButtons.height > fullShop.roi.rows) {
      std::cerr << "Local buttons out of bounds: " << localButtons << std::endl;
      return;
  }

  cv::Rect aboveButtons = cv::Rect(leftShop.bounds.x,
                                   fullShop.bounds.y,
                                   leftShop.bounds.width,
                                   localButtons.y);


  textRegion = cropTextArea(ImageRegion(fullShop.roi( cv::Rect(localButtons.x,
                                                               0,
                                                               localButtons.width,
                                                               localButtons.y) ).clone(), // Gets leftShops bound inside of fullshops roi
                                        aboveButtons)); // crops the full shop img to only above shopbuttons

  cv::Mat hsv;
  cv::cvtColor(textRegion.roi.clone(), hsv, cv::COLOR_BGR2HSV);
  
  cv::Mat levelMask = createMask(textRegion.roi, whiteTextMin, whiteTextMax); // create a mask to only highlight level text
  cv::Mat progMask = createMask(textRegion.roi, blueTextMin, blueTextMax); // creates mask to only highlight progress text

  std::vector<cv::Rect> levelBoxes = createTextBoxes(levelMask); // Create contours around each white text found, then gets rects from them
  std::vector<cv::Rect> progBoxes = createTextBoxes(progMask); // Create contours around each blue text found, then gets rects from them
  
  cv::Rect levelTextBounds = findBoxesPerim(levelBoxes); // Gets perimeter of rects around white text
  cv::Rect progTextBounds = findBoxesPerim(progBoxes); // Gets perimeter of rects around blue text

  blankPixelDistro(levelTextBounds, progTextBounds);

  cv::Rect absoluteLevel = cv::Rect(textRegion.bounds.x + levelTextBounds.x, // adds on to textRegion.bounds which account for the shop
                                    textRegion.bounds.y + levelTextBounds.y, // same here
                                    levelTextBounds.width,
                                    levelTextBounds.height);

  cv::Rect absoluteProg = cv::Rect(textRegion.bounds.x + progTextBounds.x, // adds on to textRegion.bounds which accounts for the shop
                                   textRegion.bounds.y + progTextBounds.y, // same here
                                   progTextBounds.width,
                                   progTextBounds.height);

  
  if (levelTextBounds.x + levelTextBounds.width > textRegion.roi.cols || levelTextBounds.y + levelTextBounds.height > textRegion.roi.rows)
      std::cerr << "ROI level out of bounds: " << levelTextBounds << std::endl;

  if (progTextBounds.x + progTextBounds.width > textRegion.roi.cols || progTextBounds.y + progTextBounds.height > textRegion.roi.rows)
      std::cerr << "ROI prog out of bounds: " << progTextBounds << std::endl;


  levelRegion = ImageRegion(textRegion.roi(levelTextBounds).clone(), absoluteLevel); // Usually you should get the image roi before defining
  progressRegion = ImageRegion(textRegion.roi(progTextBounds).clone(), absoluteProg);  // abosulte bounds but its fine here because we use textRegion

}

void LevelInfo::testIRVitals(cv::Mat fullImg) {

  cv::imwrite("D:/linux_img_port/Shop_Class/Level_info/level_crop.png", levelRegion.roi);
  cv::imwrite("D:/linux_img_port/Shop_Class/Level_info/prog_crop.png", progressRegion.roi);

  cv::Mat rectsDrawn = fullImg.clone();

  cv::rectangle(rectsDrawn, levelRegion.bounds, cv::Scalar(0, 255, 0), 1);
  cv::rectangle(rectsDrawn, progressRegion.bounds, cv::Scalar(0, 255, 0), 1);

  cv::imwrite("D:/linux_img_port/Shop_Class/Level_info/shop_with_borders.png", rectsDrawn);
}

// private:


ImageRegion LevelInfo::cropTextArea(const ImageRegion& aboveButtons) { // this takes in the area above the reroll and level buttons
  cv::Mat hsv = createMask(aboveButtons.roi, infoBorderMin, infoBorderMax);

  std::vector<std::vector<cv::Point>> contours;

  cv::findContours(hsv, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  cv::Rect bestMatch;
  float lowest = 0;
  for(auto& cont : contours) { // Find a contour above 90% of the width and atleast half the height(first one is probably enough) as well as the lowest
    cv::Rect contRect = cv::boundingRect(cont);
    double area = cv::contourArea(cont);
    
    if(contRect.width > aboveButtons.bounds.width * infoWidthMin && contRect.height > aboveButtons.bounds.height * infoHeightMin && contRect.y > lowest) { // lowest here is actually highest, because pixel value increases as it goes down
      bestMatch = contRect;
      lowest = contRect.y;
    }
  }

  cv::Rect absoluteBounds = cv::Rect(aboveButtons.bounds.x + bestMatch.x,
                                      aboveButtons.bounds.y + bestMatch.y,
                                      bestMatch.width,
                                      bestMatch.height);
  
  return ImageRegion(aboveButtons.roi(bestMatch).clone(), absoluteBounds);
}

cv::Mat LevelInfo::createMask(const cv::Mat& baseImg, cv::Scalar scalarMin, cv::Scalar scalarMax) {
  cv::Mat hsv;
  cv::cvtColor(baseImg, hsv, cv::COLOR_BGR2HSV);

  cv::Mat mask;
  cv::inRange(hsv, scalarMin, scalarMax, mask);

  return mask;
}

std::vector<cv::Rect> LevelInfo::createTextBoxes(const cv::Mat& imgMask) {
  std::vector<std::vector<cv::Point>> contours;

  cv::findContours(imgMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<cv::Rect> textBoxes;
  double fullSize = double(imgMask.cols) * double(imgMask.rows);
  for(auto& cont : contours) {
    cv::Rect rect = cv::boundingRect(cont);

    if(rect.area() > fullSize * textMinArea)
      textBoxes.push_back(rect);
    
  }

  return textBoxes;
}

cv::Rect LevelInfo::findBoxesPerim(std::vector<cv::Rect> textBoxes, int padding) {

  int maxX = 0, maxY = 0, minX = 999999, minY = 999999; // sentinel value

  for(cv::Rect rect : textBoxes) {
    int endX = rect.width + rect.x; // X cordinate of end of box
    int endY = rect.height + rect.y; // Y cordinate of end of box

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

void LevelInfo::blankPixelDistro(cv::Rect& levelBounds, cv::Rect& progBounds) { // This distributes the whitespace proportioanlly to the size of each text
  double blankLength = progBounds.x - (levelBounds.x + levelBounds.width); // Space in between texts
  double totalSpaceFilled = levelBounds.width + progBounds.width; // space occupied by text
  double levelPerc = (levelBounds.width / totalSpaceFilled) ; // percent of space filled occupied by level text
  double progPerc = (progBounds.width / totalSpaceFilled); // percent of space filled occupied by progress text

  levelBounds = cv::Rect(levelBounds.x,
                         levelBounds.y,
                         levelBounds.width + (blankLength * levelPerc), // no offset needed, just adds extra space
                         levelBounds.height);

  progBounds = cv::Rect(progBounds.x - (blankLength * progPerc), // Offsets x to cover blank space
                        progBounds.y,
                        progBounds.width + (blankLength * progPerc), // recovers previous width by adding offset
                        progBounds.height);

}


