#include "ImageRegion.h"
#include <champslots.h>
#include <algorithm>
// TEST: temporary headers for debugging:
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>

void ChampSlots::updateImageRegion(const cv::Mat& fullImg) {
  if (fullImg.empty()) {
    std::cout << "None or empty image given.\n";
    return;
  }
  
  for (int i = 0; i < champSlots.size(); ++i) {
    champSlots.at(i).roi = fullImg(champSlots.at(i).bounds);
  }
}

cv::Rect ChampSlots::getChampSlot(int index) {
  return champSlots.at(index).bounds;
}

std::vector<cv::Rect> ChampSlots::getAllChampSlot() {
  std::vector<cv::Rect> allBounds;
  
  for(ImageRegion& slot : champSlots) {
    allBounds.push_back(slot.bounds);
  }

  return allBounds;
}

void ChampSlots::locate(ImageRegion shopReg) {
  cv::Mat champSlotsMask;
  std::vector<cv::Rect> slotsBounds;

  champSlotsMask = createMask(shopReg.roi);

  slotsBounds = createContours(champSlotsMask);


  std::sort(slotsBounds.begin(), slotsBounds.end(), [](const cv::Rect& a, const cv::Rect& b) {
              return a.x < b.x;
            });
  
  for(cv::Rect& rect : slotsBounds) {
    cv::Rect absoluteBound = cv::Rect(shopReg.bounds.x + rect.x,
                                    shopReg.bounds.y + rect.y,
                                    rect.width,
                                    rect.height);
    
    champSlots.push_back({shopReg.roi(rect).clone(), absoluteBound});
  }
}


void ChampSlots::testIRVals(cv::Mat fullImg) {
  cv::Mat drawnOnShop = fullImg.clone();
  
  for(auto& region : champSlots) {
    cv::rectangle(drawnOnShop, region.bounds, cv::Scalar(0, 255, 0), 1);
  }
  cv::imwrite("D:/linux_img_port/Shop_Class/Champ_slots/borders_drawn.png", drawnOnShop);

  for(int i = 0; i < champSlots.size(); i++) {
    cv::imwrite("D:/linux_img_port/Shop_Class/Champ_slots/champ_slot" + std::to_string(i) + ".png", champSlots.at(i).roi);
  }
}

// private:

cv::Mat ChampSlots::createMask(cv::Mat img) {
  cv::Mat hsv;
  cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

  cv::imwrite("D:/linux_img_port/Shop_Class/Champ_slots/hsv_mask.png", hsv);

  std::vector<int> x = {  };
  std::vector<int> y = {  };
  for (int i = 0; i < x.size(); i++) {
      cv::Vec3b pixel = hsv.at<cv::Vec3b>(y[i], x[i]);

      std::cout << "Pixel at (" << x[i] << ", " << y[i] << "): " << pixel << "         ";

      bool inRange = true;
      for (size_t c = 0; c < 3; c++) {
          if (pixel[c] < champBorderMin[c] || pixel[c] > champBorderMax[c]) {
              inRange = false;
              break;
          }
      }
      if (inRange)
          std::cout << "-Val Accepted\n";
      else
          std::cout << "\n";
  }

  cv::Mat bordersMask;
  cv::inRange(hsv, champBorderMin, champBorderMax, bordersMask); // Increase H and decrease S if too much is found

  cv::imwrite("D:/linux_img_port/Shop_Class/Champ_slots/borders_mask.png", bordersMask);

  // --- Debug: create a color version of the mask and draw colored spots at x,y ---
  cv::Mat colorMask;
  cv::cvtColor(bordersMask, colorMask, cv::COLOR_GRAY2BGR);
  for (int i = 0; i < x.size(); i++) {
      // Draw a red circle at each (x, y) location
      cv::circle(colorMask, cv::Point(x[i], y[i]), 3, cv::Scalar(0, 0, 255), -1);
  }
  cv::imwrite("D:/linux_img_port/Shop_Class/Champ_slots/champ_mask_debug_spots.png", colorMask);
  // ---

  cv::Mat dilKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)); 
  cv::dilate(bordersMask, bordersMask, dilKernel, cv::Point(-1, -1), 2);

  cv::Mat inverseMask;
  cv::bitwise_not(bordersMask, inverseMask);

  for (int i = 0; i < x.size(); i++) {
      uchar pixel = bordersMask.at<uchar>(y[i], x[i]);
      std::cout << "Pixel at (" << x[i] << ", " << y[i] << "): " << static_cast<int>(pixel) << std::endl;
  }

  cv::Mat cleanedMask;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::Mat vertKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 7));
  cv::Mat horzKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 1));

  cv::morphologyEx(inverseMask, cleanedMask, cv::MORPH_OPEN, kernel, {}, 2);
  cv::morphologyEx(cleanedMask, cleanedMask, cv::MORPH_RECT, vertKernel);
  cv::morphologyEx(cleanedMask, cleanedMask, cv::MORPH_RECT, horzKernel);

  cv::imwrite("D:/linux_img_port/Shop_Class/Champ_slots/cleaned_mask.png", cleanedMask);

  return cleanedMask;
}

cv::Mat ChampSlots::approxMask(std::vector<cv::Point> contour, cv::Mat fullShop) {
    std::vector<cv::Point> approx;
    cv::Rect biggestRect = cv::boundingRect(contour);
    cv::approxPolyDP(contour, approx, epsilon, false); // Removes points that don significantly contribute to the shape

    std::vector<std::vector<cv::Point>> vecApprox = { approx };
    cv::Mat approxMask = cv::Mat::zeros(fullShop.size(), CV_8UC1); // generate black mask for drawing

    cv::polylines(approxMask, vecApprox, true, cv::Scalar(255), 2); // Draw white lines around border

    return approxMask;
}

std::vector<cv::Rect> ChampSlots::createContours(cv::Mat mask) {
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Rect> champBoxes;

  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  //cv::Mat contoursDrawn = cv::Mat::zeros(mask.size(), CV_8UC3);
  //cv::drawContours(contoursDrawn, contours, -1, cv::Scalar(0, 255, 0), 1);

  double fullArea = double(mask.rows) * double (mask.cols);
  std::vector<cv::Point> bestMatch;
  for(auto& cont : contours) {
    cv::Rect rect = cv::boundingRect(cont);
    double area = cv::contourArea(cont);
    double aspectRatio = double(rect.height) /  double(rect.width);   

    if(area > fullArea * maxArea || area < fullArea * minArea)
      continue;
    
    if (aspectRatio > boundRatio - ratioOffset && aspectRatio < boundRatio + ratioOffset) {
        champBoxes.push_back(rect);
    }
  }
  
  return champBoxes;
}
