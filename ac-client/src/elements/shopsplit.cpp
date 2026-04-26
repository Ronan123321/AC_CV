#include "ImageRegion.h"
#include "ocr.h"
#include <opencv2/core/hal/interface.h>
#include <shopsplit.h>
#include <opencv2/imgcodecs.hpp>
#include <iostream>

void ShopSplit::updateImageRegion(const cv::Mat& fullImg) {
  if (fullImg.empty()) {
    std::cout << "None or empty image given.\n";
    return;
  }
  left.roi = fullImg(left.bounds).clone();
  right.roi = fullImg(right.bounds).clone();
}

cv::Rect ShopSplit::getLeft() {

  return left.bounds;
}

cv::Rect ShopSplit::getRight() {

  return right.bounds;
}

std::pair<cv::Rect, cv::Rect> ShopSplit::getBoth() {

  return std::pair<cv::Rect, cv::Rect>(left.bounds, right.bounds);
}

void ShopSplit::locate(ImageRegion fullShop) {
    std::vector<std::vector<cv::Point>> bestMatch;

    bestMatch = findBorderMask(fullShop.roi.clone());

    cv::Mat flatMask = findFlatMask(fullShop.roi.clone(), bestMatch);

    std::vector<int> sliceCords = findSliceCords(flatMask);
    if (sliceCords.size() > 3) {
        sliceCords = sliceFallback(sliceCords);
    }

    std::vector<cv::Rect> shopBounds;
    shopBounds = shopLeftRight(sliceCords, bestMatch);
    if (shopBounds.size() > 2) {
        shopBounds = twoBiggest(shopBounds);
    }
    else if(shopBounds.size() < 2)
        std::cerr << "Shop amount mismatch:  " << shopBounds.size() << std::endl;

    cv::Rect absoluteLeftBounds = cv::Rect(fullShop.bounds.x  + shopBounds[0].x,
                                            fullShop.bounds.y + shopBounds[0].y,
                                            shopBounds[0].width,
                                            shopBounds[0].height);

    cv::Rect absoluteRightBounds = cv::Rect(fullShop.bounds.x  + shopBounds[1].x,
                                            fullShop.bounds.y + shopBounds[1].y,
                                            shopBounds[1].width,
                                            shopBounds[1].height);


    left = ImageRegion(fullShop.roi(shopBounds[0]).clone(), absoluteLeftBounds);
    right = ImageRegion(fullShop.roi(shopBounds[1]).clone(), absoluteRightBounds);
}

void ShopSplit::testIRVals(cv::Mat fullImg) {
  cv::Mat rectsDrawn = fullImg.clone();

  cv::imwrite("D:/linux_img_port/Shop_Class/Shop_split/left_split.png", left.roi);
  cv::imwrite("D:/linux_img_port/Shop_Class/Shop_split/right_split.png", right.roi);

  cv::rectangle(rectsDrawn, left.bounds, cv::Scalar(0, 255, 0));
  cv::rectangle(rectsDrawn, right.bounds, cv::Scalar(0, 255, 0));

  cv::imwrite("D:/linux_img_port/Shop_Class/Shop_split/bounds_drawn.png", rectsDrawn);
}

cv::Mat ShopSplit::joinCrop(cv::Mat mask) {
    // Join broken horizontal segments across the full mask
    cv::Size kernelSize(15, 1); // Size for horizKernel;
    cv::Mat horizKernel =
        cv::getStructuringElement(cv::MORPH_RECT, kernelSize);
    cv::Mat maskH;
    cv::morphologyEx(mask, maskH, cv::MORPH_CLOSE, horizKernel);
  

    // Find the single long horizontal run’s Y coordinate
    int flatY = maskH.rows;
    for (int y = 0; y < maskH.rows; ++y) {
        if (cv::countNonZero(maskH.row(y)) > maskH.cols / 2) {
            // “mostly full” row
            flatY = y;
            break;
        }
    }

    // crop everything above that line
    cv::Mat below = maskH.rowRange(flatY, maskH.rows).clone();
    // (if you need to keep absolute coords, remember to offset by flatY later)

    // join broken vertical segments inside the cropped region
    int vertGap = 10;  // max gap in px for your vertical lines
    cv::Mat vertKernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, vertGap));
    cv::Mat maskV;
    cv::morphologyEx(below, maskV, cv::MORPH_CLOSE, vertKernel);

    // shift maskV back into full-frame coords for contouring
    cv::Mat finalMask = cv::Mat::zeros(mask.size(), mask.type());
    maskV.copyTo(finalMask.rowRange(flatY, mask.rows));

    return finalMask;
}

cv::Mat strengthenBorder(cv::Mat mask) {
    // For vertical lines
    cv::Mat vertical = mask.clone();
    int vertical_size = 20;  
    cv::Mat vertical_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, vertical_size));
    cv::erode(vertical, vertical, vertical_kernel);
    cv::dilate(vertical, vertical, vertical_kernel);

    // For horizontal lines
    cv::Mat horizontal = mask.clone();
    int horizontal_size = 20;  
    cv::Mat horizontal_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(horizontal_size, 1));
    cv::erode(horizontal, horizontal, horizontal_kernel);
    cv::dilate(horizontal, horizontal, horizontal_kernel);

    // Combine strong vertical and horizontal lines
    cv::Mat lines;
    cv::bitwise_or(vertical, horizontal, lines);
    
    int v_gap_fill = 8;  // Connects gaps up to 8 vertically
    cv::Mat vert_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, v_gap_fill));
    cv::morphologyEx(lines, lines, cv::MORPH_CLOSE, vert_kernel);

    int h_gap_fill = 8;  // Connects gaps up to 8 vertically
    cv::Mat horz_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(h_gap_fill, 1));
    cv::morphologyEx(lines, lines, cv::MORPH_CLOSE, horz_kernel);
    
    //cv::dilate(lines, lines, vert_kernel);
    
    return lines;
}

std::vector<std::vector<cv::Point>> ShopSplit::findBorderMask(const cv::Mat& fullShop) {
  cv::Mat hsv;
  cv::cvtColor(fullShop, hsv, cv::COLOR_BGR2HSV);

  std::vector<int> x = { };
  std::vector<int> y = { };
  for (int i = 0; i < x.size(); i++) {
      cv::Vec3b pixel = hsv.at<cv::Vec3b>(y[i], x[i]);
      std::cout << "Pixel at (" << x[i] << ", " << y[i] << "): " << pixel << "         ";
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
  }


  cv::Mat borderMask;
  cv::inRange(hsv, shopBorderMin, shopBorderMax, borderMask);

  cv::Mat strengthenedMask = joinCrop(borderMask); // joinCrop may not be needed. Is only here in the case where the contour is not uniform. May cause issues with border height;

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(strengthenedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  //contours = destroyBlobs(contours);

  double bestArea = 0;
  std::vector<std::vector<cv::Point>> bestCont(1);

  for(auto& cont : contours) { // Extracts biggest contour that passes requirements
    double area = cv::contourArea(cont);
    cv::Rect rect = cv::boundingRect(cont);

    
    if(area > bestArea && rect.width > fullShop.cols * shopBoundMin) {
      bestArea = area;
      bestCont.at(0) = cont;
    }
  }

  return bestCont;
}

cv::Mat ShopSplit::vertInclude(std::vector<cv::Point> contour, cv::Mat fullShop) {
	const float angleTolDeg = 10.0f; // Tolerance for vertical angle in degrees
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, epsilon, /*closed=*/true);

    // Filter the segments by angle
    std::vector<cv::Vec4i> verticalSegs;
    size_t N = approx.size();
    for (size_t i = 0; i < N; ++i) {
        const cv::Point& p1 = approx[i];
        const cv::Point& p2 = approx[(i + 1) % N];
        float dy = float(p2.y - p1.y);
        float dx = float(p2.x - p1.x);
        float angle = std::atan2f(dy, dx) * 180.0f / CV_PI;
        if (angle < 0) angle += 180.0f;
        // Keep only those within [90-tol, 90+tol]
        if (std::abs(angle - 90.0f) < angleTolDeg) {
            verticalSegs.emplace_back(p1.x, p1.y, p2.x, p2.y);
        }
    }

	cv::Mat output = fullShop.clone();
    // Draw those vertical segments into a fresh maskz
    cv::Mat mask = cv::Mat::zeros(output.size(), CV_8UC1);
    for (auto& seg : verticalSegs) {
        cv::line(mask,
            cv::Point(seg[0], seg[1]),
            cv::Point(seg[2], seg[3]),
            cv::Scalar(255),
            /*thickness=*/2);
    }

    return mask;
}

cv::Mat ShopSplit::findFlatMask(const cv::Mat& fullShop, std::vector<std::vector<cv::Point>> contour) {
  std::vector<cv::Point> approx;
  cv::Rect biggestRect = cv::boundingRect(contour.at(0));

  cv::Mat contourMask = vertInclude(contour.at(0), fullShop.clone());

  cv::Mat projection;
  cv::reduce(contourMask, projection, 0, cv::REDUCE_SUM, CV_32S); // Flattens to 1d, sums pixel values by col

  return projection;
}

std::vector<int> ShopSplit::findSliceCords(const cv::Mat& projection) {
  bool inBorder = false;
  int borderStart = -1;
  int borderThreshold = 255 * 100; // If row had more than 2 white(255) pixels
  int borderGraceSpace = 70;

  std::vector<int> sliceImg;
  for(size_t proIt = 0; proIt < projection.cols; proIt++) {
    int val = projection.at<int>(0, proIt);

    if(val > borderThreshold) {
      if(!inBorder) {   
        inBorder = true;
        borderStart = proIt;
      }
    }
    else if(inBorder && val < borderThreshold) {
      inBorder = false;
      int borderEnd = proIt - 1; // last border pixel was last index

      int splitX = (borderStart + borderEnd) / 2; // Splits border in the middle for slicing

      sliceImg.push_back(splitX);
      proIt += borderGraceSpace;
    }
  } 

  if (inBorder) {
      int splitX = (borderStart + projection.cols) / 2;

      sliceImg.push_back(splitX);
  }

  return sliceImg; 
}

std::vector<cv::Rect> ShopSplit::shopLeftRight(std::vector<int> sliceCords, std::vector<std::vector<cv::Point>> bestMatch) {
    if (sliceCords.empty()) {
        std::cout << "Slice cords empty\n";
        return std::vector<cv::Rect>();
    }

  const cv::Rect biggestRect = cv::boundingRect(bestMatch.at(0)); // gets offsets needed for forming rects
  const int xOffSet = biggestRect.x;
  const int yOffSet = biggestRect.y;

  std::vector<cv::Rect> shopSlices;
  int prevSlice = sliceCords.at(0) > borderSliceDistance ? 0 : sliceCords.at(0); // Sets prevslice to 0 or to the first slice cord. Corrects if no first slice is found

  for(size_t sliceIt = 1; sliceIt < sliceCords.size(); sliceIt++) {
    int width = sliceCords.at(sliceIt) - prevSlice;

    shopSlices.emplace_back(prevSlice, 0 + yOffSet, width, biggestRect.height); // forms rect based on previous slice current slice and offsets
    prevSlice = sliceCords.at(sliceIt);
  }
  
  if(prevSlice < biggestRect.width - borderSliceDistance) { // maybe biggestRect.width - 1
    shopSlices.emplace_back(prevSlice, 0 + yOffSet, biggestRect.width - prevSlice, biggestRect.height); // edgecase: no last 
  }

  return shopSlices;
}

std::vector<cv::Rect> ShopSplit::twoBiggest(std::vector<cv::Rect> rectsBounds) {
    int maxArea1 = 0;
    int maxArea2 = 0;

    for (int i = 0; i < rectsBounds.size(); i++) {
        int area = rectsBounds[i].height * rectsBounds[i].width;

        if (area > maxArea1) {
            maxArea1 = i;
        }
        else if (area > maxArea2 && area != maxArea1) {
            maxArea2 = i;
        }
    }

    return { rectsBounds.at(maxArea1), rectsBounds.at(maxArea2) };
}

std::vector<int> ShopSplit::sliceFallback(std::vector<int> sliceCords) {
    std::vector<int> valueDefaults = { sliceCords.at(0), sliceCords.at(1), sliceCords.at(sliceCords.size() - 1) };

    return valueDefaults;
}