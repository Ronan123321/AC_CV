#include "ImageRegion.h"
#include "ocr.h"
#include <champnames.h>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <string>
#include <chrono>
#include <ctime>

ChampNames::ChampNames(ocr& ocr): readName(ocr) {}

void ChampNames::updateImageRegion(const cv::Mat& fullImg) {
  
  if (fullImg.empty()) {
    std::cout << "None or empty image given.\n";
    return;
  }
  for (int i = 0; i < 5; ++i) {
	  champNames.at(i).roi = fullImg(champNames.at(i).bounds);
  }
}

std::string ChampNames::getNameStr(int index) {
  if(champNames.size() != 5) {
    std::cout << "ImageRegion amount mismatch\n";
    return "";
  }

  if (champNames.at(index).bounds.empty()) {
    std::cout << "Slot empty or not found\n";
    return "";
  }

 return runOCR(champNames.at(index).roi);
  
}

std::vector<std::string> ChampNames::getAllNames() {
  if(champNames.size() != 5) {
    std::cout << "ImageRegion amount mismatch\n";
    return std::vector<std::string>();
  }
  std::vector<std::string> allNames;

  for(auto& region : champNames) {

    if (region.bounds.empty()) {
        std::cout << "Slot empty or not found\n";
        allNames.push_back("");
        continue;
    }


    allNames.push_back(runOCR(region.roi));
  }

  return allNames;
}

cv::Rect ChampNames::getBound(int index) {
  if(champNames.size() != 5) {
    std::cout << "ImageRegion amount mismatch\n";
    return cv::Rect();
  }

  return champNames.at(index).bounds;
}

std::vector<cv::Rect> ChampNames::getAllBounds() {
  if(champNames.size() != 5) {
    std::cout << "ImageRegion amount mismatch\n";
    return std::vector<cv::Rect>();
  }
  std::vector<cv::Rect> allBounds;

  for(auto& region : champNames) {
    allBounds.push_back(region.bounds);
  }

  return allBounds;

  
}

void ChampNames::locate(std::vector<ImageRegion> champSlotsReg) {
  if(champSlotsReg.size() != 5) {
    std::cout << "ImageRegion amount mismatch";
  }
  else {
      //champSlotsReg.erase(champSlotsReg.begin(), champSlotsReg.begin() + 2);
      //champSlotsReg.erase(champSlotsReg.end() - 2, champSlotsReg.end());
    for(ImageRegion region : champSlotsReg) { // can assume that champSlotsReg is sorted
      ImageRegion champName = region;
      
      std::vector<int> goodMatches = changePerRow(champName.roi.clone());
      std::vector<Interval> combinedMatches = combineRows(goodMatches);
      champName = sliceGoodRows(combinedMatches, champName);

      cv::Rect cutAtGold = fMatchGold(champName.roi.clone());

      if(cutAtGold.empty()) {
        champName.roi = champName.roi; // Keep it the same
        champName.bounds = cv::Rect(0, 0, 0, 0);
      }
      else {
        champName.roi = champName.roi(cutAtGold);

        champName.bounds = cv::Rect(champName.bounds.x,
            champName.bounds.y,
            cutAtGold.width,
            champName.bounds.height);
      }

      champNames.push_back(champName);
    }
  }
}

void ChampNames::testIRVals(cv::Mat fullImg) {
  cv::Mat rectsDrawn = fullImg.clone();

  for(size_t i = 0; i < champNames.size(); i++) {

    cv::imwrite("D:/linux_img_port/Shop_Class/Champ_names/name_" + std::to_string(i) + ".png", champNames.at(i).roi);
    cv::rectangle(rectsDrawn, champNames.at(i).bounds, cv::Scalar(0, 255, 0), 1);
    
  }

  cv::imwrite("D:/linux_img_port/Shop_Class/Champ_names/names_drawn_full.png", rectsDrawn);
}

// Private:

std::vector<int> ChampNames::changePerRow(cv::Mat roi) {
    int k = rowSimKernel;                     // window height
    float diffThresh = rowMaxDiff;       // max allowed mean abs difference per pixel
    float varThresh = groupMaxDiff;      // max allowed mean range across k rows per column

    cv::Mat gray;                  // your ROI in grayscale or a single channel
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

    int H = gray.rows, W = gray.cols;
    std::vector<int> goodRows;     // will collect y where window [y..y+k-1] qualifies
    for (int y = 0; y <= H - k; ++y) {
        // extract window of k rows
        cv::Mat win = gray.rowRange(y, y + k);    // size k × W

        // 1) Row‐to‐row change: mean abs difference
        float totalDiff = 0;
        for (int i = 0; i < k - 1; ++i) {
            cv::Mat d;
            cv::absdiff(win.row(i + 1), win.row(i), d);
            totalDiff += static_cast<float>(cv::mean(d)[0]);
        }
        float meanDiff = totalDiff / (k - 1);

        if (meanDiff > diffThresh)
            continue;   // too much vertical change


        // 2) Inter‐row similarity: for each column, compute range across k rows
        //    then average those ranges
        cv::Mat minVals, maxVals;
        cv::reduce(win, minVals, 0, cv::REDUCE_MIN);
        cv::reduce(win, maxVals, 0, cv::REDUCE_MAX);
        cv::Mat range = maxVals - minVals;       // 1 × W
        float meanRange = static_cast<float>(cv::mean(range)[0]);

        if (meanRange > varThresh)
            continue;   // too much variation across the k rows

        // This window is “flat & uniform”
        goodRows.push_back(y);
    }

    if (goodRows.empty())
        std::cerr << "No good rows found\n";

    return goodRows;
}

std::vector<ChampNames::Interval> ChampNames::combineRows(std::vector<int> goodRows) {
    const int gap = 1;
    std::vector<Interval> ivals;
    ivals.reserve(goodRows.size());
    for (int y : goodRows) {
        ivals.push_back({ y, y + rowSimKernel - 1 });
    }

    // Sort by start (just in case)
    std::sort(ivals.begin(), ivals.end(),
        [](auto& a, auto& b) { return a.start < b.start; });

    // Now merge any intervals that overlap or are within `gap` rows
    std::vector<Interval> merged;
    for (auto& iv : ivals) {
        if (merged.empty()) {
            merged.push_back(iv);
        }
        else {
            auto& last = merged.back();
            // if iv starts before last.end + gap + 1, merge
            if (iv.start <= last.end + gap) {
                // extend the end to encompass iv
                last.end = std::max(last.end, iv.end);
            }
            else {
                // disjoint, start a new cluster
                merged.push_back(iv);
            }
        }
    }

    return merged;
}

ChampNames::Interval ChampNames::bestRows(std::vector<Interval> merged) {
    auto best = std::max_element(
        merged.begin(), merged.end(),
        [](auto& a, auto& b) {
            return (a.end - a.start) < (b.end - b.start);
        }
    );

    Interval largest = { best->start, best->end };

    return largest;
}

ImageRegion ChampNames::sliceGoodRows(std::vector<Interval> goodMatches, ImageRegion fullChamp) {
    if (goodMatches.empty()) {
        cv::Rect localBound = cv::Rect(0, fullChamp.roi.rows * 0.7, fullChamp.roi.cols, fullChamp.roi.rows * 0.3);
        cv::Rect globalBound = cv::Rect(fullChamp.bounds.x + localBound.x,
            fullChamp.bounds.y + localBound.y,
            localBound.width,
            localBound.height);
        return ImageRegion(fullChamp.roi(localBound).clone(), globalBound);
    }

    if (goodMatches.size() == 1 && goodMatches.at(0).start > fullChamp.roi.rows * 0.5) {
        cv::Rect localBounds = cv::Rect(0, goodMatches.at(0).start, fullChamp.roi.cols, fullChamp.roi.rows - goodMatches.at(0).start);
        cv::Rect globalBounds = cv::Rect(fullChamp.bounds.x + localBounds.x,
            fullChamp.bounds.y + localBounds.y,
            localBounds.width,
            localBounds.height);
        return ImageRegion(fullChamp.roi(localBounds).clone(), globalBounds);
    }

    if (goodMatches.size() == 2 && goodMatches.at(0).start > fullChamp.roi.rows * 0.5) {
        cv::Rect localBounds = cv::Rect(0, goodMatches.at(0).start, fullChamp.roi.cols, goodMatches.at(1).end - goodMatches.at(0).start);
        cv::Rect globalBounds = cv::Rect(fullChamp.bounds.x + localBounds.x,
            fullChamp.bounds.y + localBounds.y,
            localBounds.width,
            localBounds.height);
        return ImageRegion(fullChamp.roi(localBounds).clone(), globalBounds);
    }

    if (goodMatches.size() > 2) {
        Interval match = bestRows(goodMatches);
        cv::Rect localBounds = cv::Rect(0, match.start, fullChamp.roi.cols, fullChamp.roi.rows - match.start);
        cv::Rect globalBounds = cv::Rect(fullChamp.bounds.x + localBounds.x,
            fullChamp.bounds.y + localBounds.y,
            localBounds.width,
            localBounds.height);
        return ImageRegion(fullChamp.roi(localBounds).clone(), globalBounds);
    }

    return ImageRegion();
}

cv::Rect ChampNames::fMatchGold(const cv::Mat& nameRegion) {
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    cv::Mat goldTemplate = cv::imread(goldTemplatePath, cv::IMREAD_GRAYSCALE);
    cv::Mat goldImage;
    cv::cvtColor(nameRegion.clone(), goldImage, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> templateKeypoints, imageKeypoints;
    cv::Mat templateDescriptor, imageDescriptor;
    sift->detectAndCompute(goldTemplate, cv::noArray(), templateKeypoints, templateDescriptor);
    sift->detectAndCompute(goldImage, cv::noArray(), imageKeypoints, imageDescriptor);



    if (templateDescriptor.empty() || imageDescriptor.empty()) {
        std::cerr << "Empty slot found!\n";
        return cv::Rect(0, 0, 0, 0);
    }

    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(templateDescriptor, imageDescriptor, matches);

    double maxDist = 0;
    for (auto& m : matches) {
        maxDist = std::max(maxDist, (double)m.distance);
    }

    std::vector<cv::DMatch> goodMatches;
    cv::DMatch closest;
    closest.distance = 9999;
    for (auto& m : matches) {
        if (m.distance < 0.7 * maxDist) {
            goodMatches.push_back(m);
            if (m.distance < closest.distance)
                closest = m;
        }
    }

    if (goodMatches.size() <= 0) {
        // Backup incase earlier empty slot check doesnt catch it
        return cv::Rect(0, 0, 0, 0);
    }
    
    cv::Point2f matchedPoint = imageKeypoints[closest.trainIdx].pt;
    if (matchedPoint.x < nameRegion.cols * minNameLength) {
		matchedPoint.x = nameRegion.cols * minNameLength;
    }

    return cv::Rect(0, 0, matchedPoint.x, nameRegion.rows);
}

std::string ChampNames::runOCR(cv::Mat nameImg) {
    cv::Mat ocrMask = readName.preprocessWhite(nameImg);
    return readName.readText(ocrMask);
}


ImageRegion ChampNames::bottomTraitIcon(ImageRegion fullChampReg) {
  cv::Mat hsv;
  cv::cvtColor(fullChampReg.roi, hsv, cv::COLOR_BGR2HSV);

  cv::Mat nameBoxMask;
  cv::inRange(hsv, traitMin, traitMax, nameBoxMask);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(nameBoxMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  const double fullArea = double(hsv.rows) * double(hsv.cols);
  int lowestTrait = 0;
  for(auto& cont : contours) {
    int bottom = cv::boundingRect(cont).y + cv::boundingRect(cont).height;
    int area = cv::contourArea(cont); // doesnt have to be double becuase not used in operations

    if(bottom > lowestTrait && area > fullArea * traitMinArea && area < fullArea * traitMaxArea)
      lowestTrait = bottom;
  }
    
  ImageRegion nameRegion;
  cv::Rect nameRect = cv::Rect(0,
                               lowestTrait,
                               fullChampReg.roi.cols,
                               fullChampReg.roi.rows - lowestTrait); // This has to be roi.x because it proportional to the last image
  
  nameRegion.roi = fullChampReg.roi(nameRect).clone();
  nameRegion.bounds = cv::Rect(fullChampReg.bounds.x,
                               fullChampReg.bounds.y + lowestTrait,
                               fullChampReg.bounds.width,
                               fullChampReg.bounds.height - lowestTrait); // has to use bounds because its building on the global bounds

  return nameRegion;
}

ImageRegion ChampNames::chopGold(ImageRegion belowTraitReg) {
  cv::Mat hsv;
  cv::cvtColor(belowTraitReg.roi, hsv, cv::COLOR_BGR2HSV);

  cv::Mat goldMask;
  cv::inRange(hsv, goldMin, goldMax, goldMask);
  
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(goldMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  const double fullArea = double(hsv.rows) * double(hsv.cols);
  int goldXLine = 9999; //Sentinel for x cordinate of gold icon
  for(auto& cont : contours) {
    int xCord = cv::boundingRect(cont).x;
    int area = cv::contourArea(cont);

    if(xCord < goldXLine && area > fullArea * goldMinArea && area < fullArea * goldMaxArea)
      goldXLine = xCord;
  }

  ImageRegion nameRegionCropped;
  cv::Rect nameRect = cv::Rect(0, 0, goldXLine, belowTraitReg.roi.rows);

  nameRegionCropped.roi = belowTraitReg.roi(nameRect).clone();
  
  nameRegionCropped.bounds = cv::Rect(belowTraitReg.bounds.x,
                                      belowTraitReg.bounds.y,
                                      goldXLine,
                                      belowTraitReg.bounds.height);
  
  return nameRegionCropped;
}