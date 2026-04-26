#include "tesseractOcr.h"
#include <iostream>
#include <string>
#include <regex>
#include <cmath>


#include <opencv2/opencv.hpp>
std::string tesseractOcr::readText(cv::Mat textImg)  {
  tesseract::TessBaseAPI api;
  api.SetVariable("debug_file", "NUL"); // On Linux
  verifyGrayscale(textImg);
  if(api.Init(NULL, "eng")) {
    std::cout << "Could not initialize tesseract.\n";
    return "";
  }

  api.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
  api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
  api.SetImage(textImg.data, textImg.cols, textImg.rows, 1, textImg.step); // here is for color channels. Maybe change to 1
  api.SetSourceResolution(300);

  std::string foundName = std::string(api.GetUTF8Text());
  int conf = api.MeanTextConf();
    
  if (conf < 5) {
	  return ""; // If the confidence is too low, return empty string
  }
  foundName.erase(std::remove_if(foundName.begin(), foundName.end(), ::isspace), foundName.end());

  api.End();
  return foundName;
}

// Function for reading numbers, with a whitelist of digits only and specific engine and page segmentation mode for better accuracy on numbers
std::string tesseractOcr::readNum(cv::Mat textImg) {
  verifyGrayscale(textImg);
  if(api.Init(NULL, "eng")) {
    std::cout << "Could not initialize tesseract.\n";
    return "";
  }

  api.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
  api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
  api.SetVariable("tessedit_char_whitelist", "0123456789");
  api.SetImage(textImg.data, textImg.cols, textImg.rows, 1, textImg.step);
  api.SetSourceResolution(300);

  std::string foundNum = std::string(api.GetUTF8Text());
  foundNum.erase(std::remove_if(foundNum.begin(), foundNum.end(), ::isspace), foundNum.end());
  
  api.End();
  return foundNum;
}

// For reading fractions, this is basically only for level progression, whitelist must include '/'
std::string tesseractOcr::readFrac(cv::Mat textImg)  {
  verifyGrayscale(textImg);
  if (api.Init(NULL, "eng")) {
    std::cout << "Could not initialize tesseract.\n";
    return "";
  }
  
  api.Init(NULL, "eng", tesseract::OEM_TESSERACT_LSTM_COMBINED);
  api.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
  api.SetVariable("tessedit_char_whitelist", "0123456789/");
  api.SetImage(textImg.data, textImg.cols, textImg.rows, 1, textImg.step);
  api.SetSourceResolution(300);

  std::string foundNum = std::string(api.GetUTF8Text());
  foundNum.erase(std::remove_if(foundNum.begin(), foundNum.end(), ::isspace), foundNum.end());

  if(std::stoi(foundNum) < 2) {
	  textImg = dilateMask(textImg); // If the number is too small, it will be dilated to make it more readable
	  api.SetImage(textImg.data, textImg.cols, textImg.rows, 1, textImg.step);
	  foundNum = std::string(api.GetUTF8Text());
      foundNum.erase(std::remove_if(foundNum.begin(), foundNum.end(), ::isspace), foundNum.end());
  }

  foundNum = verifySlash(foundNum);

  api.End();
  return foundNum;
}

// Preprocessing for white text, this is basically just a color filter to make the text more visible, then it is resized to make it easier for tesseract to read
cv::Mat tesseractOcr::preprocessWhite(const cv::Mat& rawImg)  {
  const cv::Scalar scalarFloor = cv::Scalar(0, 0, 140);
  const cv::Scalar scalarRoof = cv::Scalar(175, 70, 255);

  cv::Mat hsv;
  cv::cvtColor(rawImg, hsv, cv::COLOR_BGR2HSV);

  cv::Mat whiteMask;
  cv::inRange(hsv, scalarFloor, scalarRoof, whiteMask);

  cv::bitwise_not(whiteMask, whiteMask);

  cv::resize(whiteMask, whiteMask, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);

  std::vector<int> x = { };
  std::vector<int> y = { };
  for (int i = 0; i < x.size(); i++) {
      cv::Vec3b pixel = hsv.at<cv::Vec3b>(y[i], x[i]);
      std::cout << "Pixel at (" << x[i] << ", " << y[i] << "): " << pixel << std::endl;
  }

  return whiteMask;
}

// Preprocessing for blue text, most UI elements will be this or white at least when tesseract sees it
cv::Mat tesseractOcr::preprocessBlue(const cv::Mat& rawImg)  {
  const cv::Scalar scalarFloor  = cv::Scalar(85, 20, 80);
  const cv::Scalar scalarRoof = cv::Scalar(93, 60, 220);

  cv::Mat hsv;
  cv::cvtColor(rawImg, hsv, cv::COLOR_BGR2HSV);

  cv::Mat hsvThresh;
  cv::inRange(hsv, scalarFloor, scalarRoof, hsvThresh);

  cv::resize(hsvThresh, hsvThresh, {}, 2.0, 2.0, cv::INTER_LINEAR);

  cv::bitwise_not(hsvThresh, hsvThresh);

  std::vector<int> x = { };
  std::vector<int> y = { };
  for (int i = 0; i < x.size(); i++) {
      cv::Vec3b pixel = hsv.at<cv::Vec3b>(y[i], x[i]);
      std::cout << "Pixel at (" << x[i] << ", " << y[i] << "): " << pixel << std::endl;
  }

  //cv::imshow("hsvThresh", hsvThresh);
  //cv::waitKey(0); for debugging

  return hsvThresh;
}

// Levenshtein distance is used to compare the found text with expected text by counting required edits to get from one to the other
int tesseractOcr::levenshteinDistance(const std::string& base, const std::string& compare) {
    int m = base.length();
    int n = compare.length();
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));

    for (int i = 0; i <= m; ++i) {
        dp[i][0] = i;
    }
    for (int j = 0; j <= n; ++j) {
        dp[0][j] = j;
    }

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int cost = (base[i - 1] == compare[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i - 1][j] + 1,      // Deletion
                                dp[i][j - 1] + 1,      // Insertion
                                dp[i - 1][j - 1] + cost}); // Substitution
        }
    }
    return dp[m][n];
}

// This had to be made because the slash was inconsistent when being treated by OpenCV
std::string tesseractOcr::verifySlash(const std::string& foundText) {
  std::string slashText = foundText;

  std::cout << "Verifying slash in: " << slashText << std::endl;

  if (slashText.find('/') == std::string::npos) {
      slashText = insertSlash(slashText);
      std::cout << "No slash found. Defaulted to: " << slashText << std::endl;
  }

  return slashText;
}


// Very specific to this game, based on restrictions it can determine if there should be a slash
// or if its not possible for a slash to inserted which means the slas was probably read as 1 or 7
std::string tesseractOcr::insertSlash(const std::string& noSlash) {
  std::string withSlash = noSlash;
  int slashLocation = ceil(noSlash.size() / 2.0f) - 1.0f; // -1 because of 0 indexing

  if (noSlash.size() > 5 || noSlash.size() < 3) {
    withSlash.insert(withSlash.begin() + ceil(noSlash.size() / 2.0f), '/');
    return withSlash; // No possible place for slash
  }

  double beforeSlash = std::stoi(noSlash.substr(0, slashLocation));
  double afterSlash = std::stoi(noSlash.substr(slashLocation));

  if (noSlash[slashLocation] == '1' || noSlash[slashLocation] == '7') {// maybe just change so its anything thats not /
    if (beforeSlash / afterSlash >= 1 || afterSlash < 1) { // if the denom is larger than the num, then the slash was probably missed, it then insterts a slash in the middle
        withSlash.insert(withSlash.begin() + slashLocation, '/');
        return withSlash;
    }

    withSlash[slashLocation] = '/';
    return withSlash;
  }

  if(std::stoi(noSlash) % 2 == 0) 
    withSlash.insert(withSlash.begin() + slashLocation + 1, '/');
  else
    withSlash.insert(withSlash.begin() + slashLocation, '/');

   return withSlash;
}


// This is used to dilate the image when the text is too small it makes it more readable for tesseract but it can also make it less accurate if overused
cv::Mat tesseractOcr::dilateMask(const cv::Mat& mask) {
  cv::Mat dilateMask = cv::getStructuringElement(cv::MORPH_RECT, { 2, 2 });

  cv::dilate(mask, mask, dilateMask, {}, 1);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, dilateMask, {}, 1); // Maybe remove this dilation, it might be too much

  return mask;
}

void tesseractOcr::verifyGrayscale(cv::Mat& img) {
  if(img.channels() == 3) {
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  }
}
