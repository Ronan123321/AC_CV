#include "ImageRegion.h"
#include "tesseractOcr.h"
#include <atomic>
#include <iostream>
#include <champslots.h>
#include <shopsplit.h>
#include <opencv2/opencv.hpp>
#include <ostream>


const cv::Rect rightShop_zero_three = cv::Rect(294, 65, 1348, 204);
const cv::Rect rightShop_two_three = cv::Rect(294, 65, 1348, 204);
const cv::Rect rightShop_6_76 = cv::Rect(294, 65, 1348, 204);

const cv::Rect leftShop_two_three = cv::Rect( 19, 65, 275, 204 );
const cv::Rect leftShop_one_three = cv::Rect( 16, 50, 206, 154 );

int main(int argc, char* argv[]) {
  std::cout << "Elements main running\n";
  cv::Mat shopImg = cv::imread("D:/linux_img_port/Fullscreen_different_contrast.png");

  ShopSplit mySplit;

  mySplit.locate(ImageRegion(shopImg, cv::Rect(0, 0, shopImg.cols, shopImg.rows)));
  mySplit.testIRVals(shopImg);

  
  return 0;
}
