#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

struct ImageRegion { // TODO look into using rvalues to optimize this
  ImageRegion() = default;
  ImageRegion(cv::Mat roi, cv::Rect bounds) : roi(roi.clone()), bounds(bounds) {}
  ImageRegion(const ImageRegion& ir) : roi(ir.roi.clone()), bounds(ir.bounds) {}

  ImageRegion(ImageRegion&&) = default;
  ImageRegion& operator=(ImageRegion&&) = default;


  ImageRegion& operator=(const ImageRegion& region) {
    if (this != &region) {
      roi = region.roi.clone();
      bounds = region.bounds;
    }
    return *this;
  }

  cv::Rect localBounds(const cv::Rect&);

  cv::Mat roi;
  cv::Rect bounds;

};
