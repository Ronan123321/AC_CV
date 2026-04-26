#include <ImageRegion.h>

cv::Rect ImageRegion::localBounds(const cv::Rect& localRegion) {
  cv::Rect localBound;

  localBound = cv::Rect(localRegion.x - bounds.x, // local cords will always be higher in val
                        localRegion.y - bounds.y,
                        localRegion.width,
                        localRegion.height);
  return localBound;
}
