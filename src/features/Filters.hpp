#pragma once
// Basic neighbourhood operations, written by hand (Python: filter2d, max_filter).
// cv::Mat is only the container here, like a NumPy array in the Python version.

#include <opencv2/core.hpp>

namespace pe {

// 3x3 Sobel kernels (CV_64F)
cv::Mat sobelX();   // [-1 0 1; -2 0 2; -1 0 1]
cv::Mat sobelY();   // transpose of sobelX

// Correlation with a small kernel, reflect101 borders (NumPy mode="reflect").
// out(y, x) = sum_ij k(i, j) * img(y + i - ry, x + j - rx)
cv::Mat filter2d(const cv::Mat& img, const cv::Mat& kernel);

// Maximum over a size x size window, replicated borders (NumPy mode="edge")
cv::Mat maxFilter(const cv::Mat& img, int size);

}  // namespace pe
