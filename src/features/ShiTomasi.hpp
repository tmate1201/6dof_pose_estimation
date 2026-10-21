#pragma once
// Step 1 - Shi-Tomasi corner detection
// (Python: corner_detection_shi_tomasi.py + pick_corners / shi_tomasi_points)
// All images are CV_64F (double), single channel.

#include <vector>
#include <opencv2/core.hpp>
#include "../config/Config.hpp"
#include "../core/Types.hpp"

namespace pe {

    struct Gradients {
        cv::Mat Ix, Iy;
    };

    struct StructureTensor {
        cv::Mat Mxx, Myy, Mxy;   // window sums of Ix*Ix, Iy*Iy, Ix*Iy
    };

    Gradients sobelGradients(const cv::Mat& gray);
    StructureTensor structureTensor(const Gradients& g, int window);

    // Smaller eigenvalue of [[Mxx, Mxy], [Mxy, Myy]] at every pixel
    cv::Mat minEigenvalue(const StructureTensor& m);

    // Score image -> corners: threshold, 3x3 local maximum, border, min distance
    std::vector<Corner> pickCorners(const cv::Mat& score, const CornerParams& p, int border);

    // The whole of Step 1. `border` keeps corners far enough from the edge for Step 2's patch.
    std::vector<Corner> detectCorners(const cv::Mat& gray, const CornerParams& p, int border,
                                      cv::Mat* scoreOut = nullptr);

}  // namespace pe
