#pragma once
// Step 1 (alternative) - Harris corner detection
// Drop-in replacement for detectCorners() in ShiTomasi.hpp: same gradients, same
// structure tensor and same corner picking, only the score function differs:
//   Shi-Tomasi: lambda_min
//   Harris:     R = det(M) - k * tr(M)^2
// All images are CV_64F (double), single channel.

#include <vector>
#include <opencv2/core.hpp>
#include "../config/Config.hpp"
#include "../core/Types.hpp"
#include "ShiTomasi.hpp"   // Gradients, StructureTensor, sobelGradients, structureTensor, pickCorners

namespace pe {

    // Harris response R = det(M) - k * tr(M)^2 at every pixel
    // (positive at corners, negative on edges, close to 0 in flat regions)
    cv::Mat harrisResponse(const StructureTensor& m, double k);

    // The whole of Step 1 with the Harris score. Same arguments as detectCorners().
    // Note: R grows with contrast^4 (lambda_min with contrast^2), so the same
    // CornerParams::quality is much stricter here than for Shi-Tomasi.
    std::vector<Corner> detectCornersHarris(const cv::Mat& gray, const CornerParams& p, int border,
                                            cv::Mat* scoreOut = nullptr);

}  // namespace pe
