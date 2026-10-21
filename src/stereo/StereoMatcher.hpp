#pragma once
// Step 2 - stereo matching along epipolar lines of a RECTIFIED pair
// (Python: stereo_matching.py: zncc_along_row, search, match)
// Images are CV_64F, single channel.

#include <vector>
#include <opencv2/core.hpp>
#include "../config/Config.hpp"
#include "../core/Types.hpp"

namespace pe {

    // Result of searching one patch along one row
    struct RowSearch {
        bool valid = false;          // false: too few candidates or peak on the range edge
        double disparity = 0;        // sub-pixel
        double best = -1;            // best ZNCC
        double second = -1;          // best ZNCC at least peakExclusion px away
        std::vector<int> d;          // optional: the whole curve (for plots / debugging)
        std::vector<double> curve;
    };

    // ZNCC of two equally sized patches
    double zncc(const cv::Mat& patchA, const cv::Mat& patchB);

    // Search the patch around (x, y) of `src` along row y of `dst`.
    // direction +1: left -> right image (candidate x_dst = x - d)
    // direction -1: right -> left image (candidate x_dst = x + d)
    RowSearch searchRow(const cv::Mat& src, const cv::Mat& dst, int x, int y, int direction,
                        const StereoParams& p, bool keepCurve = false);

    // The whole of Step 2: match every left corner, with all acceptance tests.
    std::vector<StereoMatch> matchCorners(const cv::Mat& left, const cv::Mat& right,
                                          const std::vector<Corner>& corners,
                                          const StereoParams& p, MatchStats* stats = nullptr);

}  // namespace pe
