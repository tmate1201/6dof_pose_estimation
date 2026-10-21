#pragma once
// Data passed between pipeline steps. Images themselves are cv::Mat.

namespace pe {

    // Step 1 output: one detected corner
    struct Corner {
        int x = 0, y = 0;        // pixel position in the left image
        double score = 0;        // Shi-Tomasi score (min eigenvalue)
    };

    // Step 2 output: one left-right correspondence on the same image row
    struct StereoMatch {
        int xL = 0, y = 0;       // corner in the left image
        double xR = 0;           // sub-pixel position in the right image
        double disparity = 0;    // xL - xR
        double zncc = 0;         // best ZNCC score
        double secondBest = 0;   // best score of the competing peak
    };

    // Why corners were dropped in Step 2 (for tuning and the report)
    struct MatchStats {
        int corners = 0;
        int noPeak = 0;          // no valid candidate / peak on the edge of the range
        int lowScore = 0;        // ZNCC < minZncc
        int ambiguous = 0;       // second peak too close
        int failedLR = 0;        // left-right check failed
        int matched = 0;
    };


}  
