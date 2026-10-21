#pragma once
// Every tunable number of the pipeline in one place.
// Add a new struct here for each new step (triangulation, plane fitting, ...).

namespace pe {

    // Step 1 - Shi-Tomasi corners
    struct CornerParams {
        int window = 5;          // integration window (block size) for the structure tensor
        double quality = 0.01;   // keep scores above quality * max score
        int minDist = 6;         // min distance between corners (px)
        int maxCorners = 600;
    };

    // Step 2 - stereo matching along epipolar lines
    struct StereoParams {
        int half = 7;            // patch half size -> (2*half+1)^2 patch
        int dMin = 20;           // disparity search range (px)
        int dMax = 450;
        double minZncc = 0.8;    // a match must correlate at least this well
        double minUnique = 0.05; // best peak must beat the 2nd peak by this much
        int peakExclusion = 3;   // 2nd peak must be > this many px away from the best
        double lrTol = 1.0;      // allowed left-right disagreement (px)
    };

    // Step 3 - camera (ASSUMED values: read the real ones from the RealSense)
    struct CameraParams {
        double fx = 640.0, fy = 640.0;   // focal length (px)
        double cx = 640.0, cy = 360.0;   // principal point (px)
        double baseline = 95.0;          // mm
    };

    struct Config {
        CornerParams corners;
        StereoParams stereo;
        CameraParams camera;
    };

}  // namespace pe
