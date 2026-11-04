#pragma once
// Report material for the "Feature Detection" and "Stereo Correspondence" subsections
// "Triangulation", "Plane Fitting" and "Pose Estimation" subsections (Rubik's cube test).
// Regenerates every figure and every number quoted there, using the pipeline's own
// Step 1-4 code (ShiTomasi, Harris, StereoMatcher, Triangulation, dbScan, fitFaces). OpenCV is only used to load, draw and save.
//
// Call it once, e.g. at the start of main():
//     #include "src/report/ReportFigures.hpp"
//     pe::report::writeFeatureDetectionReport("data", "results/report");
//     pe::report::writeStereoCorrespondenceReport("data", "results/report");
//     pe::report::writeTriangulationReport("data", "results/report");
//     pe::report::writePlaneFittingReport("data", "results/report");
//     pe::report::writePoseEstimationReport("data", "results/report");

#include <string>

namespace pe::report {

    // Reads <dataDir>/left_001.png and <dataDir>/right_001.png, writes to <outDir>:
    //   rubik_corners_shitomasi.png  rubik_corners_harris.png  (corner overlays, q = 0.01)
    //   rubik_score_log.png          (log10 relative score maps + colour bar)
    //   rubik_gradients.png          (Sobel I_u | I_v)
    //   report_values.txt            (all numbers of the section; also printed to the console)
    // Returns 0 on success, 1 if an image could not be read.
    int writeFeatureDetectionReport(const std::string& dataDir, const std::string& outDir);

    // Same for the "Stereo Correspondence" subsection. Writes to <outDir>:
    //   rubik_matches.png       (left | right, one line per accepted match, colour = disparity)
    //   rubik_zncc_curves.png   (ZNCC along the row for three example corners)
    //   stereo_values.txt       (all numbers of the section; also printed to the console)
    int writeStereoCorrespondenceReport(const std::string& dataDir, const std::string& outDir);

    // Same for the "Triangulation" subsection. Writes to <outDir>:
    //   rubik_points3d.png        (left: points on the left image, colour = depth;
    //                              right: top view X-Z of the cube cluster)
    //   triangulation_values.txt  (all numbers of the section; also printed to the console)
    int writeTriangulationReport(const std::string& dataDir, const std::string& outDir);

    // Same for the "Plane Fitting" subsection. Writes to <outDir>:
    //   rubik_planes.png          (points on the left image coloured by face, with the face normals)
    //   planes_values.txt         (all numbers of the section; also printed to the console)
    int writePlaneFittingReport(const std::string& dataDir, const std::string& outDir);

    // Same for the "Pose Estimation" subsection. Writes to <outDir>:
    //   rubik_pose.png            (left: faces after plane fitting, right: fitted cube after pose estimation)
    //   pose_values.txt           (all numbers of the section; also printed to the console)
    int writePoseEstimationReport(const std::string& dataDir, const std::string& outDir);

}  // namespace pe::report
