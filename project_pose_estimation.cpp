// project_pose_estimation.cpp : pipeline driver.
// The algorithms live in src/ (written by hand); this file loads the images,
// runs the steps in order and shows/saves the results with OpenCV.
//
// Usage:  project_pose_estimation.exe [left.png right.png] [--no-show]
//         (default: data/cube_L.png data/cube_R.png, relative to the working directory)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "src/report/ReportFigures.hpp"
#include "src/config/Config.hpp"
#include "src/features/ShiTomasi.hpp"
#include "src/features/Harris.hpp"
#include "src/stereo/StereoMatcher.hpp"
#include "Object_isolation.hpp"
#include "PlaneFit.hpp"
#include "PlaneFitTest.hpp"

namespace {

    // Read an image as grayscale double (like cv2.imread(...).astype(np.float64))
    cv::Mat loadGray(const std::string& path) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) throw std::runtime_error("cannot read image: " + path);
        img.convertTo(img, CV_64F);   // double: gradients can be negative
        return img;
    }

    // |img| scaled so the 99.5th percentile becomes 255 (Python: norm_image)
    cv::Mat normImage(const cv::Mat& img) {
        cv::Mat a = cv::abs(img);
        std::vector<double> v(a.begin<double>(), a.end<double>());
        const size_t k = static_cast<size_t>(0.995 * (v.size() - 1));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        cv::Mat out;
        a.convertTo(out, CV_8U, 255.0 / std::max(v[k], 1e-12));   // saturates at 255
        return out;
    }

    // Corners as green circles on the gray image
    cv::Mat drawCorners(const cv::Mat& gray, const std::vector<pe::Corner>& corners) {
        cv::Mat bgr;
        gray.convertTo(bgr, CV_8U);
        cv::cvtColor(bgr, bgr, cv::COLOR_GRAY2BGR);
        for (const auto& c : corners) cv::circle(bgr, {c.x, c.y}, 3, {0, 255, 0}, -1, cv::LINE_AA);
        return bgr;
    }

    // Left and right side by side, one horizontal line per match (colour = disparity)
    cv::Mat drawMatches(const cv::Mat& left, const cv::Mat& right, const std::vector<pe::StereoMatch>& matches,
                        const pe::StereoParams& p) {
        cv::Mat both, bgr;
        cv::hconcat(left, right, both);
        both.convertTo(bgr, CV_8U);
        cv::cvtColor(bgr, bgr, cv::COLOR_GRAY2BGR);

        cv::Mat ramp(1, 256, CV_8U), colours;                      // 256-colour lookup table
        for (int i = 0; i < 256; ++i) ramp.at<uchar>(i) = static_cast<uchar>(i);
        cv::applyColorMap(ramp, colours, cv::COLORMAP_JET);

        for (const auto& m : matches) {
            const double t = std::clamp((m.disparity - p.dMin) / double(p.dMax - p.dMin), 0.0, 1.0);
            const cv::Vec3b c = colours.at<cv::Vec3b>(0, static_cast<int>(t * 255));
            const cv::Point pl(m.xL, m.y), pr(static_cast<int>(std::lround(m.xR)) + left.cols, m.y);
            cv::line(bgr, pl, pr, cv::Scalar(c[0], c[1], c[2]), 1, cv::LINE_AA);
            cv::circle(bgr, pl, 3, {0, 255, 0}, -1, cv::LINE_AA);
            cv::circle(bgr, pr, 3, {0, 255, 0}, -1, cv::LINE_AA);
        }
        return bgr;
    }

    void showCluster(const cv::Mat& left, const std::vector<cv::Point3d>& cluster, const pe::CameraParams& cam,
        const std::string& windowName = "Cluster", bool show = true, cv::Mat* out = nullptr) {
        // grayscale (CV_64F or CV_8U) -> 8-bit colour image we can draw on
        cv::Mat img;
        left.convertTo(img, CV_8U);
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);

        double zMin = 1e9, zMax = 0;
        for (const cv::Point3d& P : cluster) {
            zMin = std::min(zMin, P.z);
            zMax = std::max(zMax, P.z);
        }

        for (const cv::Point3d& P : cluster) {
            const cv::Point px = pe::project(P, cam);
            // colour by depth: near = red, far = blue
            const double t = (zMax > zMin) ? (P.z - zMin) / (zMax - zMin) : 0.0;
            const cv::Scalar colour(255 * t, 0, 255 * (1 - t));          // BGR
            cv::circle(img, px, 5, colour, cv::FILLED, cv::LINE_AA);
            cv::circle(img, px, 5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }

        const std::string info = std::to_string(cluster.size()) + " points, Z " +
            std::to_string(cvRound(zMin)) + "-" + std::to_string(cvRound(zMax)) +
            " mm (red = near, blue = far)";
        cv::putText(img, info, { 10, 30 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 4, cv::LINE_AA);
        cv::putText(img, info, { 10, 30 }, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        if (out) *out = img;
        if (show) {
            cv::imshow(windowName, img);
            cv::waitKey(0);
        }
    }

    void showFaces(const cv::Mat& left, const std::vector<pe::Plane>& faces, const std::vector<cv::Point3d>& points, const pe::CameraParams& cam) {
        cv::Mat img;
        left.convertTo(img, CV_8U);
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        
        std::vector<cv::Scalar> colors = {cv::Scalar(255,0,0), cv::Scalar(0,255,0), cv::Scalar(0,0,255)};

        for (int i = 0; i < faces.size(); i++) {
            for (int k = 0; k < faces[i].indices.size(); k++) {

                const cv::Point px = pe::project(points[faces[i].indices[k]], cam);
                cv::circle(img, px, 5, colors[i], cv::FILLED, cv::LINE_AA);


            }
        }
        cv::imshow("windowName", img);
        cv::waitKey(0);
    }
    // Same 
    /*void saveMatchesCsv(const std::string& path, const std::vector<pe::StereoMatch>& matches) {
        std::ofstream f(path);
        f << "xL,y,xR,disparity,zncc,second_best\n" << std::fixed << std::setprecision(3);
        for (const auto& m : matches)
            f << double(m.xL) << ',' << double(m.y) << ',' << m.xR << ',' << m.disparity << ','
              << m.zncc << ',' << m.secondBest << '\n';
    }*/

    // Run f, print how long it took
    template <typename F>
    auto timed(const char* name, F&& f) {
        const auto t0 = std::chrono::steady_clock::now();
        auto result = f();
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::cout << "  " << name << ": " << ms << " ms\n";
        return result;
    }

}  // namespace

int main(int argc, char** argv) {
    try {
        return pe::report::writeTriangulationReport("data", "results/report");
        std::vector<std::string> args(argv + 1, argv + argc);
        const bool show = std::find(args.begin(), args.end(), "--no-show") == args.end();
        args.erase(std::remove(args.begin(), args.end(), "--no-show"), args.end());
        const std::string leftPath = args.size() >= 2 ? args[0] : "data/left_001.png";
        const std::string rightPath = args.size() >= 2 ? args[1] : "data/right_001.png";
        const std::string outDir = "results";
        std::filesystem::create_directories(outDir);

        pe::Config cfg;   // all parameters, see src/config/Config.hpp

        // ---- input ----
        const cv::Mat left = loadGray(leftPath);
        const cv::Mat right = loadGray(rightPath);
        std::cout << "Images: " << left.cols << " x " << left.rows << "\n";

        // ---- Step 1: Shi-Tomasi corners in the left image ----
        cv::Mat score;
        const int border = cfg.stereo.half + 1;   // Step 2 needs a full patch around each corner
        const auto corners = timed("Step 1 corners", [&] {
            return pe::detectCorners(left, cfg.corners, border, &score);
        });

        std::cout << "Step 1: " << corners.size() << " corners\n";
        const cv::Mat scoreImg = normImage(score);
        const cv::Mat cornerImg = drawCorners(left, corners);
        cv::imwrite(outDir + "/step1_score.png", scoreImg);
        cv::imwrite(outDir + "/step1_corners.png", cornerImg);

        // ---- Step 2: stereo matching ----
        pe::MatchStats st;
        const auto matches = timed("Step 2 matching", [&] {
            return pe::matchCorners(left, right, corners, cfg.stereo, &st);
        });
        std::cout << "Step 2: " << st.matched << " of " << st.corners << " corners matched\n"
                  << "  rejected: " << st.noPeak << " no valid peak, " << st.lowScore << " low ZNCC, "
                  << st.ambiguous << " ambiguous, " << st.failedLR << " failed left-right check\n";
        const cv::Mat matchImg = drawMatches(left, right, matches, cfg.stereo);
        //saveMatchesCsv(outDir + "/matches.csv", matches);
        cv::imwrite(outDir + "/step2_matches.png", matchImg);

        // ---- Step 3 onwards: triangulation, cube segmentation, planes, pose ----
        // (next modules: src/reconstruction, src/geometry, ...)
        std::vector<cv::Point3d> cluster = pe::dbScan(matches, cfg.camera, 25, 4);
        std::vector<pe::Plane> faces = pe::fitFaces(cluster, cfg.planes);

        std::cout << "FACES: " << faces.size() << std::endl;

        showFaces(left, faces, cluster, cfg.camera);

        std::cout << "Results written to " << outDir << "/\n";
        if (show) {
            cv::imshow("Step 1 - score", scoreImg);
            cv::imshow("Step 1 - corners", cornerImg);
            cv::imshow("Step 2 - matches", matchImg);
            cv::waitKey(0);   // any key closes the windows
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
}
