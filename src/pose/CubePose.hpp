#pragma once
#include <vector>
#include <algorithm>
#include <limits>
#include <opencv2/core.hpp>
#include "../config/Config.hpp"
#include "../core/Types.hpp"

namespace pe {

    cv::Matx33d skew(const cv::Vec3d& w);                         // [w]x
    cv::Matx33d rodrigues(const cv::Vec3d& w);                    // exp([w]x)
    cv::Matx33d nearestRotation(const cv::Matx33d& N, bool dirty = false);            // polar iteration X <- (X + X^-T)/2
    bool solve6(cv::Matx66d H, cv::Vec6d g, cv::Vec6d& x);        // H x = g, Gaussian elimination

    inline cv::Vec3d col(const cv::Matx33d& M, int k) { return { M(0, k), M(1, k), M(2, k) }; }
    inline cv::Vec3d vec(const cv::Point3d& p) { return { p.x, p.y, p.z }; }

    // --- step 6 ---
    // label[i] = face index of point i (-1 = none); reorders faces to a right-handed set
    std::vector<int> faceLabels(const std::vector<cv::Point3d>& pts, std::vector<Plane>& faces);

    CubePose initialPose(const std::vector<cv::Point3d>& pts, const std::vector<int>& label, const std::vector<Plane>& faces, double edge);

    // residual r_i = a_k . (p_i - t) - edge/2 for every labelled point
    std::vector<double> residuals(const std::vector<cv::Point3d>& pts, const std::vector<int>& label, const CubePose& pose, double edge);

    void refineLM(const std::vector<cv::Point3d>& pts, const std::vector<int>& label, CubePose& pose, const PoseParams& p);

    std::vector<int> reassign(const std::vector<cv::Point3d>& pts, const CubePose& pose, int nFaces, const PoseParams& p);

    // the whole of step 6
    // labelsOut (optional): final face of every point after re-association (-1 = none)
    CubePose estimatePose(const std::vector<cv::Point3d>& pts, std::vector<Plane> faces, const PoseParams& p,
                          std::vector<int>* labelsOut = nullptr);
}