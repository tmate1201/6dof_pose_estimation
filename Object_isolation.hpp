#pragma once
#include <opencv2/opencv.hpp>
#include <numeric>
#include "src/core/Types.hpp"
#include "src/reconstruction/Triangulation.hpp"
#include "src/config/Config.hpp"

namespace pe {
	std::vector<cv::Point3d> dbScan(const std::vector<pe::StereoMatch>& matches, CameraParams& cam, double eps, int minPts);
}
