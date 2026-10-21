#pragma once
#include <opencv2/opencv.hpp>
#include <random>
#include <numeric>
#include "src/core/Types.hpp"
#include "src/config/Config.hpp"

namespace pe {

	Plane planeFrom3Points(const cv::Point3d& P1, const cv::Point3d& P2, const cv::Point3d& P3);
	double pointPlaneDistance(const cv::Point3d& P, const Plane& plane);
	Plane fitPlaneLeastSquares(const std::vector<cv::Point3d>& points, const std::vector<int>& indices);
	Plane ransacPlane(const std::vector<cv::Point3d>& points, const std::vector<int>& candidateIndices, const PlaneParams& params, const std::vector<cv::Vec3d>& existingNormals);
	std::vector<Plane> fitFaces(const std::vector<cv::Point3d>& points, const PlaneParams& params);
}