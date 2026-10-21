#pragma once
#include<opencv2/opencv.hpp>
#include "../config/Config.hpp"
#include "../core/Types.hpp"

namespace pe {
	//   u = fx * X / Z + cx,   v = fy * Y / Z + cy
	cv::Point2d project(const cv::Point3d& P, const CameraParams& c);
	//   X = (u - cx) * Z / fx,   Y = (v - cy) * Z / fy
	cv::Point3d deproject(double u, double v, double Z, const CameraParams& c);

	cv::Point3d triangulate(const StereoMatch& m, const CameraParams& c);
}