#pragma once
#include <opencv2/opencv.hpp>

namespace pe {

	struct CubeModel {
		double edge = 0;
		std::vector<cv::Vec3d> centres; //centre coords of sticker in the model frame
		std::vector<char> colours; //white, yellow, red, orange, green, blue
		std::vector<char> faceOf; //up, right, front, down, left, back
	};

	bool loadModel(const std::string& path, CubeModel& model);
	std::vector<cv::Matx33d> cubeRotations();
	std::vector<std::vector<int>> stickerPermutations(const std::vector<cv::Matx33d>& rots, const std::vector<cv::Vec3d>& C);



}