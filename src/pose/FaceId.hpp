#pragma once
#include <opencv2/opencv.hpp>
#include "../core/Types.hpp"
#include "../config/Config.hpp"
#include "../reconstruction/Triangulation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace pe {

	struct CubeModel {
		double edge = 0;
		std::vector<cv::Vec3d> centres; //centre coords of sticker in the model frame
		std::vector<char> colours; //white, yellow, red, orange, green, blue
		std::vector<char> faceOf; //up, right, front, down, left, back
	};
	struct Slot {
		cv::Vec3d p;     // centre in the camera frame (mm)
		int face;        // 2*axis + (sign > 0): which face of the fitted cube
	};

	bool loadModel(const std::string& path, CubeModel& model);
	std::vector<cv::Matx33d> cubeRotations();
	std::vector<std::vector<int>> stickerPermutations(const std::vector<cv::Matx33d>& rots, const std::vector<cv::Vec3d>& C);
	std::vector<Slot> visibleSlots(const std::vector<cv::Vec3d>& C, const CubePose& pose, double edge, double minCos, std::vector<int>& index);
	// b[v] belongs to slots[v]; NaN if the slot projects outside the image.
	std::vector<double> sampleBrightness(const cv::Mat& gray, const std::vector<Slot>& slots, const CameraParams& cam, double windowFrac, double pitch = 19.0);

	// Step 6: IR brightness class of a sticker colour: W,Y -> 2 (bright), R,O -> 1, G,B -> 0 (dark)
	int irClass(char colour);
	// Step 6a: brightness standardised face by face, z = (b - mean of its face) / std of its face.
	// Depends only on the image, not on the rotation, so it is computed once. NaN samples stay NaN.
	std::vector<double> zScorePerFace(const std::vector<double>& b, const std::vector<Slot>& slots);
	// Step 6b: score of one rotation (one row of the perm table): Pearson correlation between z and
	// the predicted IR classes, centred face by face. In [-1, 1], 1 = the pattern matches perfectly.
	double scoreHypothesis(const std::vector<double>& z, const std::vector<Slot>& slots, const std::vector<int>& index,
	                       const std::vector<int>& permG, const std::vector<char>& colours);

	// Steps 7 + 8: try all 24 rotations, keep the best one, R = A * G.
	// nFaces: faces seen in step 6. With 2 faces the centre along the unseen axis (column 2 of A) is
	// uncertain, so the cube is also shifted along it (-shiftRange..+shiftRange, 1 mm steps) and the
	// middle of the range of shifts with the highest score is kept (step 8). With 3 faces there is no shift.
	// valid = the best rotation beats the second best by at least p.minMargin.
	// scoresOut (optional): the 24 scores at the chosen shift, in the order of rots.
	// bestOut (optional): index of the chosen rotation in rots.
	Orientation identifyFaces(const cv::Mat& gray, const CubePose& pose, int nFaces, const CubeModel& model,
	                          const std::vector<cv::Matx33d>& rots, const std::vector<std::vector<int>>& perm,
	                          const CameraParams& cam, const FaceIdParams& p,
	                          std::vector<double>* scoresOut = nullptr, int* bestOut = nullptr);


}