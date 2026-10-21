#include "PlaneFit.hpp"

namespace pe {

	Plane planeFrom3Points(const cv::Point3d& P1, const cv::Point3d& P2, const cv::Point3d& P3) {
		Plane plane;
		cv::Point3d u = P2 - P1;
		cv::Point3d v = P3 - P1;
		plane.normal = u.cross(v);
		
		double normLen = cv::norm(plane.normal);
		if (normLen < 1e-6)
			return plane;

		plane.valid = true;
		plane.normal /= normLen;
		plane.d = (-1) * plane.normal.dot(P1);
		
		return plane;
	}

	double pointPlaneDistance(const cv::Point3d& P, const Plane& plane) {
		return plane.d + plane.normal.dot(P);
	}

	Plane fitPlaneLeastSquares(const std::vector<cv::Point3d>& points, const std::vector<int>& indices) {
		Plane pl;
		pl.indices = std::vector<int>(indices);

		if (indices.size() < 3)
			return pl;
				
		
		cv::Point3d centroid(0, 0, 0);

		for (int i = 0; i < indices.size(); i++) {
			centroid += points[indices[i]];
		}
		centroid *= (1.0) / indices.size();
		pl.centroid = centroid;
		cv::Matx33d cov(0, 0, 0, 0, 0, 0, 0, 0, 0);
		for (int idx : indices) {
			cv::Point3d p = points[idx] - centroid;
			cov(0, 0) += p.x * p.x; cov(0, 1) += p.x * p.y; cov(0, 2) += p.x * p.z;
			cov(1, 0) += p.y * p.x; cov(1, 1) += p.y * p.y; cov(1, 2) += p.y * p.z;
			cov(2, 0) += p.z * p.x; cov(2, 1) += p.z * p.y; cov(2, 2) += p.z * p.z;
		}

		cv::Mat W, U, VT;
		cv::SVD::compute(cv::Mat(cov), W, U, VT);

		pl.normal = cv::Point3d(VT.at<double>(2, 0), VT.at<double>(2, 1), VT.at<double>(2, 2));
		pl.normal = pl.normal / cv::norm(pl.normal);
		pl.rms = std::sqrt(W.at<double>(2, 0) / indices.size());

		pl.d = (-1) * pl.normal.dot(centroid);
		pl.valid = true;

		if (pl.normal.dot(centroid) > 0) {
			pl.normal *= -1;
			pl.d *= -1;
		}

		return pl;
	}

	Plane ransacPlane(const std::vector<cv::Point3d>& points, const std::vector<int>& candidateIndices, const PlaneParams& params, const std::vector<cv::Point3d>& existingNormals) {

		if (candidateIndices.size() < 3) return Plane{};

		Plane best_plane;
		int max_inliers = 0;

		std::random_device rd;
		std::mt19937 gen(42); //
		std::uniform_int_distribution<> dis(0, candidateIndices.size() - 1);

		for (int i = 0; i < params.iterations; i++) {
			int idx1 = dis(gen);
			int idx2 = dis(gen);
			int idx3 = dis(gen);

			if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3) continue;

			const auto& p1 = candidateIndices[idx1];
			const auto& p2 = candidateIndices[idx2];
			const auto& p3 = candidateIndices[idx3];

			Plane plane = planeFrom3Points(points[p1], points[p2], points[p3]);

			if (!plane.valid) continue;
			bool skip = false;

			for (int l = 0; l < existingNormals.size(); l++)
				if (std::abs(plane.normal.dot(existingNormals[l])) > params.perpendicularityLimit)
					skip = true;

			if (skip) continue;

			for (int k = 0; k < candidateIndices.size(); ++k) {
				if (std::abs(pointPlaneDistance(points[candidateIndices[k]], plane)) < params.threshold) {
					plane.indices.push_back(candidateIndices[k]);
				}
			}

			if (plane.indices.size() > max_inliers) {
				max_inliers = plane.indices.size();
				best_plane = plane;
			}
		}

		Plane refit = fitPlaneLeastSquares(points, best_plane.indices);

		if (!refit.valid)
			return Plane{};


		std::vector<int> indices;

		for (int k = 0; k < candidateIndices.size(); ++k) {
			if (std::abs(pointPlaneDistance(points[candidateIndices[k]], refit)) < params.threshold) {
				indices.push_back(candidateIndices[k]);
			}
		}

		refit = fitPlaneLeastSquares(points, indices);

		return refit;
	}

	std::vector<Plane> fitFaces(const std::vector<cv::Point3d>& points, const PlaneParams& params) {
		std::vector<int> candidates(points.size());
		std::iota(candidates.begin(), candidates.end(), 0);
		std::vector<Plane> fits;
		std::vector<cv::Point3d> existingNormals;

		for (int i = 0; i < params.maxFaces; i++) {
			Plane plane = ransacPlane(points, candidates, params, existingNormals);

			if (!plane.valid || plane.indices.size() < params.minInIiers)
				break;

			fits.push_back(plane);
			existingNormals.push_back(plane.normal);

			for (int k = 0; k < plane.indices.size(); k++) {
				auto it = std::find(candidates.begin(), candidates.end(), plane.indices[k]);
				if (it != candidates.end()) {
					candidates.erase(it);
				}
			}
		}

		return fits;
	}
}