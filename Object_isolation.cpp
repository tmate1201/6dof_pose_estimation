#include "Object_isolation.hpp"

namespace pe {

	std::vector<int> rangeQuery(std::vector<cv::Point3d>& points, int index, double eps) {
		std::vector<int> neighbors;
		for (int i = 0; i < points.size(); i++) {
			if (cv::norm(points[index] - points[i]) < eps)
				neighbors.push_back(i);
		}
		return neighbors;
	}

	std::vector<cv::Point3d> dbScan(const std::vector<pe::StereoMatch>& matches, CameraParams& cam, double eps, int minPts) {
		const int UNASSIGNED = -2, NOISE = -1;
		int clusters = 0;
		std::vector<cv::Point3d> points;

		for (int i = 0; i < matches.size(); i++) {
			points.push_back(pe::triangulate(matches[i], cam));
		}

		std::vector<int> order(points.size());
		std::iota(order.begin(), order.end(), 0);         
		std::sort(order.begin(), order.end(), [&](int a, int b) { return points[a].z < points[b].z; });

		std::vector<bool> visited(points.size(), false);
		std::vector<int> label(points.size(), UNASSIGNED);

		for (int i : order) {	
			if (visited[i]) continue;
			visited[i] = true;

			std::vector<int> neighbors = rangeQuery(points, i, eps);
			if (neighbors.size() < minPts)
			{
				label[i] = NOISE;
				continue;
			}
			
			const int c = clusters++;
			label[i] = c;

			std::vector<int> seeds = neighbors;              // points still to look at
			for (int k = 0; k < seeds.size(); k++) {
				const int j = seeds[k];
				if (label[j] == NOISE) label[j] = c;
				if (visited[j]) continue;
				visited[j] = true;
				label[j] = c;

				std::vector<int> jNeighbors = rangeQuery(points, j, eps);
				if (jNeighbors.size() >= minPts)   // j is a core point too: keep expanding
					seeds.insert(seeds.end(), jNeighbors.begin(), jNeighbors.end());
			}
		}

		std::vector<cv::Point3d> cluster;
		for (size_t i = 0; i < points.size(); i++)
			if (label[i] == 0) cluster.push_back(points[i]);
		return cluster;
	}
}