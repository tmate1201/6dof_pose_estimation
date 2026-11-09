#include "FaceId.hpp"

namespace pe {

	bool loadModel(const std::string& path, CubeModel& model) {
		cv::FileStorage fs(path, cv::FileStorage::READ);
		if (!fs.isOpened()) return false;

		model.edge = (double)fs["edge_mm"];
		const cv::FileNode faces = fs["faces"];
		for (const cv::FileNode& face : faces) {
			const char faceName = face.name()[0];
			for (const cv::FileNode& sticker : face["stickers"]) {
				const cv::FileNode& centre = sticker["centre_mm"];
				model.centres.emplace_back((double)centre[0], (double)centre[1], (double)centre[2]);
				model.colours.push_back(((std::string)sticker["colour"])[0]);
				model.faceOf.push_back(faceName);
			}
		}
	}

    std::vector<cv::Matx33d> cubeRotations() {
        std::vector<cv::Matx33d> out;

        int perm[3] = { 0, 1, 2 };

        do {
            for (int sx : {1, -1}) {
                for (int sy : {1, -1}) {
                    for (int sz : {1, -1}) {
                        cv::Matx33d G = cv::Matx33d::zeros();
                        double signs[3] = { static_cast<double>(sx),
                                            static_cast<double>(sy),
                                            static_cast<double>(sz) };


                        for (int i = 0; i < 3; ++i) {
                            G(i, perm[i]) = signs[i];
                        }

                        if (cv::determinant(G) > 0.0) {
                            out.push_back(G);
                        }
                    }
                }
            }
        } while (std::next_permutation(perm, perm + 3));

        return out;
    }

    std::vector<std::vector<int>> stickerPermutations(const std::vector<cv::Matx33d>& rots,
        const std::vector<cv::Vec3d>& C) {
        std::vector<std::vector<int>> perm(rots.size(), std::vector<int>(C.size(), -1));
        for (size_t g = 0; g < rots.size(); ++g)
            for (size_t i = 0; i < C.size(); ++i) {
                const cv::Vec3d m = rots[g].t() * C[i];           // where slot i comes from in the model
                for (size_t j = 0; j < C.size(); ++j)
                    if (cv::norm(C[j] - m) < 1.0) { perm[g][i] = static_cast<int>(j); break; }
            }
        return perm;
    }




}




