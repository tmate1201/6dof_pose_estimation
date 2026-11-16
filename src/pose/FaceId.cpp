#include "FaceId.hpp"
#include "CubePose.hpp"   // col()

namespace pe {

    // Loads the rubik_model.json, which contains the current configuration of the rubics cube
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
        return model.centres.size() == 54;
	}

    // Calculates all the possible 24 rotations of the cube with positive determinant, returning them as a vector
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

    std::vector<std::vector<int>> stickerPermutations(const std::vector<cv::Matx33d>& rots, const std::vector<cv::Vec3d>& C) {

        std::vector<std::vector<int>> perm(rots.size(), std::vector<int>(C.size(), -1));
        for (size_t g = 0; g < rots.size(); ++g)
            for (size_t i = 0; i < C.size(); ++i) {
                const cv::Vec3d m = rots[g].t() * C[i];           // where slot i comes from in the model
                for (size_t j = 0; j < C.size(); ++j)
                    if (cv::norm(C[j] - m) < 1.0) {
                        perm[g][i] = static_cast<int>(j);
                        break;
                    }
            }
        return perm;
    }
    // Visible slots of the fitted cube.
    std::vector<Slot> visibleSlots(const std::vector<cv::Vec3d>& C, const CubePose& pose, double edge, double minCos, std::vector<int>& index) {
        std::vector<Slot> out;
        index.clear();
        for (size_t i = 0; i < C.size(); ++i) {
            int axis = 0;                                        // the coordinate equal to +-edge/2
            for (int k = 0; k < 3; ++k)
                if (std::abs(std::abs(C[i][k]) - edge / 2.0) < 1e-6) axis = k;
            cv::Vec3d n(0, 0, 0);
            n[axis] = C[i][axis] > 0 ? 1.0 : -1.0;               // outward normal, cube coordinates

            const cv::Vec3d p = pose.A * C[i] + pose.t;          // slot centre, camera frame
            const cv::Vec3d nCam = pose.A * n;                   // its normal, camera frame
            const cv::Vec3d toCam = -p / cv::norm(p);            // direction back to the camera
            if (nCam.dot(toCam) > minCos) {
                out.push_back({ p, 2 * axis + (n[axis] > 0 ? 1 : 0) });
                index.push_back(static_cast<int>(i));
            }
        }
        return out;
    }

    // Brightness of each visible slot: median grey value in a square window around its projection.
    std::vector<double> sampleBrightness(const cv::Mat& image, const std::vector<Slot>& slots, const CameraParams& cam, double windowFrac, double pitch) {
        std::vector<double> b(slots.size(), std::numeric_limits<double>::quiet_NaN());
        std::vector<uchar> values;

        cv::Mat gray = image;                                             // the pixels are read as 8-bit below
        if (image.depth() != CV_8U) image.convertTo(gray, CV_8U);

        for (size_t v = 0; v < slots.size(); ++v) {
            const cv::Vec3d& p = slots[v].p;
            if (p[2] <= 0) continue;                                      // behind the camera

            // 1) project the slot centre into the image
            const cv::Point2d uv = project(cv::Point3d(p[0], p[1], p[2]), cam);
            const int u = static_cast<int>(std::lround(uv.x));
            const int w = static_cast<int>(std::lround(uv.y));

            // 2) window radius: a fraction of one sticker's size in pixels (pitch * f / Z)
            const double stickerPx = pitch * cam.fx / p[2];
            const int r = std::max(2, static_cast<int>(std::lround(windowFrac * stickerPx)));

            // 3) collect the pixels of the (2r+1)x(2r+1) window, clipped to the image
            const int u0 = std::max(u - r, 0), u1 = std::min(u + r, gray.cols - 1);
            const int v0 = std::max(w - r, 0), v1 = std::min(w + r, gray.rows - 1);
            if (u0 > u1 || v0 > v1) continue;                             // completely outside

            values.clear();
            for (int y = v0; y <= v1; ++y) {
                const uchar* row = gray.ptr<uchar>(y);
                for (int x = u0; x <= u1; ++x) values.push_back(row[x]);
            }

            // 4) median: partially sort so the middle element is in place
            const size_t mid = values.size() / 2;
            std::nth_element(values.begin(), values.begin() + mid, values.end());
            b[v] = values[mid];
        }
        return b;
    }

    // ---------------------------------------------------------------- Step 6: scoring

    int irClass(char colour) {
        switch (colour) {
            case 'W': case 'Y': return 2;   // bright in IR
            case 'R': case 'O': return 1;   // medium
            default:            return 0;   // 'G', 'B': dark
        }
    }

    // z = (b - mean_face) / std_face, face by face. Each face is lit differently, so only
    // "brighter or darker than the rest of its own face" can be compared.
    std::vector<double> zScorePerFace(const std::vector<double>& b, const std::vector<Slot>& slots) {
        std::vector<double> z(b.size(), std::numeric_limits<double>::quiet_NaN());
        for (int f = 0; f < 6; ++f) {
            double sum = 0.0, sq = 0.0;
            int n = 0;
            for (size_t v = 0; v < b.size(); ++v)
                if (slots[v].face == f && !std::isnan(b[v])) { sum += b[v]; ++n; }
            if (n == 0) continue;
            const double mean = sum / n;
            for (size_t v = 0; v < b.size(); ++v)
                if (slots[v].face == f && !std::isnan(b[v])) sq += (b[v] - mean) * (b[v] - mean);
            const double spread = std::sqrt(sq / n) + 1e-6;                // +1e-6: a perfectly flat face gives z = 0
            for (size_t v = 0; v < b.size(); ++v)
                if (slots[v].face == f && !std::isnan(b[v])) z[v] = (b[v] - mean) / spread;
        }
        return z;
    }

    // Pearson correlation: score = sum(z * e~) / sqrt(sum(z^2) * sum(e~^2)),
    // e~ = predicted IR class minus the mean predicted class of its face.
    double scoreHypothesis(const std::vector<double>& z, const std::vector<Slot>& slots, const std::vector<int>& index,
                           const std::vector<int>& permG, const std::vector<char>& colours) {
        // predicted class of every visible slot under this rotation
        std::vector<double> e(z.size(), 0.0);
        for (size_t v = 0; v < z.size(); ++v)
            e[v] = irClass(colours[permG[index[v]]]);

        // centre the prediction face by face (only over the slots that have a measurement)
        for (int f = 0; f < 6; ++f) {
            double sum = 0.0;
            int n = 0;
            for (size_t v = 0; v < z.size(); ++v)
                if (slots[v].face == f && !std::isnan(z[v])) { sum += e[v]; ++n; }
            if (n == 0) continue;
            for (size_t v = 0; v < z.size(); ++v)
                if (slots[v].face == f) e[v] -= sum / n;
        }

        // correlation over all measured slots
        double zz = 0.0, ee = 0.0, ze = 0.0;
        for (size_t v = 0; v < z.size(); ++v) {
            if (std::isnan(z[v])) continue;
            ze += z[v] * e[v];
            zz += z[v] * z[v];
            ee += e[v] * e[v];
        }
        if (zz <= 0.0 || ee <= 1e-12) return -1.0;                        // no pattern to compare
        return ze / std::sqrt(zz * ee);
    }

    // ---------------------------------------------------------------- Steps 7 + 8

    Orientation identifyFaces(const cv::Mat& gray, const CubePose& pose, int nFaces, const CubeModel& model,
                              const std::vector<cv::Matx33d>& rots, const std::vector<std::vector<int>>& perm,
                              const CameraParams& cam, const FaceIdParams& p,
                              std::vector<double>* scoresOut, int* bestOut) {
        Orientation out;
        if (!pose.valid || rots.empty()) return out;

        cv::Mat gray8 = gray;                                             // convert once, not for every shift
        if (gray.depth() != CV_8U) gray.convertTo(gray8, CV_8U);

        // Step 8: cube centres to try, in order along the unseen axis. With 2 faces, column 2 of A
        // has no seen face, so the centre is shifted along it. With 3 faces: only the step-6 centre.
        std::vector<cv::Vec3d> centres{ pose.t };
        if (nFaces == 2) {
            centres.clear();
            const cv::Vec3d a = col(pose.A, 2);
            for (double d = -p.shiftRange; d <= p.shiftRange + 1e-9; d += 1.0)
                centres.push_back(pose.t + d * a);
        }

        // Steps 4-6 for every centre
        struct Trial { std::vector<double> scores; int g = -1; int measured = 0; };
        std::vector<Trial> trials(centres.size());
        int iBest = -1;
        for (size_t i = 0; i < centres.size(); ++i) {
            CubePose shifted = pose;
            shifted.t = centres[i];

            // Steps 4 + 5: visible slots and their brightness (the same for all 24 rotations)
            std::vector<int> index;
            const std::vector<Slot> slots = visibleSlots(model.centres, shifted, model.edge, p.minCos, index);
            const std::vector<double> b = sampleBrightness(gray8, slots, cam, p.windowFrac);
            const std::vector<double> z = zScorePerFace(b, slots);
            Trial& tr = trials[i];
            for (double x : z) if (!std::isnan(x)) ++tr.measured;
            if (tr.measured < 6) continue;                                // too few stickers to decide, g stays -1

            // Step 6: score every rotation
            tr.scores.resize(rots.size());
            for (size_t g = 0; g < rots.size(); ++g)
                tr.scores[g] = scoreHypothesis(z, slots, index, perm[g], model.colours);
            tr.g = static_cast<int>(std::max_element(tr.scores.begin(), tr.scores.end()) - tr.scores.begin());
            if (iBest < 0 || tr.scores[tr.g] > trials[iBest].scores[trials[iBest].g]) iBest = static_cast<int>(i);
        }
        if (iBest < 0) return out;

        // Step 8: the score is flat while the sampling windows stay inside the stickers, so the
        // maximum is only known to within a few mm. Take the middle of that plateau: the neighbouring
        // centres with the same rotation and a score within plateauTol of the best.
        const int g = trials[iBest].g;
        const double top = trials[iBest].scores[g];
        auto onPlateau = [&](int i) {
            return i >= 0 && i < static_cast<int>(trials.size()) && trials[i].g == g && trials[i].scores[g] >= top - p.plateauTol;
        };
        int lo = iBest, hi = iBest;
        while (onPlateau(lo - 1)) --lo;
        while (onPlateau(hi + 1)) ++hi;
        const int iMid = (lo + hi) / 2;
        const Trial& tr = trials[iMid];

        // Step 7: the chosen rotation and how clearly it wins
        double second = -1.0;
        for (size_t h = 0; h < tr.scores.size(); ++h)
            if (static_cast<int>(h) != g) second = std::max(second, tr.scores[h]);

        out.R = pose.A * rots[g];                                         // model -> fitted box -> camera
        out.t = centres[iMid];
        out.best = tr.scores[g];
        out.second = second;
        out.visible = tr.measured;
        out.valid = out.best - second >= p.minMargin;                     // a near tie means the pattern can't decide
        if (scoresOut) *scoresOut = tr.scores;
        if (bestOut) *bestOut = g;
        return out;
    }

}




