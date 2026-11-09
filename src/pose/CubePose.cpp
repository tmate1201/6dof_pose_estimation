#include "CubePose.hpp"

#include <cmath>
#include <utility>

namespace pe {
	
    cv::Matx33d skew(const cv::Vec3d& w) {
        return cv::Matx33d(
            0.0, -w[2], w[1],
            w[2], 0.0, -w[0],
            -w[1], w[0], 0.0
        );
    }

    cv::Matx33d rodrigues(const cv::Vec3d& w) {
        double theta2 = w.dot(w);
        double theta = std::sqrt(theta2);

        cv::Matx33d I = cv::Matx33d::eye();
        cv::Matx33d Wx = skew(w);

        double a, b;
        if (theta < 1e-6) {
            a = 1.0 - theta2 / 6.0;
            b = 0.5 - theta2 / 24.0;
        }
        else {
            a = std::sin(theta) / theta;
            b = (1.0 - std::cos(theta)) / theta2;
        }
        // R = I + a * Wx + b * Wx^2
        return I + a * Wx + b * (Wx * Wx);
    }

    cv::Matx33d nearestRotation(const cv::Matx33d& N, bool dirty) {
        
        if (!dirty)
        {
            cv::Mat M = cv::Mat(N);

            cv::SVD svd(M, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

            cv::Mat R = svd.u * svd.vt;
            double det = cv::determinant(R);
            if (det < 0) {

                cv::Matx33d U_corrected = svd.u;
                U_corrected(0, 2) *= -1;
                U_corrected(1, 2) *= -1;
                U_corrected(2, 2) *= -1;
                R = U_corrected * svd.vt;
            }
            cv::Matx33d R_out;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    R_out(i,j) = R.at<double>(i, j);
                }
            }
            return R;
        }
        else {
            cv::Vec3d n1(N(0 ,0), N(1, 0), N(2, 0));
            cv::Vec3d n2(N(0, 1), N(1, 1), N(2, 1));

            n1 = n1 / cv::norm(n1);
            n2 = n2 - n2.dot(n1) * n1;
            n2 = n2 / cv::norm(n2);

            cv::Vec3d n3 = n1.cross(n2);
            cv::Matx33d R(n1(0), n2(0), n3(0), n1(1), n2(1), n3(1), n1(2), n2(2), n3(2));
            return R;
        }
    }

    bool solve6(cv::Matx66d H, cv::Vec6d g, cv::Vec6d& x) {

        // solves H * x = g; refineLM passes -g as g, so no sign change here
        cv::Vec6d b = g;

        const double EPS = 1e-12;

        // 1. Forward Elimination with Partial Pivoting
        for (int c = 0; c < 6; ++c) {
            int pivot_row = c;
            double max_val = std::abs(H(c, c));
            for (int r = c + 1; r < 6; ++r) {
                double val = std::abs(H(r, c));
                if (val > max_val) {
                    max_val = val;
                    pivot_row = r;
                }
            }

            if (pivot_row != c) {
                for (int j = 0; j < 6; ++j) {
                    std::swap(H(c, j), H(pivot_row, j));
                }
                std::swap(b[c], b[pivot_row]);
            }

            if (std::abs(H(c, c)) < EPS) {
                return false;
            }

            for (int r = c + 1; r < 6; ++r) {
                double factor = H(r, c) / H(c, c);
                H(r, c) = 0.0;
                for (int j = c + 1; j < 6; ++j) {
                    H(r, j) -= factor * H(c, j);
                }
                b[r] -= factor * b[c];
            }
        }

        // 2. Back Substitution
        for (int r = 5; r >= 0; --r) {
            double sum = 0.0;
            for (int j = r + 1; j < 6; ++j) {
                sum += H(r, j) * x[j];
            }
            x[r] = (b[r] - sum) / H(r,r);
        }

        return true;
    }

    std::vector<int> faceLabels(const std::vector<cv::Point3d>& pts, std::vector<Plane>& faces) {

        if (faces.size() == 3 && faces[0].normal.dot(faces[1].normal.cross(faces[2].normal)) < 0)
            std::swap(faces[0], faces[1]);

        std::vector<int> labels(pts.size(), -1);

        for (size_t k = 0; k < faces.size(); ++k)
            for (int i : faces[k].indices)
                labels[i] = static_cast<int>(k);

        return labels;
    }

    CubePose initialPose(const std::vector<cv::Point3d>& pts, const std::vector<int>& label, const std::vector<Plane>& faces, double edge) {
        CubePose pose;
        if (faces.size() < 2) return pose;   

        // Get rotation
        const cv::Vec3d n0 = vec(faces[0].normal);
        const cv::Vec3d n1 = vec(faces[1].normal);
        const cv::Vec3d n2 = faces.size() == 3 ? vec(faces[2].normal) : n0.cross(n1);

        const cv::Matx33d N(n0[0], n1[0], n2[0],
                            n0[1], n1[1], n2[1],
                            n0[2], n1[2], n2[2]);
        pose.A = nearestRotation(N);

        // Center
        //  a_k . t = a_k . c_k - edge/2   ->   t = sum_k a_k (a_k . c_k - edge/2)
        pose.t = cv::Vec3d(0, 0, 0);
        for (size_t k = 0; k < faces.size(); ++k) {
            const cv::Vec3d a = col(pose.A, static_cast<int>(k));
            pose.t += a * (a.dot(vec(faces[k].centroid)) - edge / 2.0);
        }

        // the third axis has no face, so slide an edge-long window along it
        // and put the centre where the window holds the most points
        if (faces.size() == 2) {

            const cv::Vec3d a = col(pose.A, 2);
            std::vector<double> x;

            for (size_t i = 0; i < pts.size(); ++i)
                if (label[i] >= 0)
                    x.push_back(a.dot(vec(pts[i])));

            std::sort(x.begin(), x.end());

            int best = 0;
            double sum = 0.0;
            int nBest = 0;

            for (size_t s = 0; s < x.size(); ++s) {

                const auto end = std::upper_bound(x.begin(), x.end(), x[s] + edge);
                const int count = static_cast<int>(end - (x.begin() + s));   // points in [x_s, x_s + edge]

                if (count > best){ 
                    best = count;
                    sum = x[s];
                    nBest = 1;
                }
                else if (count == best) { 
                    sum += x[s];
                    ++nBest;
                }
            }
            if (nBest > 0) 
                pose.t += a * (sum / nBest + edge / 2.0);   // window start + half an edge
        }

        pose.points = 0;

        for (int l : label) 
            if (l >= 0) 
                ++pose.points;

        pose.valid = true;
        return pose;
    }

    // r_i = a_k . (p_i - t) - edge/2  for a point on face k: its signed distance from the
    // model face plane (mm). 0 = exactly on the face, > 0 outside the cube, < 0 inside.
    // Points with label -1 get NaN.

    std::vector<double> residuals(const std::vector<cv::Point3d>& pts, const std::vector<int>& label, const CubePose& pose, double edge) {
        std::vector<double> r(pts.size(), std::numeric_limits<double>::quiet_NaN());

        for (size_t i = 0; i < pts.size(); ++i) {
            const int k = label[i];

            if (k < 0) 
                continue;     // on no face

            const cv::Vec3d a(pose.A(0, k), pose.A(1, k), pose.A(2, k));   // axis of face k
            const cv::Vec3d d(pts[i].x - pose.t[0], pts[i].y - pose.t[1], pts[i].z - pose.t[2]); // vector from the cube centre to the point
            r[i] = a.dot(d) - edge / 2.0; // distance from the plane
        }
        return r;
    }
    // Levenberg-Marquardt on the 6 pose parameters (rotation vector w, shift dt).
    // Residual of a point on face k: r = a_k . (p - t) - edge/2.
    // Jacobian row: [ (a_k x (p - t))^T , -a_k^T ].  Update: A <- exp([w]x) A,  t <- t + dt.
    void refineLM(const std::vector<cv::Point3d>& pts, const std::vector<int>& label, CubePose& pose, const PoseParams& p) {
        if (!pose.valid)
            return;
        bool thirdFace = false;

        for (int l : label)
            if (l == 2) { thirdFace = true; break; }

        std::vector<double> r = residuals(pts, label, pose, p.edge);
        double lambda = p.lambda0;

        for (int it = 0; it < p.lmIterations; ++it) {
            // ---- 1. Huber weights from the current residuals: 1 if |r| < delta, else delta/|r|
            std::vector<double> w(pts.size(), 0.0);
            for (size_t i = 0; i < pts.size(); ++i)
                if (label[i] >= 0)
                    w[i] = std::abs(r[i]) < p.huber ? 1.0 : p.huber / std::abs(r[i]);

            // ---- 2. normal equations: H = sum w J^T J,  g = sum w J^T r,  cost = sum w r^2
            cv::Matx66d H = cv::Matx66d::zeros();
            cv::Vec6d g(0, 0, 0, 0, 0, 0);
            double cost = 0.0;
            for (size_t i = 0; i < pts.size(); ++i) {
                const int k = label[i];
                if (k < 0) continue;
                const cv::Vec3d a(pose.A(0, k), pose.A(1, k), pose.A(2, k));
                const cv::Vec3d d(pts[i].x - pose.t[0], pts[i].y - pose.t[1], pts[i].z - pose.t[2]);
                const cv::Vec3d jw = a.cross(d);                                // d r / d w
                const cv::Vec6d J(jw[0], jw[1], jw[2], -a[0], -a[1], -a[2]);  // d r / d(w, dt)
                for (int row = 0; row < 6; ++row) {
                    g[row] += w[i] * J[row] * r[i];
                    for (int c = 0; c < 6; ++c) H(row, c) += w[i] * J[row] * J[c];
                }
                cost += w[i] * r[i] * r[i];
            }

            // ---- 3. damped step; grow lambda until the step lowers the cost
            bool accepted = false;
            cv::Vec6d dx(0, 0, 0, 0, 0, 0);
            while (lambda <= 1e6) {
                cv::Matx66d Hd = H;
                for (int j = 0; j < 6; ++j) Hd(j, j) += lambda * (H(j, j) + 1e-9);   // H + lambda diag(H)
                if (solve6(Hd, -g, dx)) {
                    if (!thirdFace) {                                   // NEW: no step along the hidden axis
                        const cv::Vec3d h = col(pose.A, 2);
                        const double s = dx[3] * h[0] + dx[4] * h[1] + dx[5] * h[2];
                        dx[3] -= s * h[0];
                        dx[4] -= s * h[1];
                        dx[5] -= s * h[2];
                    }
                    CubePose cand = pose;
                    cand.A = rodrigues(cv::Vec3d(dx[0], dx[1], dx[2])) * pose.A;     // rotate the axes
                    cand.t = pose.t + cv::Vec3d(dx[3], dx[4], dx[5]);                // shift the centre
                    const std::vector<double> r2 = residuals(pts, label, cand, p.edge);
                    double cost2 = 0.0;
                    for (size_t i = 0; i < pts.size(); ++i)
                        if (label[i] >= 0) cost2 += w[i] * r2[i] * r2[i];          // same weights!
                    if (cost2 < cost) {                                             // better: accept
                        pose = cand;
                        r = r2;
                        lambda *= 0.3;
                        accepted = true;
                        break;
                    }
                }
                lambda *= 10.0;                                                     // worse: damp more
            }

            // ---- 4. stop when no step helps any more, or the step became negligible
            if (!accepted) break;
            double step = 0.0;
            for (int j = 0; j < 6; ++j) step += dx[j] * dx[j];
            if (std::sqrt(step) < 1e-8) break;
        }
    }

    // This function basically reassings each points according to the new orientation. So the first big plane that RANSAC found gets split
    // and its points are reassigned to a plane where it fits the most, within tolerance.
    std::vector<int> reassign(const std::vector<cv::Point3d>& pts, const CubePose& pose,
        int nFaces, const PoseParams& p) {
        const double half = p.edge / 2.0, tol = p.reassignTol;
        std::vector<int> label(pts.size(), -1);
        for (size_t i = 0; i < pts.size(); ++i) {
            // cube coordinates X = A^T (p - t): position along each cube axis, centre = origin
            const cv::Vec3d d(pts[i].x - pose.t[0], pts[i].y - pose.t[1], pts[i].z - pose.t[2]);
            const cv::Vec3d X = pose.A.t() * d;

            double best = tol;                                    // must be closer than tol
            for (int k = 0; k < nFaces; ++k) {
                const double dist = std::abs(X[k] - half);        // distance from face k's plane
                bool inside = true;                               // inside the face square?
                for (int j = 0; j < 3; ++j)
                    if (j != k && std::abs(X[j]) > half + tol) inside = false;
                if (inside && dist < best) { best = dist; label[i] = k; }
            }
        }
        return label;
    }

    // Step 6: labels -> initial pose -> LM -> (reassign -> LM) x reassignRounds -> quality
    CubePose estimatePose(const std::vector<cv::Point3d>& pts, std::vector<Plane> faces, const PoseParams& p, std::vector<int>* labelsOut) {

        std::vector<int> label = faceLabels(pts, faces);              // also fixes the face order
        CubePose pose = initialPose(pts, label, faces, p.edge);
        if (!pose.valid)
            return pose;                                               // fewer than 2 faces

        refineLM(pts, label, pose, p);
        const int nFaces = static_cast<int>(faces.size());
        for (int round = 0; round < p.reassignRounds; ++round) {      // ICP-like: assign, then fit
            label = reassign(pts, pose, nFaces, p);
            refineLM(pts, label, pose, p);
        }

        // quality: RMS of the residuals below inlierTol, and how many points those are
        const std::vector<double> r = residuals(pts, label, pose, p.edge);
        double sum = 0.0;
        pose.points = 0;
        pose.inliers = 0;
        for (size_t i = 0; i < pts.size(); ++i) {
            if (label[i] < 0) continue;
            ++pose.points;
            if (std::abs(r[i]) < p.inlierTol) {
                sum += r[i] * r[i];
                ++pose.inliers;
            }
        }
        pose.rms = pose.inliers > 0 ? std::sqrt(sum / pose.inliers) : 0.0;

        if (labelsOut) *labelsOut = label;
        return pose;
    }
}
