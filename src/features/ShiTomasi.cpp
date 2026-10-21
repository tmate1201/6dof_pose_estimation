#include "ShiTomasi.hpp"

#include <algorithm>
#include <cmath>
#include "Filters.hpp"

namespace pe {

Gradients sobelGradients(const cv::Mat& gray) {
    return {filter2d(gray, sobelX()), filter2d(gray, sobelY())};
}

StructureTensor structureTensor(const Gradients& g, int window) {
    const cv::Mat ones = cv::Mat::ones(window, window, CV_64F);   // box window -> window sum
    return {filter2d(g.Ix.mul(g.Ix), ones),     // .mul = element-wise product (NumPy Ix * Ix)
            filter2d(g.Iy.mul(g.Iy), ones),
            filter2d(g.Ix.mul(g.Iy), ones)};
}

cv::Mat minEigenvalue(const StructureTensor& m) {
    cv::Mat out(m.Mxx.size(), CV_64F);
    for (int y = 0; y < out.rows; ++y) {
        const double* a = m.Mxx.ptr<double>(y);   // pointer to row y
        const double* c = m.Myy.ptr<double>(y);
        const double* b = m.Mxy.ptr<double>(y);
        double* o = out.ptr<double>(y);
        for (int x = 0; x < out.cols; ++x) {
            // lambda_min = (tr - sqrt(tr^2 - 4 det)) / 2
            const double tr = a[x] + c[x];
            const double disc = tr * tr - 4.0 * (a[x] * c[x] - b[x] * b[x]);
            // rounding can make disc slightly negative on flat areas -> score 0
            // (the Python version gets NaN there and nan_to_num turns it into 0)
            o[x] = disc < 0.0 ? 0.0 : (tr - std::sqrt(disc)) / 2.0;
        }
    }
    return out;
}

std::vector<Corner> pickCorners(const cv::Mat& score, const CornerParams& p, int border) {
    double maxScore = 0.0;
    cv::minMaxLoc(score, nullptr, &maxScore);           // like score.max()
    const double threshold = p.quality * maxScore;
    const cv::Mat localMax = maxFilter(score, 3);

    // 1. candidates: strong enough, a 3x3 local maximum, not near the border
    std::vector<Corner> cand;
    for (int y = border; y < score.rows - border; ++y) {
        const double* s = score.ptr<double>(y);
        const double* lm = localMax.ptr<double>(y);
        for (int x = border; x < score.cols - border; ++x)
            if (s[x] > threshold && s[x] == lm[x]) cand.push_back({x, y, s[x]});
    }

    // 2. strongest first (stable: equal scores keep row-by-row order)
    std::stable_sort(cand.begin(), cand.end(),
                     [](const Corner& a, const Corner& b) { return a.score > b.score; });

    // 3. greedy minimum distance: a corner too close to a stronger kept one is dropped
    std::vector<Corner> kept;
    const double minD2 = static_cast<double>(p.minDist) * p.minDist;
    for (const Corner& c : cand) {
        bool farEnough = true;
        for (const Corner& k : kept) {
            const double dx = c.x - k.x, dy = c.y - k.y;
            if (dx * dx + dy * dy < minD2) { farEnough = false; break; }
        }
        if (farEnough) {
            kept.push_back(c);
            if (static_cast<int>(kept.size()) == p.maxCorners) break;
        }
    }
    return kept;
}

std::vector<Corner> detectCorners(const cv::Mat& gray, const CornerParams& p, int border, cv::Mat* scoreOut) {
    const Gradients g = sobelGradients(gray);
    const StructureTensor m = structureTensor(g, p.window);
    cv::Mat score = minEigenvalue(m);
    std::vector<Corner> corners = pickCorners(score, p, border);
    if (scoreOut) *scoreOut = score;
    return corners;
}

}  // namespace pe
