#include "Harris.hpp"

namespace pe {

cv::Mat harrisResponse(const StructureTensor& m, double k) {
    cv::Mat out(m.Mxx.size(), CV_64F);
    for (int y = 0; y < out.rows; ++y) {
        const double* a = m.Mxx.ptr<double>(y);   // pointer to row y
        const double* c = m.Myy.ptr<double>(y);
        const double* b = m.Mxy.ptr<double>(y);
        double* o = out.ptr<double>(y);
        for (int x = 0; x < out.cols; ++x) {
            const double det = a[x] * c[x] - b[x] * b[x];
            const double tr = a[x] + c[x];
            o[x] = det - k * tr * tr;
        }
    }
    return out;
}

std::vector<Corner> detectCornersHarris(const cv::Mat& gray, const CornerParams& p, int border, cv::Mat* scoreOut) {
    const Gradients g = sobelGradients(gray);
    const StructureTensor m = structureTensor(g, p.window);
    cv::Mat score = harrisResponse(m, p.harrisK);
    std::vector<Corner> corners = pickCorners(score, p, border);   // threshold = quality * max R
    if (scoreOut) *scoreOut = score;
    return corners;
}

}  // namespace pe
