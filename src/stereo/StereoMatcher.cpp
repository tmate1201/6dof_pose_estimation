#include "StereoMatcher.hpp"

#include <cmath>

namespace pe {

    namespace {
        // returns a (2h+1) x (2h+1) sized patch around the given pixel in the image 
        cv::Mat patchAt(const cv::Mat& img, int x, int y, int h) {
            return img(cv::Rect(x - h, y - h, 2 * h + 1, 2 * h + 1));
        }
    }  // namespace

    // zero mean cross correlation -> ZNCC = sum((f-fmean) * (g-gmean))/sqrt(sum(f-fmean)^2*sum(g-gmean)^2)
    // zero mean ignores brightness offset, the sqrt normalising ignores contrast
    double zncc(const cv::Mat& patchA, const cv::Mat& patchB) {
        
        const int rows = patchA.rows;
        const int cols = patchA.cols;
        const int n = rows * cols;

        // 1. means
        double ma = 0.0, mb = 0.0;
        for (int y = 0; y < rows; ++y) {
            const double* a = patchA.ptr<double>(y); // take a row
            const double* b = patchB.ptr<double>(y);
            for (int x = 0; x < cols; ++x) { // add each score in the row
                ma += a[x];
                mb += b[x]; 
            }
        }
        ma /= n;
        mb /= n;

        // 2. sum (f - f_mean)(g - g_mean) and the two sums of squares
        double num = 0.0, sa = 0.0, sb = 0.0;
        for (int y = 0; y < rows; ++y) {
            const double* a = patchA.ptr<double>(y);
            const double* b = patchB.ptr<double>(y); // take a row
            for (int x = 0; x < cols; ++x) {
                const double fa = a[x] - ma;
                const double fb = b[x] - mb;   // zero mean: ignores brightness offset
                num += fa * fb; // build sum f-fmean g-gmean
                sa += fa * fa;  //f-fmean squared
                sb += fb * fb; // g-gmean squared
            }
        }
        const double den = std::sqrt(sa * sb);                // normalising: ignores contrast
        return den > 1e-9 ? num / den : -1.0;                 // flat patch: no texture to correlate
    }

    RowSearch searchRow(const cv::Mat& src, const cv::Mat& dst, int x, int y, int direction,
                        const StereoParams& p, bool keepCurve) {
        RowSearch r;
        const int W = dst.cols, h = p.half;
        const cv::Mat patch = patchAt(src, x, y, h);

        // ZNCC for every disparity whose candidate patch lies fully inside dst
        std::vector<int> ds;
        std::vector<double> curve;

        for (int d = p.dMin; d <= p.dMax; ++d) {
            const int cx = x - direction * d;
            if (cx - h < 0 || cx + h >= W) continue;
            ds.push_back(d);
            curve.push_back(zncc(patch, patchAt(dst, cx, y, h)));
        }
        if (ds.size() < 3) return r;

        // best peak (first maximum, like np.argmax)
        size_t i = 0;
        for (size_t k = 1; k < curve.size(); ++k)
            if (curve[k] > curve[i])
                i = k;
        r.best = curve[i];

        // second-best value at least peakExclusion px away from the best
        for (size_t k = 0; k < curve.size(); ++k)
            if (std::abs(ds[k] - ds[i]) > p.peakExclusion && curve[k] > r.second)
                r.second = curve[k];

        if (keepCurve) {
            r.d = ds;
            r.curve = curve;
        }

        if (i == 0 || i == curve.size() - 1)
            return r;   // peak on the edge of the range: unreliable

        // sub-pixel: top of the parabola through the peak and its two neighbours
        const double a = curve[i - 1], b = curve[i], c = curve[i + 1];
        const double denom = a - 2.0 * b + c;
        const double offset = denom < 0.0 ? 0.5 * (a - c) / denom : 0.0;

        r.valid = true;
        r.disparity = ds[i] + offset;
        return r;
    }

    std::vector<StereoMatch> matchCorners(const cv::Mat& left, const cv::Mat& right,
                                          const std::vector<Corner>& corners,
                                          const StereoParams& p, MatchStats* stats) {
        std::vector<StereoMatch> out;
        MatchStats s;
        s.corners = static_cast<int>(corners.size());

        for (const Corner& c : corners) {
            // search along the row, take the best peak, test it
            const RowSearch fwd = searchRow(left, right, c.x, c.y, +1, p);
            if (!fwd.valid) { ++s.noPeak; continue; } // invalid search
            if (fwd.best < p.minZncc) { ++s.lowScore; continue; } // the best is lower than the preset min zncc score
            if (fwd.best - fwd.second < p.minUnique) { ++s.ambiguous; continue; } // the 2 best ones are almost the same, skip

            // left-right check: search back from where we landed in the right image
            const int xr = static_cast<int>(std::nearbyint(c.x - fwd.disparity));
            if (xr - p.half < 0 || xr + p.half >= right.cols) { ++s.failedLR; continue; }
            const RowSearch back = searchRow(right, left, xr, c.y, -1, p);
            if (!back.valid || std::abs(back.disparity - fwd.disparity) > p.lrTol) { ++s.failedLR; continue; }

            out.push_back({c.x, c.y, c.x - fwd.disparity, fwd.disparity, fwd.best, fwd.second});
        }
        s.matched = static_cast<int>(out.size());
        if (stats) *stats = s;
        return out;
    }

}  // namespace pe
