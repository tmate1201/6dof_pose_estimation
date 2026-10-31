#include "ReportFigures.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../config/Config.hpp"
#include "../features/Harris.hpp"
#include "../features/ShiTomasi.hpp"
#include "../stereo/StereoMatcher.hpp"
#include "../reconstruction/Triangulation.hpp"
#include "../../Object_isolation.hpp"   // dbScan

namespace pe::report {
namespace {

    // ---------------------------------------------------------------- settings
    // Crop used for all figures (left image): x 380..1059, y 0..469
    const cv::Rect kCrop(380, 0, 680, 470);

    // Hand-drawn bounding boxes around the cube, for the "on cube" counts
    const cv::Rect kCubeLeft(690, 190, 296, 261);    // x 690..985, y 190..450
    const cv::Rect kCubeRight(400, 190, 266, 261);   // x 400..665, y 190..450

    // Wall region for the score statistics (left image): x 400..999, y 20..179
    const cv::Rect kWall(400, 20, 600, 160);

    const double kLogMin = -5.0;      // lower end of the log colour scale
    const double kSharedDist = 2.0;   // px, "shared" corners
    const double kNearDist = 12.0;    // px, nearest-neighbour statistic

    // ---------------------------------------------------------------- helpers
    cv::Mat loadGray(const std::string& path) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!img.empty()) img.convertTo(img, CV_64F);
        return img;
    }

    std::vector<double> values(const cv::Mat& m) {
        cv::Mat c = m.isContinuous() ? m : m.clone();
        return std::vector<double>(c.begin<double>(), c.end<double>());
    }

    // Percentile with linear interpolation between the two neighbouring ranks
    // (same definition as NumPy's default).
    double percentile(std::vector<double> v, double p) {
        const double pos = p / 100.0 * (v.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = std::min(lo + 1, v.size() - 1);
        std::nth_element(v.begin(), v.begin() + lo, v.end());
        const double a = v[lo];
        const double b = (hi == lo) ? a : *std::min_element(v.begin() + lo + 1, v.end());
        return a + (pos - lo) * (b - a);
    }

    double maxOf(const cv::Mat& m) {
        double mx = 0.0;
        cv::minMaxLoc(m, nullptr, &mx);
        return mx;
    }

    std::vector<Corner> detect(const cv::Mat& score, CornerParams p, double quality, int maxCorners, int border) {
        p.quality = quality;
        p.maxCorners = maxCorners;
        return pickCorners(score, p, border);
    }

    int countInside(const std::vector<Corner>& cs, const cv::Rect& box) {
        int n = 0;
        for (const Corner& c : cs)   // inclusive box, like the hand-drawn limits
            if (c.x >= box.x && c.x <= box.x + box.width - 1 && c.y >= box.y && c.y <= box.y + box.height - 1) ++n;
        return n;
    }

    // Number of corners in a that have a corner of b within maxDist
    int countShared(const std::vector<Corner>& a, const std::vector<Corner>& b, double maxDist) {
        int n = 0;
        for (const Corner& p : a)
            for (const Corner& q : b) {
                const double dx = p.x - q.x, dy = p.y - q.y;
                if (dx * dx + dy * dy <= maxDist * maxDist) { ++n; break; }
            }
        return n;
    }

    // log10(score / max score), clipped from below at 10^floorExp
    cv::Mat logRelative(const cv::Mat& score, double floorExp) {
        const double mx = maxOf(score), lo = std::pow(10.0, floorExp);
        cv::Mat out(score.size(), CV_64F);
        for (int y = 0; y < score.rows; ++y) {
            const double* s = score.ptr<double>(y);
            double* o = out.ptr<double>(y);
            for (int x = 0; x < score.cols; ++x) o[x] = std::log10(std::max(s[x] / mx, lo));
        }
        return out;
    }

    // ---------------------------------------------------------------- drawing
    cv::Mat toBgr(const cv::Mat& gray) {
        cv::Mat u8, bgr;
        gray.convertTo(u8, CV_8U);
        cv::cvtColor(u8, bgr, cv::COLOR_GRAY2BGR);
        return bgr;
    }

    cv::Mat drawCorners(const cv::Mat& gray, const std::vector<Corner>& cs, const cv::Scalar& colour) {
        cv::Mat img = toBgr(gray);
        for (const Corner& c : cs) {
            cv::circle(img, {c.x, c.y}, 5, {0, 0, 0}, cv::FILLED, cv::LINE_AA);   // dark outline
            cv::circle(img, {c.x, c.y}, 4, colour, cv::FILLED, cv::LINE_AA);
        }
        return img(kCrop).clone();
    }

    // Signed gradient: 0 -> gray 128, +-(99.5th percentile of |g|) -> 255 / 0
    cv::Mat signedImage(const cv::Mat& g) {
        const double v = percentile(values(cv::abs(g)), 99.5);
        cv::Mat out;
        g.convertTo(out, CV_8U, 127.0 / v, 128.0);   // saturates outside [0, 255]
        return out;
    }

    cv::Mat sideBySide(const cv::Mat& a, const cv::Mat& b, int gap) {
        cv::Mat white(a.rows, gap, a.type(), cv::Scalar::all(255)), out;
        cv::hconcat(std::vector<cv::Mat>{a, white, b}, out);
        return out;
    }

    // log score in [kLogMin, 0] -> magma colours
    cv::Mat colourLog(const cv::Mat& logScore) {
        cv::Mat u8, bgr;
        logScore.convertTo(u8, CV_8U, 255.0 / -kLogMin, 255.0);   // kLogMin -> 0, 0 -> 255
        cv::applyColorMap(u8, bgr, cv::COLORMAP_MAGMA);
        return bgr;
    }

    // Vertical colour bar with tick labels and the threshold q = 0.01 (log10 = -2) in cyan
    cv::Mat colourBar(int height) {
        const int barW = 22, labelW = 50, pad = 12;
        cv::Mat out(height, barW + labelW, CV_8UC3, cv::Scalar::all(255));
        cv::Mat ramp(height - 2 * pad, 1, CV_8U);
        for (int i = 0; i < ramp.rows; ++i) ramp.at<uchar>(i) = cv::saturate_cast<uchar>(255.0 * (ramp.rows - 1 - i) / (ramp.rows - 1));
        cv::Mat bar;
        cv::applyColorMap(ramp, bar, cv::COLORMAP_MAGMA);
        cv::resize(bar, bar, {barW, ramp.rows}, 0, 0, cv::INTER_NEAREST);
        bar.copyTo(out(cv::Rect(0, pad, barW, ramp.rows)));
        cv::rectangle(out, cv::Rect(0, pad, barW, ramp.rows), {0, 0, 0}, 1);
        auto yOf = [&](double v) { return pad + static_cast<int>(std::lround((0.0 - v) / -kLogMin * (ramp.rows - 1))); };
        for (int v = 0; v >= static_cast<int>(kLogMin); --v) {
            const int y = yOf(v);
            cv::line(out, {barW, y}, {barW + 5, y}, {0, 0, 0}, 1);
            cv::putText(out, std::to_string(v), {barW + 8, y + 6}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
        }
        const int yT = yOf(-2.0);
        cv::line(out, {0, yT}, {barW, yT}, {255, 255, 0}, 3);   // cyan (BGR)
        return out;
    }

    std::string fmt(double v, int prec = 2) {
        std::ostringstream s;
        s << std::setprecision(prec) << v;
        return s.str();
    }

    // ---------------------------------------------------------------- per image
    struct ImageResult {
        int shi = 0, shiOnCube = 0;
        int harris = 0, harrisOnCube = 0;
        double harrisQ = 0;                   // quality giving the same count
        int harrisN = 0, harrisNOnCube = 0, shared = 0;
    };

    ImageResult analyse(const cv::Mat& gray, const cv::Rect& cube, const CornerParams& p, int border,
                        cv::Mat* shiScore = nullptr, cv::Mat* harrisScore = nullptr,
                        std::vector<Corner>* shiOut = nullptr, std::vector<Corner>* harrisOut = nullptr) {
        const StructureTensor m = structureTensor(sobelGradients(gray), p.window);
        const cv::Mat S = minEigenvalue(m);
        const cv::Mat R = harrisResponse(m, p.harrisK);

        ImageResult r;
        const auto shi = detect(S, p, p.quality, p.maxCorners, border);
        const auto har = detect(R, p, p.quality, p.maxCorners, border);
        r.shi = static_cast<int>(shi.size());       r.shiOnCube = countInside(shi, cube);
        r.harris = static_cast<int>(har.size());    r.harrisOnCube = countInside(har, cube);

        // lower the Harris quality until it finds at least as many corners, keep the strongest r.shi
        double q = p.quality;
        while (static_cast<int>(detect(R, p, q, p.maxCorners, border).size()) < r.shi) q /= 1.5;
        const auto harN = detect(R, p, q, r.shi, border);
        r.harrisQ = q;
        r.harrisN = static_cast<int>(harN.size());  r.harrisNOnCube = countInside(harN, cube);
        r.shared = countShared(shi, harN, kSharedDist);

        if (shiScore) *shiScore = S;
        if (harrisScore) *harrisScore = R;
        if (shiOut) *shiOut = shi;
        if (harrisOut) *harrisOut = har;
        return r;
    }

}  // namespace

int writeFeatureDetectionReport(const std::string& dataDir, const std::string& outDir) {
    const cv::Mat left = loadGray(dataDir + "/left_001.png");
    const cv::Mat right = loadGray(dataDir + "/right_001.png");
    if (left.empty() || right.empty()) {
        std::cerr << "writeFeatureDetectionReport: cannot read images in " << dataDir << "\n";
        return 1;
    }
    std::filesystem::create_directories(outDir);

    const Config cfg{};                                // same parameters as the pipeline
    const CornerParams& p = cfg.corners;
    const int border = cfg.stereo.half + 1;          // same border as Step 1 in main()

    // ---- corners and counts (Table "Corners found by the two scores")
    cv::Mat S, R;
    std::vector<Corner> shiL, harL;
    const ImageResult rl = analyse(left, kCubeLeft, p, border, &S, &R, &shiL, &harL);
    const ImageResult rr = analyse(right, kCubeRight, p, border);

    // ---- figures: corner overlays
    cv::imwrite(outDir + "/rubik_corners_shitomasi.png", drawCorners(left, shiL, {0, 255, 0}));
    cv::imwrite(outDir + "/rubik_corners_harris.png", drawCorners(left, harL, {0, 140, 255}));

    // ---- figure: log-scale scores (Harris R <= 0 ends up at the lowest colour)
    const cv::Mat logS = logRelative(S, kLogMin), logR = logRelative(R, kLogMin);
    const cv::Mat panels = sideBySide(colourLog(logS(kCrop)), colourLog(logR(kCrop)), 12);
    const cv::Mat bar = colourBar(panels.rows);
    cv::Mat logFig;
    cv::hconcat(std::vector<cv::Mat>{panels, cv::Mat(panels.rows, 16, CV_8UC3, cv::Scalar::all(255)), bar}, logFig);
    cv::imwrite(outDir + "/rubik_score_log.png", logFig);

    // ---- figure: gradients
    const Gradients g = sobelGradients(left);
    cv::imwrite(outDir + "/rubik_gradients.png",
                sideBySide(signedImage(g.Ix)(kCrop), signedImage(g.Iy)(kCrop), 12));

    // ---- wall statistics (log10 relative score; floor far below anything that occurs)
    const std::vector<double> wallS = values(logRelative(S, -12.0)(kWall).clone());
    const std::vector<double> wallR = values(logRelative(R, -12.0)(kWall).clone());

    // ---- nearest-neighbour distances of the Shi-Tomasi corners on the cube (left)
    std::vector<Corner> onCube;
    for (const Corner& c : shiL)
        if (countInside({c}, kCubeLeft)) onCube.push_back(c);
    std::vector<double> nn;
    for (size_t i = 0; i < onCube.size(); ++i) {
        double best = 1e18;
        for (size_t j = 0; j < onCube.size(); ++j)
            if (i != j) best = std::min(best, std::hypot(onCube[i].x - onCube[j].x, onCube[i].y - onCube[j].y));
        nn.push_back(best);
    }
    const auto nearCount = std::count_if(nn.begin(), nn.end(), [](double d) { return d < kNearDist; });

    // ---- report values
    std::ostringstream o;
    o << "Feature detection report values (left_001.png / right_001.png)\n"
      << "Parameters: window " << p.window << "x" << p.window << ", q = " << p.quality
      << ", minDist = " << p.minDist << " px, border = " << border << " px, maxCorners = " << p.maxCorners
      << ", Harris k = " << p.harrisK << "\n\n";
    auto row = [&](const char* name, const ImageResult& r) {
        o << name << ":\n"
          << "  Shi-Tomasi, q = " << p.quality << ": " << r.shi << " corners, " << r.shiOnCube << " on cube\n"
          << "  Harris,     q = " << p.quality << ": " << r.harris << " corners, " << r.harrisOnCube << " on cube\n"
          << "  Harris, same count (q = " << fmt(r.harrisQ) << "): " << r.harrisN << " corners, "
          << r.harrisNOnCube << " on cube, shared with Shi-Tomasi (<= " << kSharedDist << " px): " << r.shared << "\n";
    };
    row("Left image", rl);
    row("Right image", rr);
    o << "\nWall, log10(score / max score), left image:\n"
      << "  Shi-Tomasi: median " << fmt(percentile(wallS, 50), 3) << ", 99th percentile " << fmt(percentile(wallS, 99), 3) << "\n"
      << "  Harris:     median " << fmt(percentile(wallR, 50), 3) << ", 99th percentile " << fmt(percentile(wallR, 99), 3) << "\n"
      << "  threshold q = " << p.quality << " -> " << fmt(std::log10(p.quality), 3) << "\n"
      << "\nShi-Tomasi corners on the cube (left): " << onCube.size() << ", nearest other corner < "
      << kNearDist << " px: " << nearCount << " (median distance " << fmt(percentile(nn, 50), 3) << " px)\n";

    std::cout << o.str();
    std::ofstream(outDir + "/report_values.txt") << o.str();
    std::cout << "Report figures written to " << outDir << "/\n";
    return 0;
}


// ======================================================================= Step 2
namespace {

    // Example corners (left image) for the ZNCC curve figure: accepted / ambiguous / low score
    struct Example { int x, y; const char* label; };
    const Example kExamples[] = {
        {872, 384, "accepted"},
        {844, 237, "rejected: ambiguous"},
        {739, 347, "rejected: low ZNCC"},
    };

    // Rows shown in the match figure (both images): y 0..469, full width
    const int kMatchRows = 470;

    // Colour scale of the match lines (disparity in px)
    const double kDispLo = 150.0, kDispHi = 400.0;

    cv::Vec3b jet(double t) {   // t in [0, 1] -> BGR from OpenCV's JET colour map
        cv::Mat px(1, 1, CV_8U, cv::Scalar(cv::saturate_cast<uchar>(255.0 * std::clamp(t, 0.0, 1.0)))), c;
        cv::applyColorMap(px, c, cv::COLORMAP_JET);
        return c.at<cv::Vec3b>(0, 0);
    }

    // Left | right (cropped to kMatchRows), one line per match coloured by disparity
    cv::Mat drawMatchFigure(const cv::Mat& left, const cv::Mat& right, const std::vector<StereoMatch>& ms) {
        cv::Mat both;
        cv::hconcat(left, right, both);
        cv::Mat img = toBgr(both);
        for (const StereoMatch& m : ms) {
            const cv::Vec3b c = jet((m.disparity - kDispLo) / (kDispHi - kDispLo));
            const cv::Point pl(m.xL, m.y), pr(static_cast<int>(std::lround(m.xR)) + left.cols, m.y);
            cv::line(img, pl, pr, cv::Scalar(c[0], c[1], c[2]), 1, cv::LINE_AA);
        }
        for (const StereoMatch& m : ms)   // points on top of the lines
            for (const cv::Point& p : {cv::Point(m.xL, m.y), cv::Point(static_cast<int>(std::lround(m.xR)) + left.cols, m.y)}) {
                cv::circle(img, p, 5, {0, 0, 0}, cv::FILLED, cv::LINE_AA);
                cv::circle(img, p, 4, {0, 255, 0}, cv::FILLED, cv::LINE_AA);
            }
        return img(cv::Rect(0, 0, img.cols, kMatchRows)).clone();
    }

    // Horizontal colour bar for the disparity scale, with labels
    cv::Mat disparityBar(int width) {
        const int h = 18, pad = 160, H = 44;
        cv::Mat out(H, width, CV_8UC3, cv::Scalar::all(255));
        for (int x = pad; x < width - pad; ++x) {
            const cv::Vec3b c = jet(double(x - pad) / (width - 2 * pad - 1));
            cv::line(out, {x, 4}, {x, 4 + h}, cv::Scalar(c[0], c[1], c[2]), 1);
        }
        cv::rectangle(out, cv::Rect(pad, 4, width - 2 * pad, h + 1), {0, 0, 0}, 1);
        for (double d = kDispLo; d <= kDispHi + 1e-9; d += 50.0) {
            const int x = pad + static_cast<int>(std::lround((d - kDispLo) / (kDispHi - kDispLo) * (width - 2 * pad - 1)));
            cv::line(out, {x, 4 + h}, {x, 4 + h + 5}, {0, 0, 0}, 1);
            const std::string t = std::to_string(static_cast<int>(d));
            cv::putText(out, t, {x - 12, H - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
        }
        cv::putText(out, "disparity [px]", {pad - 140, 4 + h}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
        return out;
    }

    // One ZNCC-vs-disparity plot
    cv::Mat curvePlot(const RowSearch& r, const StereoParams& p, const std::string& title) {
        const int W = 520, H = 330, l = 52, rgt = 14, top = 34, bot = 46;
        cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255));
        auto X = [&](double d) { return l + static_cast<int>(std::lround((d - p.dMin) / double(p.dMax - p.dMin) * (W - l - rgt))); };
        auto Y = [&](double z) { return top + static_cast<int>(std::lround((1.0 - z) / 2.0 * (H - top - bot))); };   // z in [-1, 1]

        // grid, axes, ticks
        for (double z = -1.0; z <= 1.0 + 1e-9; z += 0.5) {
            cv::line(img, {l, Y(z)}, {W - rgt, Y(z)}, {225, 225, 225}, 1);
            std::ostringstream t; t << std::fixed << std::setprecision(1) << z;
            cv::putText(img, t.str(), {6, Y(z) + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1, cv::LINE_AA);
        }
        for (int d = 100; d <= p.dMax; d += 100) {
            cv::line(img, {X(d), top}, {X(d), H - bot}, {225, 225, 225}, 1);
            cv::putText(img, std::to_string(d), {X(d) - 14, H - bot + 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1, cv::LINE_AA);
        }
        cv::rectangle(img, cv::Point(l, top), cv::Point(W - rgt, H - bot), {0, 0, 0}, 1);
        cv::putText(img, "disparity [px]", {W / 2 - 50, H - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
        cv::putText(img, title, {l, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 1, cv::LINE_AA);

        // acceptance threshold minZncc (dashed red)
        for (int x = l; x < W - rgt; x += 10)
            cv::line(img, {x, Y(p.minZncc)}, {std::min(x + 5, W - rgt), Y(p.minZncc)}, {40, 40, 200}, 1, cv::LINE_AA);

        // the curve
        std::vector<cv::Point> pts;
        for (size_t k = 0; k < r.d.size(); ++k) pts.push_back({X(r.d[k]), Y(r.curve[k])});
        cv::polylines(img, pts, false, {160, 80, 20}, 1, cv::LINE_AA);

        // best peak and best competing peak (> peakExclusion px away)
        size_t ib = 0;
        for (size_t k = 1; k < r.curve.size(); ++k) if (r.curve[k] > r.curve[ib]) ib = k;
        size_t is = r.curve.size();
        for (size_t k = 0; k < r.curve.size(); ++k)
            if (std::abs(r.d[k] - r.d[ib]) > p.peakExclusion && (is == r.curve.size() || r.curve[k] > r.curve[is])) is = k;
        cv::circle(img, pts[ib], 5, {0, 160, 0}, cv::FILLED, cv::LINE_AA);
        if (is < pts.size()) cv::circle(img, pts[is], 5, {0, 140, 255}, cv::FILLED, cv::LINE_AA);
        return img;
    }

    void range(const std::vector<StereoMatch>& ms, bool (*keep)(const StereoMatch&), double& lo, double& hi, int& n) {
        lo = 1e18; hi = -1e18; n = 0;
        for (const StereoMatch& m : ms)
            if (keep(m)) { lo = std::min(lo, m.disparity); hi = std::max(hi, m.disparity); ++n; }
    }

    bool onCubeLeft(int x, int y) {
        return x >= kCubeLeft.x && x <= kCubeLeft.x + kCubeLeft.width - 1 && y >= kCubeLeft.y && y <= kCubeLeft.y + kCubeLeft.height - 1;
    }

}  // namespace

int writeStereoCorrespondenceReport(const std::string& dataDir, const std::string& outDir) {
    const cv::Mat left = loadGray(dataDir + "/left_001.png");
    const cv::Mat right = loadGray(dataDir + "/right_001.png");
    if (left.empty() || right.empty()) {
        std::cerr << "writeStereoCorrespondenceReport: cannot read images in " << dataDir << "\n";
        return 1;
    }
    std::filesystem::create_directories(outDir);

    const Config cfg{};
    const StereoParams& sp = cfg.stereo;
    const int border = sp.half + 1;

    // ---- Step 1 + Step 2, exactly as in main()
    const std::vector<Corner> corners = detectCorners(left, cfg.corners, border);
    MatchStats st;
    const std::vector<StereoMatch> ms = matchCorners(left, right, corners, sp, &st);

    // ---- figure: matches
    const cv::Mat matchImg = drawMatchFigure(left, right, ms);
    cv::Mat matchFig;
    cv::vconcat(matchImg, disparityBar(matchImg.cols), matchFig);
    cv::imwrite(outDir + "/rubik_matches.png", matchFig);

    // ---- figure: ZNCC curves of the example corners
    std::vector<cv::Mat> plots;
    std::ostringstream ex;
    for (const Example& e : kExamples) {
        const RowSearch r = searchRow(left, right, e.x, e.y, +1, sp, true);
        std::ostringstream title;
        title << "Corner at (" << e.x << ", " << e.y << ") " << e.label;
        plots.push_back(curvePlot(r, sp, title.str()));
        ex << "  (" << e.x << ", " << e.y << ") " << e.label << ": best " << fmt(r.best, 3)
           << " at d = " << fmt(r.disparity, 4) << " px, second " << fmt(r.second, 3)
           << ", margin " << fmt(r.best - r.second, 2) << "\n";
    }
    cv::Mat curves;
    cv::hconcat(plots, curves);
    cv::imwrite(outDir + "/rubik_zncc_curves.png", curves);

    // ---- where the matches and the rejections are
    int matchedOnCube = 0;
    for (const StereoMatch& m : ms) if (onCubeLeft(m.xL, m.y)) ++matchedOnCube;
    double cubeLo, cubeHi, wallLo, wallHi; int nCube, nWall;
    range(ms, [](const StereoMatch& m) { return onCubeLeft(m.xL, m.y); }, cubeLo, cubeHi, nCube);
    range(ms, [](const StereoMatch& m) { return !onCubeLeft(m.xL, m.y); }, wallLo, wallHi, nWall);
    std::vector<double> best;
    for (const StereoMatch& m : ms) best.push_back(m.zncc);

    std::ostringstream o;
    o << "Stereo correspondence report values (left_001.png / right_001.png)\n"
      << "Parameters: patch " << 2 * sp.half + 1 << "x" << 2 * sp.half + 1 << ", disparity " << sp.dMin << ".." << sp.dMax
      << " px, minZncc = " << sp.minZncc << ", minUnique = " << sp.minUnique << ", peakExclusion = " << sp.peakExclusion
      << " px, lrTol = " << sp.lrTol << " px\n\n"
      << "Corners: " << st.corners << "\n"
      << "  rejected, no valid peak:          " << st.noPeak << "\n"
      << "  rejected, ZNCC < minZncc:         " << st.lowScore << "\n"
      << "  rejected, ambiguous:              " << st.ambiguous << "\n"
      << "  rejected, failed left-right check:" << st.failedLR << "\n"
      << "  accepted:                         " << st.matched << " (" << matchedOnCube << " on cube, "
      << st.matched - matchedOnCube << " on the wall)\n\n"
      << "Disparity of accepted matches:\n"
      << "  cube: " << fmt(cubeLo, 4) << " .. " << fmt(cubeHi, 4) << " px (" << nCube << ")\n"
      << "  wall: " << fmt(wallLo, 4) << " .. " << fmt(wallHi, 4) << " px (" << nWall << ")\n"
      << "Best ZNCC of accepted matches: median " << fmt(percentile(best, 50), 3)
      << ", min " << fmt(*std::min_element(best.begin(), best.end()), 3) << "\n\n"
      << "Example corners (ZNCC curve figure):\n" << ex.str();

    // Rejections on the cube, by reason and by region (left face x <= 790, top face y <= 250)
    int low = 0, lowLeft = 0, amb = 0, ambTop = 0;
    for (const Corner& c : corners) {
        if (!onCubeLeft(c.x, c.y)) continue;
        const RowSearch f = searchRow(left, right, c.x, c.y, +1, sp);
        if (!f.valid) continue;
        if (f.best < sp.minZncc) { ++low; if (c.x <= 790) ++lowLeft; }
        else if (f.best - f.second < sp.minUnique) { ++amb; if (c.y <= 250) ++ambTop; }
    }
    o << "\nRejected corners on the cube:\n"
      << "  low ZNCC:  " << low << ", of which x <= 790 px (left face): " << lowLeft << "\n"
      << "  ambiguous: " << amb << ", of which y <= 250 px (top face):  " << ambTop << "\n";

    // Ambiguous rejections whose competing value lies on the flank of the best peak itself
    int flank = 0;
    for (const Corner& c : corners) {
        const RowSearch f = searchRow(left, right, c.x, c.y, +1, sp, true);
        if (!f.valid || f.best < sp.minZncc || f.best - f.second >= sp.minUnique) continue;
        size_t ib = 0, is = 0;
        for (size_t k = 1; k < f.curve.size(); ++k) if (f.curve[k] > f.curve[ib]) ib = k;
        double sv = -2.0;
        for (size_t k = 0; k < f.curve.size(); ++k)
            if (std::abs(f.d[k] - f.d[ib]) > sp.peakExclusion && f.curve[k] > sv) { sv = f.curve[k]; is = k; }
        if (std::abs(f.d[is] - f.d[ib]) <= sp.half) ++flank;
    }
    o << "\nAmbiguous rejections with the competing value within " << sp.half
      << " px of the best peak (flank of the same peak): " << flank << " of " << st.ambiguous << "\n"
      << "Sensitivity to peakExclusion:\n";
    for (int ex : {sp.peakExclusion, sp.half}) {
        StereoParams q = sp;
        q.peakExclusion = ex;
        MatchStats s2;
        matchCorners(left, right, corners, q, &s2);
        o << "  peakExclusion = " << ex << " px: " << s2.matched << " accepted, " << s2.ambiguous
          << " ambiguous, " << s2.failedLR << " failed left-right check\n";
    }

    std::cout << o.str();
    std::ofstream(outDir + "/stereo_values.txt") << o.str();
    std::cout << "Stereo report figures written to " << outDir << "/\n";
    return 0;
}


// ======================================================================= Step 3
namespace {

    // DBSCAN parameters used in main()
    const double kDbEps = 25.0;   // mm
    const int kDbMinPts = 4;

    // Depth colour scale for the 3D figure (mm): near = red, far = blue (JET reversed)
    cv::Scalar depthColour(double z, double zNear, double zFar) {
        const cv::Vec3b c = jet(1.0 - (z - zNear) / (zFar - zNear));
        return cv::Scalar(c[0], c[1], c[2]);
    }

    // Top view (X right, Z down = away from the camera) of the cube points, equal axis scale
    cv::Mat topView(const std::vector<cv::Point3d>& pts, double zNear, double zFar, int H) {
        const int l = 60, r = 20, t = 34, b = 46;
        double x0 = 1e18, x1 = -1e18, z0 = 1e18, z1 = -1e18;
        for (const cv::Point3d& P : pts) { x0 = std::min(x0, P.x); x1 = std::max(x1, P.x); z0 = std::min(z0, P.z); z1 = std::max(z1, P.z); }
        x0 = std::floor(x0 / 10) * 10 - 10; x1 = std::ceil(x1 / 10) * 10 + 10;
        z0 = std::floor(z0 / 10) * 10 - 10; z1 = std::ceil(z1 / 10) * 10 + 10;
        const double scale = (H - t - b) / (z1 - z0);                // px per mm, same for X and Z
        const int W = l + r + static_cast<int>(std::lround((x1 - x0) * scale));
        cv::Mat img(H, W, CV_8UC3, cv::Scalar::all(255));
        auto X = [&](double x) { return l + static_cast<int>(std::lround((x - x0) * scale)); };
        auto Z = [&](double z) { return t + static_cast<int>(std::lround((z - z0) * scale)); };

        for (double x = x0; x <= x1 + 1e-9; x += 20) {
            cv::line(img, {X(x), t}, {X(x), H - b}, {225, 225, 225}, 1);
            cv::putText(img, std::to_string(static_cast<int>(x)), {X(x) - 12, H - b + 18}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1, cv::LINE_AA);
        }
        for (double z = z0; z <= z1 + 1e-9; z += 20) {
            cv::line(img, {l, Z(z)}, {W - r, Z(z)}, {225, 225, 225}, 1);
            cv::putText(img, std::to_string(static_cast<int>(z)), {8, Z(z) + 5}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 0, 0}, 1, cv::LINE_AA);
        }
        cv::rectangle(img, cv::Point(l, t), cv::Point(W - r, H - b), {0, 0, 0}, 1);
        cv::putText(img, "X [mm]", {W / 2 - 20, H - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
        cv::putText(img, "Z [mm]   top view of the cube points", {8, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 0, 0}, 1, cv::LINE_AA);
        for (const cv::Point3d& P : pts) {
            cv::circle(img, {X(P.x), Z(P.z)}, 5, {0, 0, 0}, cv::FILLED, cv::LINE_AA);
            cv::circle(img, {X(P.x), Z(P.z)}, 4, depthColour(P.z, zNear, zFar), cv::FILLED, cv::LINE_AA);
        }
        return img;
    }

}  // namespace

int writeTriangulationReport(const std::string& dataDir, const std::string& outDir) {
    const cv::Mat left = loadGray(dataDir + "/left_001.png");
    const cv::Mat right = loadGray(dataDir + "/right_001.png");
    if (left.empty() || right.empty()) {
        std::cerr << "writeTriangulationReport: cannot read images in " << dataDir << "\n";
        return 1;
    }
    std::filesystem::create_directories(outDir);

    Config cfg{};
    const CameraParams& cam = cfg.camera;
    const std::vector<Corner> corners = detectCorners(left, cfg.corners, cfg.stereo.half + 1);
    const std::vector<StereoMatch> ms = matchCorners(left, right, corners, cfg.stereo);

    // ---- Step 3: triangulation of every match, then DBSCAN as in main()
    std::vector<cv::Point3d> pts;
    for (const StereoMatch& m : ms) pts.push_back(triangulate(m, cam));
    const std::vector<cv::Point3d> cube = dbScan(ms, cfg.camera, kDbEps, kDbMinPts);
    auto inCube = [&](const cv::Point3d& P) {
        for (const cv::Point3d& Q : cube) if (cv::norm(P - Q) < 1e-9) return true;
        return false;
    };

    double zc0 = 1e18, zc1 = -1e18, zw0 = 1e18, zw1 = -1e18;
    int nRemoved = 0;
    for (const cv::Point3d& P : pts) {
        if (inCube(P)) { zc0 = std::min(zc0, P.z); zc1 = std::max(zc1, P.z); }
        else { zw0 = std::min(zw0, P.z); zw1 = std::max(zw1, P.z); ++nRemoved; }
    }
    double extent = 0.0;
    for (size_t i = 0; i < cube.size(); ++i)
        for (size_t j = i + 1; j < cube.size(); ++j) extent = std::max(extent, cv::norm(cube[i] - cube[j]));

    // ---- figure: left image with the points (colour = depth, white = removed) | top view
    cv::Mat img = toBgr(left);
    for (size_t i = 0; i < ms.size(); ++i) {
        const cv::Point p(ms[i].xL, ms[i].y);
        if (inCube(pts[i])) {
            cv::circle(img, p, 6, {0, 0, 0}, cv::FILLED, cv::LINE_AA);
            cv::circle(img, p, 5, depthColour(pts[i].z, zc0, zc1), cv::FILLED, cv::LINE_AA);
        } else {   // removed by DBSCAN: white ring
            cv::circle(img, p, 6, {0, 0, 0}, 3, cv::LINE_AA);
            cv::circle(img, p, 6, {255, 255, 255}, 1, cv::LINE_AA);
        }
    }
    const cv::Mat overlay = img(kCrop).clone();
    const cv::Mat top = topView(cube, zc0, zc1, overlay.rows);
    cv::Mat fig;
    cv::hconcat(std::vector<cv::Mat>{overlay, cv::Mat(overlay.rows, 12, CV_8UC3, cv::Scalar::all(255)), top}, fig);
    cv::imwrite(outDir + "/rubik_points3d.png", fig);

    // ---- depth resolution: dZ/dd = Z^2 / (f B) per pixel of disparity
    auto dzPerPx = [&](double z) { return z * z / (cam.fx * cam.baseline); };
    const double cubeEdge = 57.0;

    std::ostringstream o;
    o << "Triangulation report values (left_001.png / right_001.png)\n"
      << "Camera (ASSUMED, Config.hpp): fx = " << cam.fx << " px, fy = " << cam.fy << " px, cx = " << cam.cx
      << " px, cy = " << cam.cy << " px, baseline = " << cam.baseline << " mm\n"
      << "DBSCAN: eps = " << kDbEps << " mm, minPts = " << kDbMinPts << "\n\n"
      << "Triangulated points: " << pts.size() << "\n"
      << "  kept by DBSCAN (cube): " << cube.size() << ", Z = " << fmt(zc0, 4) << " .. " << fmt(zc1, 4) << " mm\n"
      << "  removed (wall):        " << nRemoved << ", Z = " << fmt(zw0, 4) << " .. " << fmt(zw1, 4) << " mm\n"
      << "Largest distance between two cube points: " << fmt(extent, 3) << " mm"
      << " (cube " << cubeEdge << " mm: face diagonal " << fmt(cubeEdge * std::sqrt(2.0), 3)
      << " mm, space diagonal " << fmt(cubeEdge * std::sqrt(3.0), 3) << " mm)\n"
      << "Depth change per 1 px of disparity, Z^2 / (f B):\n"
      << "  nearest cube point (Z = " << fmt(zc0, 4) << " mm): " << fmt(dzPerPx(zc0), 2) << " mm\n"
      << "  farthest cube point (Z = " << fmt(zc1, 4) << " mm): " << fmt(dzPerPx(zc1), 2) << " mm\n"
      << "  wall (Z = " << fmt(zw1, 4) << " mm): " << fmt(dzPerPx(zw1), 2) << " mm\n";

    std::cout << o.str();
    std::ofstream(outDir + "/triangulation_values.txt") << o.str();
    std::cout << "Triangulation report figures written to " << outDir << "/\n";
    return 0;
}

}  // namespace pe::report
