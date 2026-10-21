#include "Filters.hpp"

namespace pe {

    cv::Mat sobelX() {
        double data[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        return cv::Mat(3, 3, CV_64F, data).clone();
    }
    cv::Mat sobelY() { return sobelX().t(); }

    cv::Mat filter2d(const cv::Mat& img, const cv::Mat& kernel) {
        CV_Assert(img.type() == CV_64F && kernel.type() == CV_64F);
        const int kh = kernel.rows, kw = kernel.cols;
        const int ry = kh / 2, rx = kw / 2;
        const int H = img.rows, W = img.cols;

        // extend the image by mirroring the border (np.pad(img, ..., mode="reflect"))
        cv::Mat pad;
        cv::copyMakeBorder(img, pad, ry, ry, rx, rx, cv::BORDER_REFLECT_101);

        // same trick as in Python: loop over the few kernel cells, and add a shifted
        // copy of the whole image each time (out += k[i, j] * pad[i:i+H, j:j+W])
        cv::Mat out = cv::Mat::zeros(H, W, CV_64F);
        for (int i = 0; i < kh; ++i)
            for (int j = 0; j < kw; ++j)
                out += kernel.at<double>(i, j) * pad(cv::Rect(j, i, W, H));
        return out;
    }

    cv::Mat maxFilter(const cv::Mat& img, int size) {
        const int r = size / 2, H = img.rows, W = img.cols;
        cv::Mat pad;
        cv::copyMakeBorder(img, pad, r, r, r, r, cv::BORDER_REPLICATE);

        cv::Mat out = pad(cv::Rect(0, 0, W, H)).clone();
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                out = cv::max(out, pad(cv::Rect(j, i, W, H)));   // element-wise maximum
        return out;
    }

}  // namespace pe
