# project_pose_estimation (C++)

Stereo-vision 6-DoF pose estimation of a CubeSat model.

## Layout

```
project_pose_estimation.cpp   main: load images, run the steps, show and save results
src/
  core/        Types.hpp       data passed between steps (Corner, StereoMatch, MatchStats)
  config/      Config.hpp      every tunable parameter, one struct per step
  features/    Filters         filter2d (Sobel, box sum), maxFilter
               ShiTomasi       Step 1: gradients, structure tensor, min eigenvalue, pickCorners
  stereo/      StereoMatcher   Step 2: ZNCC row search, uniqueness, sub-pixel, left-right check
data/          input images (cube_L.png, cube_R.png)
results/       created at run time
```

## What OpenCV is used for (and what not)

- **Used:** `cv::Mat` as the image container (like a NumPy array), element-wise
  maths (`+`, `*`, `.mul()`, `cv::max`, `cv::abs`), padding (`copyMakeBorder`,
  like `np.pad`), `imread` / `imwrite` / `imshow`, `cvtColor`, drawing
  (`line`, `circle`, `applyColorMap`).
- **Not used:** the algorithms themselves. `filter2d`, Sobel gradients, the
  structure tensor, the eigenvalue, non-maximum suppression, ZNCC, the
  sub-pixel fit and the left-right check are all written by hand in `src/`.
  (Not `cv::filter2D`, `cv::Sobel`, `cv::dilate`, `cv::cornerMinEigenVal`,
  `cv::goodFeaturesToTrack`, `cv::matchTemplate`, `cv::StereoBM`.)

## Conventions

- Images are single channel `CV_64F` (double). Access: `img.at<double>(y, x)`
  or row pointers `img.ptr<double>(y)[x]` in hot loops.
- Each step: one module = `.hpp` (what it does) + `.cpp` (how), one params
  struct in `Config.hpp`, one output type in `Types.hpp`, one block in `main`.
- Everything is in `namespace pe`.

## Running

Results are saved to `/results`.
