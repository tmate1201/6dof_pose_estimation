#pragma once
// Synthetic cube scenes with a known answer, to test fitFaces().
#include "PlaneFit.hpp"

namespace pe {

// Runs 4 scenes (2 faces, 2 faces + outliers, 3 faces, noise only) and prints
// found faces, inliers, RMS, normal error vs. the truth and face-to-face angles.
void runPlaneFitTest(const PlaneParams& params);

}  // namespace pe
