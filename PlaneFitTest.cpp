// PlaneFitTest.cpp - synthetic cube scene to test fitFaces().
//
// Builds a 57 mm cube with a KNOWN pose in front of the camera, puts a 4x4 grid of
// points (like the grid-line corners of a Rubik's cube) on every face the camera can
// see, adds measurement noise and a few random outliers, runs pe::fitFaces and
// compares the result with the truth.
//
// Usage: call pe::runPlaneFitTest(params) from main (temporarily), or add a "--test" switch.
// Uses from your code: pe::fitFaces, pe::Plane {normal, indices, rms}, pe::PlaneParams.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <opencv2/core.hpp>
#include "PlaneFitTest.hpp"

namespace pe {

struct TestScene {
    std::vector<cv::Point3d> points;      // what fitFaces gets
    std::vector<int> trueFace;            // per point: index into trueNormals, -1 = outlier
    std::vector<cv::Point3d> trueNormals; // outward normals of the visible faces
};

// Rotation about the y axis (yaw) then about the x axis (pitch), angles in degrees
static cv::Matx33d rotation(double yawDeg, double pitchDeg) {
    const double a = yawDeg * CV_PI / 180.0, b = pitchDeg * CV_PI / 180.0;
    const cv::Matx33d Ry(std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a));
    const cv::Matx33d Rx(1, 0, 0, 0, std::cos(b), -std::sin(b), 0, std::sin(b), std::cos(b));
    return Ry * Rx;
}

// center: cube centre in the left-camera frame (mm). edge: cube size (mm).
// noise: standard deviation of the added noise (mm). outliers: number of random points.
TestScene makeCubeScene(cv::Point3d center, double yawDeg, double pitchDeg, double edge,
                        double noise, int outliers, unsigned seed) {
    TestScene s;
    std::mt19937 gen(seed);
    std::normal_distribution<double> n01(0.0, noise);
    const cv::Matx33d R = rotation(yawDeg, pitchDeg);
    const double h = edge / 2.0;

    // the 6 faces in the cube's own frame: outward normal + two in-face directions
    const cv::Vec3d axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int a = 0; a < 3; ++a) {
        for (int sign : {+1, -1}) {
            const cv::Vec3d nCube = sign * axes[a];
            const cv::Vec3d u = axes[(a + 1) % 3], v = axes[(a + 2) % 3];
            const cv::Vec3d nCam = R * nCube;                        // normal in camera frame
            const cv::Vec3d faceCenter = cv::Vec3d(center) + nCam * h;
            // visible only if it faces the camera, and not almost edge-on (> 75 deg):
            // such faces are hardly seen and never matched in real images
            const double cosView = -nCam.dot(faceCenter) / cv::norm(faceCenter);
            if (cosView < std::cos(75.0 * CV_PI / 180.0)) continue;

            const int faceId = static_cast<int>(s.trueNormals.size());
            s.trueNormals.emplace_back(nCam[0], nCam[1], nCam[2]);
            // 4 x 4 grid: -h, -h/3, +h/3, +h  (grid-line corners of a 3x3 face)
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    const double su = -h + i * edge / 3.0, sv = -h + j * edge / 3.0;
                    const cv::Vec3d pCube = nCube * h + u * su + v * sv;
                    const cv::Vec3d p = R * pCube + cv::Vec3d(center);
                    s.points.emplace_back(p[0] + n01(gen), p[1] + n01(gen), p[2] + n01(gen));
                    s.trueFace.push_back(faceId);
                }
        }
    }
    // outliers: random points in a box around the cube (like wrong matches)
    std::uniform_real_distribution<double> box(-1.2 * edge, 1.2 * edge);
    for (int k = 0; k < outliers; ++k) {
        s.points.emplace_back(center.x + box(gen), center.y + box(gen), center.z + box(gen));
        s.trueFace.push_back(-1);
    }
    return s;
}

static double angleDeg(const cv::Point3d& a, const cv::Point3d& b) {
    const double c = std::abs(a.dot(b)) / (cv::norm(a) * cv::norm(b));
    return std::acos(std::min(1.0, c)) * 180.0 / CV_PI;
}

static void runOne(const char* name, const TestScene& s, const PlaneParams& params) {
    std::printf("\n=== %s: %zu points, %zu visible faces ===\n", name, s.points.size(), s.trueNormals.size());
    const std::vector<Plane> faces = fitFaces(s.points, params);
    std::printf("found %zu faces\n", faces.size());

    for (size_t f = 0; f < faces.size(); ++f) {
        const Plane& p = faces[f];
        // which true face is it? the one with the most parallel normal
        int best = 0;
        for (size_t t = 1; t < s.trueNormals.size(); ++t)
            if (angleDeg(p.normal, s.trueNormals[t]) < angleDeg(p.normal, s.trueNormals[best])) best = (int)t;
        // how many of its inliers really belong to that face / are outliers
        int correct = 0, outl = 0;
        for (int idx : p.indices) {
            if (s.trueFace[idx] == best) ++correct;
            if (s.trueFace[idx] == -1) ++outl;
        }
        const bool facesCamera = p.normal.dot(s.trueNormals[best]) > 0;
        std::printf("face %zu: %2zu inliers (%2d from true face %d, %d outliers), rms %.2f mm, "
                    "normal error %.2f deg, oriented %s\n",
                    f, p.indices.size(), correct, best, outl, p.rms,
                    angleDeg(p.normal, s.trueNormals[best]), facesCamera ? "OK" : "WRONG");
    }
    for (size_t a = 0; a < faces.size(); ++a)
        for (size_t b = a + 1; b < faces.size(); ++b)
            std::printf("angle face %zu - face %zu: %.2f deg (true 90)\n", a, b,
                        angleDeg(faces[a].normal, faces[b].normal));
}

void runPlaneFitTest(const PlaneParams& params) {
    // 1. easy: two faces, little noise, no outliers
    runOne("two faces, clean", makeCubeScene({30, 20, 230}, 35, 0, 57, 0.3, 0, 1), params);
    // 2. like your real data: two faces, 0.5 mm noise, 6 wrong matches
    runOne("two faces + outliers", makeCubeScene({30, 20, 230}, 35, 0, 57, 0.5, 6, 2), params);
    // 3. three faces visible (cube tilted towards the camera)
    runOne("three faces", makeCubeScene({20, 60, 230}, 40, 35, 57, 0.5, 6, 3), params);

    // 4. no cube at all, only scattered points: the right answer is 0 faces
    TestScene noise;
    std::mt19937 gen(4);
    std::uniform_real_distribution<double> box(-60.0, 60.0);
    for (int k = 0; k < 40; ++k) {
        noise.points.emplace_back(box(gen), box(gen), 230 + box(gen));
        noise.trueFace.push_back(-1);
    }
    noise.trueNormals.emplace_back(0, 0, -1);   // dummy, only used for the printout
    runOne("only noise (expect 0 faces)", noise, params);
}

}  // namespace pe
