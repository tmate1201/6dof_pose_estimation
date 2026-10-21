#include "Triangulation.hpp"

namespace pe {

    cv::Point2d project(const cv::Point3d& P, const CameraParams& c) {
        return { c.fx * P.x / P.z + c.cx,
                c.fy * P.y / P.z + c.cy };
    }

    cv::Point3d deproject(double u, double v, double Z, const CameraParams& c) {
        return { (u - c.cx) * Z / c.fx,
                (v - c.cy) * Z / c.fy,
                Z };
    }

    cv::Point3d triangulate(const StereoMatch& m, const CameraParams& c) {
        const double Z = c.fx * c.baseline / m.disparity; 
        return deproject(m.xL, m.y, Z, c);                  
    }
}