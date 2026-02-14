#pragma once

#include <optional>

#include "drolib/polynomial/cubic_spline3d.h"
#include "drolib/polynomial/piecewise_polynomial.hpp"
#include "drolib/system/minco_snap_trajectory.hpp"
#include "drolib/type/types.hpp"

namespace drolib {
class ArclenTrajConverter {
public:
  struct Settings {
    size_t num_samples{100};
    size_t arclen_resolution{2000};
    double min_sample_dist{1e-3};
  };

  ArclenTrajConverter(const Settings &settings)
      : num_samples_(settings.num_samples),
        min_sample_dist_(settings.min_sample_dist),
        arclen_resolution_(settings.arclen_resolution) {}

  ~ArclenTrajConverter() = default;

  std::optional<spline::CubicSpline3D>
  convert(const MincoSnapTrajectory &traj) {
    const auto &poly = traj.polys;
    double T = poly.getTotalDuration();

    std::vector<double> s_lut, time_lut;
    build_lookup_table(poly, s_lut, time_lut);

    double total_length = s_lut.back();
    if (total_length < kMinLength)
      return std::nullopt;

    // 2. Uniformly sample s and find corresponding t via LUT interpolation
    std::vector<double> s_samples;
    std::vector<Eigen::Vector3d> points;
    double ds = total_length / num_samples_;

    for (size_t i = 0; i <= num_samples_; ++i) {
      double target_s = i * ds;

      // Find t such that s(t) = target_s using the LUT
      auto it = std::lower_bound(s_lut.begin(), s_lut.end(), target_s);
      size_t idx = std::distance(s_lut.begin(), it);

      double ti;
      if (idx == 0)
        ti = 0.0;
      else if (idx >= s_lut.size())
        ti = T;
      else {
        // Linear interpolation between LUT points
        double s0 = s_lut[idx - 1];
        double s1 = s_lut[idx];
        double t0 = time_lut[idx - 1];
        double t1 = time_lut[idx];
        double slope = (target_s - s0) / (s1 - s0);
        ti = t0 + slope * (t1 - t0);
      }

      s_samples.push_back(target_s);
      points.push_back(poly.getPos(ti));
    }

    spline_.plan(points, s_samples);
    return spline_;
  }

  bool save(const std::string &fname, double step_sz = 2e-2) {
    if (!spline_.isValid()) {
      return false;
    }

    std::ofstream file;
    file.open(fname.c_str());
    file.precision(6);
    file << "arc_length_s\tgamma_x(s)\tgamma_y(s)\tgamma_z(s)\n";

    std::optional<Eigen::Vector2d> spline_domain_opt = spline_.get_domain();
    if (!spline_domain_opt) {
      return false;
    }
    Eigen::Vector2d spline_domain = spline_domain_opt.value();

    size_t num_samples = (spline_domain(1) - spline_domain(0)) / step_sz;
    for (size_t step = 0; step < num_samples; ++step) {
      double s = spline_domain(0) + step * step_sz;
      Eigen::Vector3d pos = spline_.position(s);
      file << s << "\t\t" << pos(0) << "\t\t" << pos(1) << "\t\t" << pos(2)
           << "\n";
    }

    file.precision();
    file.close();

    return true;
  }

private:
  void build_lookup_table(const PiecewisePolynomial<POLY_DEG> &poly,
                          std::vector<double> &s_lut,
                          std::vector<double> &time_lut) {
    double T = poly.getTotalDuration();

    // 1. High-resolution integration to find total length and build LUT
    double dt = T / arclen_resolution_;

    time_lut.reserve(arclen_resolution_);
    time_lut.emplace_back(0);
    s_lut.reserve(arclen_resolution_);
    s_lut.emplace_back(0);

    double accumulated_s = 0.0;
    Eigen::Vector3d prev_pos = poly.getPos(0.0);

    for (size_t i = 1; i <= arclen_resolution_; ++i) {
      double t = i * dt;
      Eigen::Vector3d curr_pos = poly.getPos(t);
      accumulated_s += (curr_pos - prev_pos).norm();

      s_lut.emplace_back(accumulated_s);
      time_lut.emplace_back(t);
      prev_pos = curr_pos;
    }
  }

private:
  static constexpr double kMinLength = 1e-3;
  size_t num_samples_{100};

  size_t arclen_resolution_{2000};
  double min_sample_dist_{1e-3};

  spline::CubicSpline3D spline_;
};

} // namespace drolib
