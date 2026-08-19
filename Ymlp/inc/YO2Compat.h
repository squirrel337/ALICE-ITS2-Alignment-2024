#ifndef ROOT_YO2Compat
#define ROOT_YO2Compat

// Local stand-ins for the handful of O2 header-only helpers that YImpactParameter
// uses. Everything else in the module reaches O2 only through YDetectorGeometry,
// which has its own cache backend, so this header plus that backend is what lets
// the module build and run without O2.
//
// ---------------------------------------------------------------------------
// READ THIS BEFORE TRUSTING RESULTS
//
// Angle2Alpha and kB2C below are reconstructed from the O2 conventions, not copied
// from the O2 source, because no O2 checkout was available where this was written.
// They are NOT cosmetic: YImpactParameter feeds the selection cut at
// YMultiLayerPerceptron.cxx:4656 (|ip_r| > RANGE_IMPACTPARAMS_R, |ip_z| >
// RANGE_IMPACTPARAMS_Z) and the skip at :5355, so getting either wrong changes
// which tracks enter the cost.
//
// Confirm both against O2 before using this for anything but a smoke test:
//   Angle2Alpha -> o2::math_utils::detail::angle2Alpha  (MathUtils/detail/basicMath.h)
//   kB2C        -> o2::constants::math::B2C             (CommonConstants/MathConstants.h)
// Both are isolated here so a correction is a one-line change.
// ---------------------------------------------------------------------------

#include <cmath>
#include <array>

namespace YO2 {

// o2::constants::math::PIHalf
constexpr double kPIHalf = 1.570796326794896619;

// o2::constants::math::B2C — converts (kG, GeV/c, cm) to a curvature.
constexpr double kB2C = -0.299792458e-3;

// o2::constants::math — the barrel is divided into 18 sectors of 20 degrees.
constexpr double kSectorSpanDeg = 20.0;
constexpr double kSectorSpanRad = kSectorSpanDeg * M_PI / 180.0;

// o2::math_utils::detail::angle2Alpha — quantises an azimuth to the centre of the
// sector containing it.
inline double Angle2Alpha(double phi)
{
   const int sect = int(phi / kSectorSpanRad + (phi < 0 ? -0.0 : 0.0));
   // sector index counted from -pi so that negative phi lands in the right sector
   const int s = (phi >= 0) ? sect : sect - 1;
   return kSectorSpanRad * (double(s) + 0.5);
}

// o2::math_utils::detail::rotateZ, acting in place on a 3-vector
inline void RotateZ(std::array<double, 3>& v, double alpha)
{
   const double s = std::sin(alpha), c = std::cos(alpha);
   const double x = v[0], y = v[1];
   v[0] = x * c - y * s;
   v[1] = x * s + y * c;
}

inline void SinCos(double a, double& s, double& c) { s = std::sin(a); c = std::cos(a); }

// A sink for O2's LOG(level) macro, used once in YImpactParameter. O2 routes
// LOG(debug) to FairLogger, where debug level is suppressed by default, so
// discarding matches the behaviour the module already had.
struct NullStream {
   template <typename T> NullStream& operator<<(const T&) { return *this; }
};

} // namespace YO2

#ifndef LOG
 #define LOG(level) YO2::NullStream()
#endif

#endif
