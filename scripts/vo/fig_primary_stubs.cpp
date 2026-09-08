// Link stubs for code paths the FIG test never calls (BuildCameraGraph needs a real rig;
// we hand the graph in directly).
#include "common/log.h"
#include "camera/camera.h"
namespace cuvslam { namespace Trace {
Verbosity GetVerbosity() { return Verbosity::Error; }
void PrintDecoratedMsg(const char* const, const char* const, ...) {}
}}
namespace cuvslam { namespace camera {
static Vector2T g_res(0,0);
const Vector2T& ICameraModel::getResolution() const { return g_res; }
bool ICameraModel::normalizePoint(const Vector2T&, Vector2T&) const { return false; }
bool ICameraModel::denormalizePoint(const Vector2T&, Vector2T&) const { return false; }
}}
