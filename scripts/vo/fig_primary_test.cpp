// Feed cuVSLAM's own FrustumIntersectionGraph the exact topology of the BEV rig and print
// which cameras survive as primary. No images, no GPU, no rig maths - just the library's
// own camera-selection logic, which is where the dead virtual camera is decided.
#include <cstdio>
#include <vector>
#include "camera/frustum_intersection_graph.h"

using namespace cuvslam;
using namespace cuvslam::camera;

static const char* NAMES[8] = {"cam1-45","cam1+45","cam2-45","cam2+45",
                               "cam3-45","cam3+45","cam4-45","cam4+45"};

// argv[1] = "1" if cuvslam2.cpp currently sets enable_depth_stereo_tracking_fig = true in the
// Multicamera branch (i.e. our patch is applied). The gate then tests the configuration the
// library will ACTUALLY build, not a hardcoded one.
int main(int argc, char** argv) {
  const bool patched = (argc > 1 && argv[1][0] == '1');
  // The BEV rig: 8 virtual pinholes forming 4 DISJOINT stereo pairs.
  // Measured overlaps from scripts/vo/verify_rig_build.sh.
  const int pairs[4][2] = {{0,3},{1,4},{2,7},{5,6}};
  const float fir[4]    = {0.936f, 0.934f, 0.916f, 0.945f};

  std::vector<FrustumIntersectionGraph::CameraGraphNode> graph(8);
  for (int p = 0; p < 4; ++p) {
    const int a = pairs[p][0], b = pairs[p][1];
    graph[a].stereo_camera_pairs.emplace_back(b, fir[p]); graph[a].degree_of_vertex++;
    graph[b].stereo_camera_pairs.emplace_back(a, fir[p]); graph[b].degree_of_vertex++;
  }

  auto report = [&](const char* label, const FigSettings& s) {
    FrustumIntersectionGraph fig(graph, s);
    printf("\n%s\n  valid=%d  primary cameras:", label, (int)fig.is_valid());
    std::vector<bool> is_prim(8,false);
    for (CameraId c : fig.primary_cameras()) is_prim[c] = true;
    for (int c = 0; c < 8; ++c) if (is_prim[c]) printf(" %d", c);
    printf("\n");
    for (int c = 0; c < 8; ++c)
      if (!is_prim[c]) printf("  ** camera %d (%s) IS NOT PRIMARY -> no mono SOF is created for it\n",
                              c, NAMES[c]);
  };

  // Exactly what cuvslam2.cpp does for OdometryMode::Multicamera:
  //   depth_ids = {0}, allow_stereo_track_for_depth = false   (cuvslam2.cpp:511-513)
  FigSettings as_shipped;
  as_shipped.mode = MulticameraMode::Precision;
  as_shipped.depth_ids = {0};
  as_shipped.allow_stereo_track_for_depth = false;
  report("AS SHIPPED  (Precision, depth_ids={0}, allow_stereo_track_for_depth=false)", as_shipped);

  FigSettings no_depth;
  no_depth.mode = MulticameraMode::Precision;
  no_depth.depth_ids = {};
  report("WITHOUT the depth_ids={0} line", no_depth);

  FigSettings allow;
  allow.mode = MulticameraMode::Precision;
  allow.depth_ids = {0};
  allow.allow_stereo_track_for_depth = true;
  report("WITH allow_stereo_track_for_depth=true", allow);
  // GATE on the configuration this cuVSLAM tree will actually build.
  FigSettings effective = as_shipped;
  effective.allow_stereo_track_for_depth = patched;
  printf("\ngating on the CURRENT cuvslam2.cpp (%s)\n",
         patched ? "patched: enable_depth_stereo_tracking_fig = true"
                 : "UNPATCHED: enable_depth_stereo_tracking_fig = false");
  FrustumIntersectionGraph fig(graph, effective);
  std::vector<bool> prim(8,false);
  for (CameraId c : fig.primary_cameras()) prim[c] = true;
  int dropped = 0;
  for (int c = 0; c < 8; ++c) if (!prim[c]) dropped++;
  if (dropped) {
    printf("\nFAIL - %d of 8 virtual cameras are not primary and will never be tracked.\n"
           "       Fix: cuvslam2.cpp Multicamera branch must set\n"
           "       enable_depth_stereo_tracking_fig = true (see patch/cuvslam/).\n", dropped);
    return 1;
  }
  printf("\nPASS - all 8 virtual cameras are primary.\n");
  return 0;
}
