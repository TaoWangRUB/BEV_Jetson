// Shared Argus rig plumbing: which physical port is which Argus camera, and what instant
// a frame corresponds to. Header-only, and used by BOTH the capture node and the fused
// VO node — these two facts must not be allowed to drift apart between them.
//
// The full reasoning is in README 4.7. In short:
//   * Argus assigns sensor-ids in /dev/video bind order, which is not port order and is
//     not stable across boots, so the map is resolved at runtime from each node's i2c
//     name — never hard-coded.
//   * A frame's instant is its EXPOSURE MIDPOINT: the metadata SOF is the start of
//     READOUT, which on a global shutter is after the exposure ended, so the exposure
//     spans [SOF - exposure, SOF]. IFrame::getTime() is consumer-side and must not be
//     used for anything comparable across cameras.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <dirent.h>

#include <Argus/Argus.h>
#include <EGLStream/ArgusCaptureMetadata.h>
#include <EGLStream/Frame.h>

namespace bev_camera {

// Physical carrier port -> the i2c name the sensor on it reports. IMX296 sits at
// 0x1a/0x18 where IMX219 sits at 0x10/0x12, so the same port reports a different name
// depending on which module is fitted; match either family.
struct PortEntry { const char* port; const char* imx296; const char* imx219; };
constexpr PortEntry kPortTable[] = {
    {"a", nullptr,  "1-0010"},
    {"b", nullptr,  "1-0012"},
    {"c", "2-001a", "2-0010"},
    {"d", "2-0018", "2-0012"},
    {"e", "7-001a", "7-0010"},
    {"f", "7-0018", "7-0012"},
};

struct CameraNode { int sensor_id; std::string family; std::string i2c; };

// Resolve port -> Argus sensor-id from sysfs. Binding port F before E was observed live
// to give video4=7-0012 (port f) and video5=7-0010 (port e), which shifts every
// hard-coded index by one and silently hands the rig extrinsics to the wrong images.
inline std::map<std::string, CameraNode> scan_video_nodes() {
  std::vector<int> nodes;
  if (DIR* d = opendir("/sys/class/video4linux")) {
    while (dirent* e = readdir(d)) {
      int n = -1;
      if (std::sscanf(e->d_name, "video%d", &n) == 1) nodes.push_back(n);
    }
    closedir(d);
  }
  std::sort(nodes.begin(), nodes.end());  // numeric order == Argus sensor-id order

  std::map<std::string, CameraNode> out;
  for (size_t sid = 0; sid < nodes.size(); ++sid) {
    std::ifstream f("/sys/class/video4linux/video" + std::to_string(nodes[sid]) + "/name");
    std::string name;  // e.g. "vi-output, imx296 2-001a"
    if (!std::getline(f, name)) continue;
    const auto sp = name.rfind(' ');
    if (sp == std::string::npos) continue;
    const std::string i2c = name.substr(sp + 1);
    for (const auto& p : kPortTable) {
      if (p.imx296 && i2c == p.imx296) out[p.port] = {static_cast<int>(sid), "imx296", i2c};
      else if (p.imx219 && i2c == p.imx219) out[p.port] = {static_cast<int>(sid), "imx219", i2c};
    }
  }
  return out;
}

// The IMX296 driver exposes Fast Trigger mode as a module parameter, not a control.
inline bool external_trigger_active() {
  std::ifstream f("/sys/module/imx296/parameters/trigger_mode");
  int v = 0;
  return static_cast<bool>(f >> v) && v == 1;
}

struct FrameTiming {
  uint64_t stamp_ns = 0;      // the instant the frame depicts (exposure midpoint)
  uint64_t sof_ns = 0;        // raw kernel SOF, so the correction stays undoable
  uint64_t exposure_ns = 0;
  uint32_t capture_id = 0;    // session-side: what Argus produced
  uint64_t number = 0;        // consumer-side: what reached us
  bool from_metadata = false; // false = fell back to the consumer-side frame time
};

// `rig_exposure_ns` is latched by the caller and applied to every camera: all four fire
// on the same trigger edge for the same pulse width, so the exposure is identical by
// construction, while Argus reports each camera's own AE state. Pass 0 on the first call
// to latch whatever Argus reports (but prefer the measured trigger pulse width — Argus
// reported 0.521 ms against a real 4.986 ms here, since it does not own the exposure
// under an external trigger).
inline FrameTiming frame_timing(EGLStream::Frame* frame, EGLStream::IFrame* iframe,
                                uint64_t* rig_exposure_ns, bool midpoint = true) {
  FrameTiming ft;
  ft.number = iframe->getNumber();
  uint64_t sof = 0, exposure_ns = 0;
  if (auto* iacm = Argus::interface_cast<EGLStream::IArgusCaptureMetadata>(frame)) {
    if (auto* imeta = Argus::interface_cast<Argus::ICaptureMetadata>(iacm->getMetadata())) {
      sof = imeta->getSensorTimestamp();
      exposure_ns = imeta->getSensorExposureTime();
      ft.capture_id = imeta->getCaptureId();
    }
  }
  if (sof == 0) {                      // no metadata: caller must warn, this is not comparable
    ft.stamp_ns = iframe->getTime();
    return ft;
  }
  if (*rig_exposure_ns == 0) *rig_exposure_ns = exposure_ns;
  ft.from_metadata = true;
  ft.sof_ns = sof;
  ft.exposure_ns = *rig_exposure_ns;
  ft.stamp_ns = midpoint ? sof - ft.exposure_ns / 2 : sof;
  return ft;
}

}  // namespace bev_camera
