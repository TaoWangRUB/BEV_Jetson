// How much does the virtual-pinhole remap actually cost, and how does it scale with the
// virtual resolution? The modular VO node spends ~31 ms per set on this against a 33 ms
// budget at 30 Hz (4.5), so the question "should we lower the resolution" needs a number
// rather than an argument.
//
// Standalone on purpose: no cameras, no ROS, no trigger. It builds the same maps the node
// builds (bev_cuvslam/virtual_pinhole.hpp, same calibration files) and remaps a synthetic
// source, so the only thing being timed is the gather.
//
//   bench_remap <calib_dir> [iters]
#include <chrono>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include "bev_cuvslam/virtual_pinhole.hpp"

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: bench_remap <calib_dir> [iters]\n"); return 2; }
  const std::string dir = argv[1];
  const int iters = argc > 2 ? std::atoi(argv[2]) : 30;
  const char* cams[4] = {"cam1", "cam2", "cam3", "cam4"};

  // A textured source: a flat image would be unrealistically cache-friendly for the
  // gather, and a constant one lets the compiler and the prefetcher off too lightly.
  std::vector<cv::Mat> src(4);
  cv::RNG rng(1);
  for (int i = 0; i < 4; ++i) {
    src[i] = cv::Mat(1088, 1456, CV_8UC1);
    rng.fill(src[i], cv::RNG::UNIFORM, 0, 256);
  }

  std::printf("%-12s %-10s %8s %10s %12s\n", "virtual", "focal_px", "Mpix", "ms/set", "ns/pixel");
  for (auto wh : {std::pair<int,int>{768,576}, {640,480}, {512,384}, {384,288}}) {
    const int W = wh.first, H = wh.second;
    std::vector<bev_cuvslam::VirtualPinhole> vp;
    std::vector<int> vsrc;
    for (int i = 0; i < 4; ++i) {
      const auto omni = bev_cuvslam::LoadOmni(dir + "/" + cams[i] + ".yaml");
      for (int k = 0; k < 2; ++k) {
        vp.push_back(bev_cuvslam::BuildVirtualPinhole(omni, (k ? +1 : -1) * CV_PI / 4.0, W, H, 70.0));
        vsrc.push_back(i);
      }
    }
    std::vector<cv::Mat> dst(vp.size());
    for (auto& d : dst) d.create(H, W, CV_8UC1);
    for (size_t k = 0; k < vp.size(); ++k)          // warm the caches and the maps
      cv::remap(src[vsrc[k]], dst[k], vp[k].map1, vp[k].map2, cv::INTER_LINEAR);

    const auto t0 = std::chrono::steady_clock::now();
    for (int n = 0; n < iters; ++n)
      for (size_t k = 0; k < vp.size(); ++k)
        cv::remap(src[vsrc[k]], dst[k], vp[k].map1, vp[k].map2, cv::INTER_LINEAR);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count() / iters;

    const double mpix = vp.size() * double(W) * H / 1e6;
    std::printf("%4dx%-7d %10.1f %8.2f %10.2f %12.1f\n",
                W, H, vp[0].focal, mpix, ms, ms * 1e6 / (mpix * 1e6));
  }
  std::printf("\nthreads %d of %d cpus\n", cv::getNumThreads(), cv::getNumberOfCPUs());
  return 0;
}
