## 1. The viewer

- [x] 1.1 **`scripts/vo/rerun_multicam.py` — replay a recorded log into one Rerun timeline.**
  Reads the odometry + observations bag (`datasets/replay_out/obs_*`) and the raw image log
  (`datasets/imglog_*_bag`), and logs: the 8 virtual pinholes with their tracked features, the 4 raw
  fisheyes, the landmark map, and the trajectory. Reuses the verified Mei projection from
  `rerun_virtual_pinholes.py` and the bag readers from `rerun_odometry.py` — no second implementation.

- [x] 1.2 **Entity naming — fixed a silent view collision.** Virtual pinholes own `rig/cam0..cam7`;
  logging the physical cameras at `rig/cam1..cam4` collided and Rerun let the later log win. Symptom:
  "cam1 raw" and "cam4 raw" appeared identical — they were really vpin[1] and vpin[4], two carves
  1.63° apart. Physical cameras now live at `rig/raw_camN`. `VCAMS` order is cuVSLAM's own vcam index
  (referenced by each observation's `z` field); the blueprint is reordered for display, the array is not.

- [x] 1.3 **Fisheye domes** so a ~192° lens is viewable in 3D at all. `rr.Pinhole` cannot express a
  field beyond 180°, so each fisheye is textured onto a spherical cap mesh (11 180 triangles/camera,
  `--dome-radius`, `--tex-scale`, `--dome-stride`). Rerun's `Mesh3D` albedo must be **RGB** — a
  greyscale texture is rejected with "Bad albedo texture shape".

- [x] 1.4 **Inverted mounting handled at display only** (`--upright`), since the calibration is solved
  on the raw inverted frames and nothing downstream un-rolls them.

- [x] 1.5 **Viewer documented in `scripts/README.md`** — the four `vo/rerun_*` / `render_*` rows
  plus a "Running the Rerun viewer" section: the venv (rerun-sdk is not installable into the
  system python here, PEP 668 — `python3 -m venv --system-site-packages .venv` keeps rosbags,
  cv2 and numpy from the system), the two-bag invocation, and the `--serve` port convention
  (`--port` alone still binds 9090 for the UI, so concurrent servers need `--port` *and*
  `--web-viewer-port`).

  Two things had to be fixed before the documented invocation actually ran:

  - **`--t-range START:END`** on `rerun_multicam.py`. `--frames` subsampled the *whole* run, so
    on a 57 s log the default 180 frames is one pose in five — enough to miss the 0.75 s
    tracking freeze in `retarget-vo-to-imx296-rig` 5.0g completely. `--frames` now subsamples
    the selected window instead.
  - **`find_bag()` no longer trips over our own bag naming.** rosbags picks ROS 1 vs ROS 2 from
    the SUFFIX (`any(x.suffix != '.bag' ...)` in `highlevel/anyreader.py`), and README §3.3 tells
    people to write ROS 2 bags as `raw_log_to_bag.py -o /tmp/run1.bag` — a *directory* named
    `.bag`. Every viewer therefore got the rosbag1 reader and died with "Could not open file
    ...: Is a directory". It now hands back a suffix-free symlink; nothing is renamed.

  `replay_host.sh` gained `OBS=1` to publish and record `/cuvslam/landmarks` +
  `/cuvslam/observations` (naming the output `obs_*`), which is what the viewer needs and what
  the default rate-measuring runs must NOT carry.

- [x] 1.6 **Pane order walks the ring.** The two rows now run in opposite carve order —
  row 1 `cam1 +45, cam1 -45, cam2 +45, cam2 -45`, row 2 `cam3 -45, cam3 +45, cam4 -45, cam4 +45`
  — so reading each row left-to-right continues around the rig instead of jumping back across it
  at the row break. Display order only; `VCAMS` is cuVSLAM's own indexing and is what each
  observation's vcam field refers to, so the array is untouched.

- [ ] 1.7 **Loop closure: cuVSLAM has it, this node does not use it.** The viewer cannot show
  loop closures because nothing produces any — `cuvslam_multicam_node` builds a
  `cuvslam::Odometry`, which is pure VO with no pose graph. The library also ships
  `cuvslam::Slam` (`cuvslam2.h:885`), constructed as `Slam(rig, primary_cameras, config)` — so it
  takes a multicamera rig — with `throttling_time_ms` (minimum interval between loop-closure
  events), `max_map_size` (pose-graph cap, 300 for real-time), `enable_reading_internals` for the
  pose graph and loop closures, a `lc_status` field, a `LoopClosure` landmark layer, and
  `GetAllSlamPoses()`.

  This is the structural fix for the drift in 5.1/5.2, not just a visualisation: pure VO has
  nothing to pin the trajectory. Order of work:
  - [x] 1.7a **Done, and it closes loops on run1.** `cuvslam::Slam` is built behind
        `enable_slam` over all 8 virtual pinholes as primary cameras (matching
        `MulticameraMode::Precision`), consuming `Odometry::State` — it augments the tracker,
        it does not replace it. `/cuvslam/slam_odometry` carries the corrected pose;
        `/cuvslam/odometry` stays pure VO so the §5 figures remain comparable.

        **Every number this entry has carried was wrong, in three different ways.** They are
        listed rather than deleted because each was published here as fact:
        1. "3 loop closures, correction to 1.68 m" — from accumulating `/cuvslam/slam_odometry`
           per frame (1.7f). The count was `lc_status` rising edges, not events.
        2. "42 closures, optimised 22.26 m vs VO 97.78 m, the pose graph absorbs the teleport" —
           the two covered different intervals (1.7h), and the optimised path never reached the
           teleport at all.
        3. Everything measured with SLAM on at 1.0x — SLAM starves the tracker (1.7j), so those
           trajectories were degraded before loop closure entered into it.

        **Corrected, from the optimised trajectory:**

        | | poses | path | max step | steps > 0.5 m |
        |---|---|---|---|---|
        | optimised `slam_path` | 720 | **22.26 m** | **1.37 m** | 2 |
        | pure VO | 933 | 97.78 m | **44.76 m** | 11 |

        **The honest verdict: this log cannot answer whether loop closure helps.** From the
        one matched-rate run (`obs_slam_v6`, 0.4x, 1153 of 1155 sets, both trajectories from the
        SAME run):

        | | t < 43 s | whole run |
        |---|---|---|
        | pure VO | 859 poses, 22.21 m, max step 0.39 m | 35.52 m, max 50.22 m |
        | optimised SLAM | 859 poses, 22.95 m, max step 0.39 m | 36.12 m, max 50.22 m |

        Three independent reasons the question stays open:
        - **No real loops.** A ground-truth-free test — each loop edge joins two poses cuVSLAM
          matched as the same place, so measure their separation in each trajectory — gives
          **0.579 m in SLAM against 0.647 m in VO, a 1.12x improvement**. And those 63 revisits
          are a median **2.45 s** apart: the tracker rematching somewhere it saw seconds ago,
          with no drift accumulated. Nothing to correct.
        - **Correlated by construction**, as the operator pointed out: the graph's edges ARE the
          odometry deltas, so SLAM can only redistribute VO's error. Before the first closure
          the two are identical by definition.
        - **The effect is under the noise floor.** SLAM-vs-VO is 0.70 m median; the same
          pipeline at a different replay rate differs from ITSELF by 1.13 m (1.7k).

        Answering it needs a run built for the question: a closed circuit, returned to the
        start, with `tape_metres.txt`. That is 5.1/5.2 and 1.7d — this log cannot stand in.

  - [x] 1.7j **SLAM starves the tracker it is built on.** Enabling SLAM turns the observation
        and landmark exports on INSIDE the Odometry config, so the cost lands in `Track()`:
        9-12 ms without, **16-33 ms with, spiking to 133 ms**. At 1.0x the budget is 50 ms/set,
        so it overran, dropped sets, and the widened gaps made the TRACKER fail — a 3.42 m step
        appeared that is simply absent from a clean run, and the operator spotted it in the
        scene. Every SLAM figure recorded before this was measured at 13-16 Hz against the
        20 Hz the data supports.

        At 0.4x (125 ms/set) the VO returns to the baseline exactly: 859 vs 861 poses,
        22.21 vs 22.24 m, worst step 0.39 vs 0.47 m. `replay_host.sh` now defaults `SLAM=1` to
        0.4x and warns above 0.6x. **Note this is offline-only**: the TX2 cannot slow time, and
        its 50-90 ms `Track()` against a 50 ms budget means the degraded trajectory IS the
        real-time result there.

  - [ ] 1.7k **Offline replay is not reproducible, and the flags do not fix it.** Two lossless
        replays of the same bag at 0.4x and 0.2x (1153 and 1155 of 1155 sets) gave trajectories
        a median **1.13 m** apart — larger than the effect anyone would be trying to measure.

        `Odometry::async_sba` and `Slam::sync_mode` put bundle adjustment and SLAM on background
        threads, so iterations-per-frame depend on wall-clock arrival. **Turning them off made it
        worse**: synchronous BA is slower, so the node dropped MORE sets (1022 vs 1153 at 0.4x)
        and different ones, and the two rates then diverged by a median **3.28 m**. The
        frame-drop difference dominates the threading. Both are parameters now, left at the
        library defaults.

        The fix is to take the wall clock out of the loop entirely — read the bag in-process and
        call `Track()` per set, no DDS, no real-time coupling
        (`retarget-vo-to-imx296-rig` 5.10c). **Until that exists, treat any offline difference
        below ~1 m as noise**, including every VO-vs-SLAM comparison in this file.

        **Cost: 20.02 Hz -> 12.99 Hz on the host** (687 poses against 1149 without SLAM). Enabling
        SLAM forces `enable_observations_export` and `enable_landmarks_export`, because
        `Slam::Track` takes `Odometry::State` and `GetState()` throws without them — so the export
        we keep off for rate runs is not optional here.

        **Do NOT read this run as "loop closure fixed the teleport."** The 50 m jump at t = 47 s
        is absent — the worst step is 1.49 m at t = 46.74 s, in the same saturation window — but
        the set sequence differs (13 Hz vs 20 Hz), so it is not a controlled comparison. Whether
        that is SLAM, the forced export, or simply a different frame sequence is untested.
  - [~] 1.7l **This rig meets the condition in cuVSLAM issue #136, and we have never been
        able to see it.** Read from source at `69e2f29`, not inferred:

        A SLAM landmark is born in a staging area and only counts for loop closure once
        promoted into the map. The promotion test is `lsi_grid.cpp:405-407`:

        ```cpp
        activate &= !IsLandmarkInAnyFrustum(rig_, cam_ids, cams_from_world, xyz);
        activate &= !GetLandmarkRelation(id, pose_graph_head);
        ```

        *Any* frustum is the union over every camera in the rig, and `IsLandmarkInFrustum`
        has its normalised-coordinate bounds commented out (`lsi_grid.cpp:48-53`), so there
        is no far plane — distance alone never puts a landmark outside. A stereo pair drops
        a landmark from the union as soon as the rig turns. **Our eight virtual pinholes
        close a 360 deg ring, so the only way out is vertically, past the top or bottom of
        the carve.** That is #136's "full-coverage rig" case exactly.

        Two other drains exist, and both are worse than the one they replace:
        - tracking loss promotes the WHOLE staging area at once (`lsi_grid.cpp:409-412`) —
          a mass promotion at the moment the poses backing it are worthless;
        - a 60 s staging lifetime (`max_staged_landmark_lifetime_ns_ = 60e9`,
          `lsi_grid.h:177`) force-promotes stale entries.

        **The 60 s drain has never fired in any run we have.** Both SLAM replays are
        **57.69 s of DATA time** — the clock cuVSLAM keys on is the image stamp, not the
        0.4x replay's 143 s wall clock. We are 2.3 s under the only reliable drain, by luck,
        and the run that first exceeds 60 s will behave differently from every run recorded
        so far.

        **What the existing bags do and do not show** (`scripts/vo/slam_cost_and_map.py`,
        figure at `datasets/replay_out/slam_cost_and_map.png`):

        | | obs_slam_v6 | obs_slam_v9 | vo_clean_04 (no SLAM) |
        |---|---|---|---|
        | data time | 57.69 s | 57.69 s | 57.69 s |
        | loop closures | 9 | 12 | — |
        | last closure | data t=39.2 s | data t=39.6 s | — |
        | silence after | 18.5 s | 18.1 s | — |

        Closures DO fire, so promotion is happening by some route — most likely the vertical
        escape, since our ring is closed horizontally but not vertically. Then they stop at
        t≈39.6 s in both runs and never resume.

        **A claim made and withdrawn in the same session:** `/cuvslam/landmarks` grows
        monotonically to ~54 k with zero thinning events, which looks exactly like #136's
        "accumulate indefinitely". It is not evidence. `GetFinalLandmarks` returns
        `impl->final_landmarks` (`cuvslam2.cpp:807-813, 892`), a map that is only ever
        inserted into and never erased — an odometry track dump whose unbounded growth is by
        design and which says nothing about the SLAM map's staging.

        **MEASURED 2026-09-10, and the answer is NO — #136 does not bite on this rig.**
        Host build stood up on the WSL box (see 1.7n) and the FULL 88.86 s bag replayed at
        0.4x, essentially lossless: 1774 sets received, 1772 poses, 1 dropped, 1 skew-rejected.
        This is the first run that ever crossed the 60 s staging lifetime.

        | | 0.4x | 1.0x |
        |---|---|---|
        | sets of 1774 | 1773 (99.9%) | 1721 (97.0%) |
        | data time | 88.6 s | 88.6 s |
        | promoted map landmarks | **62,865** | 58,921 |
        | loop closures | 21 | 11 |
        | pose-graph optimisations | 394 | 381 |

        The map is populated and grows **smoothly from the first seconds** — it is not stuck
        near zero, the `SLAM MAP IS EMPTY` guard never fired, and there were **zero
        single-frame jumps > 500 landmarks**, so no mass staging flush ever happened. Crossing
        60 s produces no step at all (58,881 -> 59,002 -> 59,145 across t=59/60/61 s), which
        means the lifetime drain is not what is doing the work: promotion is happening
        continuously by the ordinary route.

        **So the earlier reasoning here was right about the condition and wrong about the
        consequence.** The frustum union really is closed horizontally, but the ±45° carves
        leave the ring open vertically, and that is evidently enough to drain staging
        continuously. Meeting #136's stated condition is not sufficient to reproduce it.

        Kept as `[~]` only because one thing remains unchecked: whether a rig with a *wider*
        vertical carve, or a genuinely enclosed scene, would close that escape route.

  - [ ] 1.7m **Issue #77 (per-set cost growing with trajectory length) is untestable on the
        bags we have, and the reason is worth writing down.** #77 plots `track()` climbing
        ~10 ms -> 50 ms+ against frame index on a 12-camera rig. Neither of our two timing
        sources can show that:
        - the node's periodic report prints a 5 s windowed MAXIMUM, which hides a trend;
        - the interval between recorded messages is floored by the replay rate — 0.4x of a
          20 Hz log is 125 ms/set, and both SLAM runs sit flat at a 124.6 ms median across
          all four quarters of the run. A 10->50 ms growth is entirely under that floor.

        **A rank correlation on the overruns is not a substitute.** Spearman rho of
        (interval excess over the floor) against set index gives +0.284 / +0.273 on the two
        SLAM runs — and **+0.255 on `vo_clean_04`, which has SLAM off**. The control matches
        the treatment, so that statistic is measuring noise structure, not cost growth. Do
        not report it.

        What the bags DO show is a localised burst, not a trend: sets overrunning the floor
        by >25% run at 0.9-2.6% for deciles 1-5, spike to **14.8 / 39.1 / 17.4%** in deciles
        6-8, and fall back to 0-1.7% in deciles 9-10. Reproduced in both SLAM runs at the
        same deciles, and **absent from the no-SLAM control** (flat ~1% throughout). The
        burst overlaps the loop-closure window but outlasts it — closures end at set ~802,
        the burst runs to ~922 — so it is not simply "loop closures are expensive" and is
        not yet attributed.

        **WE DO REPRODUCE #77. The first answer here said we did not, and it was measuring
        the wrong component — caught by the operator, 2026-09-10.**

        `track_us` times `tracker_->Track()`, which is `cuvslam::Odometry`: the FRONTEND —
        PnP every frame, plus triangulation and SBA on keyframes. #77's degradation is
        loop-closure matching and pose-graph optimisation, which are the SLAM BACKEND. With
        `slam_sync_mode=false` (the library default, and what every run above used) those
        run on cuVSLAM's OWN worker thread (`async_slam.cpp:142-201` pushes a keyframe and
        returns), so **no timer in this node could see them**. A frontend-only plot returns
        "no degradation" no matter how bad the backend gets.

        Worse, the one column that might have caught it was broken: `slam_us` was a running
        max over the 5 s report window, written per set as though it were a per-set value —
        154 distinct values across 1773 rows, in repeating runs. Now `slam_track_us`, a true
        per-set measurement, plus a `pgo_events` column.

        **Re-measured with the backend INLINE** (`SLAM_SYNC=1` → `slam_sync_mode:=true`),
        full bag at 0.2x, 1772 sets / 512 keyframes / 88.6 s, 404 PGOs, 23 loop closures.
        Keyframes only, scene held healthy (`frame_landmarks > 300`):

        | landmarks in map | n | **backend** `Slam::Track` | frontend `Odometry::Track` |
        |---|---|---|---|
        | 0–10k | 47 | 91.8 ms | 64.2 ms |
        | 10–20k | 55 | 89.6 ms | 59.2 ms |
        | 20–30k | 52 | 165.7 ms | 15.6 ms |
        | 30–40k | 63 | 153.6 ms | 17.2 ms |
        | 40–50k | 120 | 199.5 ms | 20.1 ms |
        | 50–60k | 106 | **255.4 ms** | 19.8 ms |

        **Backend grows ~2.8x while the frontend stays flat.** By thirds: 107.8 → 164.7 →
        264.7 ms. Non-keyframe backend cost is **0.1 ms**, confirming the backend does work
        only on keyframes — the frontend runs at frame rate, the backend at keyframe rate.

        This is with `slam_max_map_size:=0` (unlimited pose graph), which is the replay
        default and the configuration most exposed to the growth. The 300-pose cap is the
        header's real-time answer and would bound it.

        **It also explains the async results.** At 1.0x async, loop closures halved (21 → 11)
        and the map came out smaller — the backend thread falling behind, which is the same
        cost showing up as lost work instead of lost time. New in the inline run: the map
        **prunes** at t≈83 s (65k → 33k), a thinning that never appeared in the async runs.

        Figure: `datasets/replay_out/slam_backend_sync_rate0.2.png`.

        **Retained below, as the record of what the frontend does** — it is a real result,
        it just does not answer #77:

        The raw decile table looks exactly like #77 and is a trap. Keyframe `Track()` goes
        **16.8 ms in the first third to 73.4 ms in the last** — a 4.4x rise — while
        non-keyframe stays flat (26.3 -> 28.8 ms). That is #77's signature.

        It does not survive attribution. Two things move together over a run: map size, and
        scene quality. `frame_landmarks` collapses from ~480 to ~100 at t≈50 s (the documented
        exposure/texture window), and keyframe fraction collapses with it (85% -> 5%), so the
        late deciles are a handful of distressed frames. **Hold the scene healthy
        (`frame_landmarks > 300`) and bin on map size instead:**

        | promoted map | n | median keyframe `Track()` |
        |---|---|---|
        | 0–10k | 47 | 15.9 ms |
        | 10–25k | 73 | 16.9 ms |
        | 25–40k | 90 | 16.9 ms |
        | 40–55k | 221 | 23.4 ms |
        | 55–70k | 9 | **15.8 ms** |

        Flat across a **10x** growth in the map. The 1.0x run agrees independently (30.5 /
        19.9 / 26.2 / 29.2 ms over 6k -> 45k). So the FRONTEND genuinely does not degrade
        with map size — that part of the analysis survives. What it cannot do is answer #77,
        which lives in the backend.

        Note this also retires the rank-correlation result above: it was noise.

  - [x] 1.7n **Host cuVSLAM build on the WSL box — it works, and `/usr/local/cuda` being
        empty was not the blocker it looked like.** `docker-compose.host.yml` already had the
        whole chain; what was missing was a CUDA toolkit to mount. Rather than a multi-GB
        install, extract one from the `nvidia/cuda:12.6.3-devel-ubuntu24.04` image that was
        already pulled locally and point `CUDA_HOST_MOUNT` at it:

        ```
        docker create --name cudaextract nvidia/cuda:12.6.3-devel-ubuntu24.04
        docker cp cudaextract:/usr/local/cuda-12.6 ~/cuda-12.6 && docker rm cudaextract
        CUDA_HOST_MOUNT=~/cuda-12.6 docker compose -f docker-compose.host.yml build
        CUDA_HOST_MOUNT=~/cuda-12.6 docker compose -f docker-compose.host.yml run --rm build-cuvslam-host
        CUDA_HOST_MOUNT=~/cuda-12.6 docker compose -f docker-compose.host.yml run --rm build-ws-host
        CUDA_HOST_MOUNT=~/cuda-12.6 SLAM=1 ./scripts/vo/replay_host.sh <bag> 0.4
        ```

        GPU passthrough under WSL2 works (`--gpus all` sees the RTX 2000 Ada); ROS Foxy's
        focal apt repo is still live. `build_cuvslam_host.sh` had a **wrong default**: sm_86,
        from a machine with an RTX A2000. This box is Ada, sm_89. A wrong arch does not fail
        loudly — CUDA JITs the embedded PTX, so it builds and runs and is merely slower, which
        is the exact quantity #77 is about. It now asks `nvidia-smi`.

        **20 fps is viable on this host**, contrary to 1.7j's finding on the previous one:
        1721 of 1774 sets at 1.0x (3% loss) against 1773 at 0.4x. But loop closures halve
        (21 -> 11) and the map comes out smaller (58.9k vs 62.9k), so 0.4x remains the right
        default for anything where the SLAM layer is the measurement.

  - [ ] 1.7b Publish loop-closure events and the pose graph; add a viewer pane keyed on
        `lc_status`, plus a corrected-trajectory line beside the raw VO one.
  - [ ] 1.7c Measure the cost on the TX2 before believing any of it. `Track()` there is already
        50-90 ms/set against a 20 Hz budget (`retarget-vo-to-imx296-rig` 5.10); SLAM runs its own
        thread unless `sync_mode`, but the board has no headroom to give away.
  - [ ] 1.7d Re-run 5.1/5.2 with loop closure on and state the drift both ways. Needs a
        return-to-origin run: this log's end-to-start distance moved 5.55 m -> 5.03 m with SLAM,
        which is only a drift figure if the rig was physically returned to its start pose, and
        it was not.
  - [x] 1.7e **Answered, and the answer is NO: loop closure does not rescue the teleport.**
        This was claimed twice in the opposite direction here and both claims were wrong. Both
        rested on comparing a `slam_path` that ended at 41.6 s against a VO trajectory of 54 s —
        the optimised path never covered the failure, so it neither absorbed it nor was reset by
        it. See 1.7h for why the path was short.

        On a run where the path DOES cover the failure:

        ```
        t=47.44s  [-0.05  3.37 -5.98]  step= 0.000 m   <-- VO teleports here
        t=47.49s  [40.71 32.63 -4.86]  step=50.191 m   <-- so does the optimised path
        ```

        | | path | max step | steps > 0.5 m |
        |---|---|---|---|
        | optimised | 93.94 m | **50.191 m** | 9 |
        | pure VO | 94.55 m | **50.191 m** | 8 |

        Same jump, same instant, same destination — and the frozen poses before it are in the
        optimised path too. **Why it cannot help:** pose-graph edges come from odometry, so a
        50 m odometry delta becomes an edge, and no loop closure spans the jump to contradict
        it. Loop closure corrects drift; it cannot invent evidence the images did not contain.
        The fix for the teleport stays where 5.11/5.12 put it — the exposure.

  - [x] 1.7h **`/cuvslam/slam_path` was published only on loop closures, so it always ended at
        the last one.** 41.6 s of a 54 s run on one replay, 37.4 s on another — each exactly its
        final closure, which read as SLAM giving up mid-run and made every optimised-vs-VO
        comparison a comparison of different intervals. Now published every 20 sets (1 s at
        20 Hz), so the last one is at most a second short of the end even though the replay
        wrapper SIGKILLs the node.

        `max_map_size` was investigated first and is NOT the cause — with it unlimited the path
        stopped *earlier*. The 300-node cap is real (the graph reached 299 on one run) and would
        bite on a longer session, so it is exposed as a launch argument with the header's
        real-time default kept.

  - [x] 1.7f **The SLAM trajectory must be re-read, never accumulated.** The first version drew
        `/cuvslam/slam_odometry` — the CURRENT corrected pose — into a growing line. A loop
        closure re-optimises the whole graph, so such a line is stale everywhere behind the head
        and splices pre- and post-optimisation segments at each event: it stepped visibly and
        came out **worse than the raw VO**, which is how the operator spotted it.

        cuVSLAM's own app states the rule — *"if slam is enabled, overwrite all slam poses in the
        end after LCs and PGOs"*, re-reading `get_all_slam_poses()`
        (`tools/cuvslam_app/cuvslam_app.py`). Its euroc C++ example appends `GetPose()` and has
        the same artifact, so the python app is the reference, not the C++ one.

        Now: the node publishes `/cuvslam/slam_path` from `GetAllSlamPoses()` on each closure and
        the viewer draws the last one statically. Two further fixes came with it:
        - loop closures are **accumulated and de-duplicated on timestamp** (the euroc example's
          `reported_loop_closures` set). `GetLoopClosurePoses` returns a rolling last-10 window,
          so taking the latest message loses old closures and re-counts live ones — 7 events had
          been drawing 10 markers.
        - `/cuvslam/loop_closure_edges` publishes every pose-graph edge whose node ids are
          **non-adjacent**. A sequential odometry link joins consecutive ids, so a non-adjacent
          edge is a loop link, and drawing it shows what each closure connects BACK TO — a marker
          alone says only that a loop closed. cuVSLAM leaves pose-graph visualisation as a
          commented-out "future extension" in the euroc example, so this reading is ours.

  - [x] 1.7i **The rig follows the OPTIMISED pose, and the closure markers sit on the optimised
        line.** Two display faults the operator caught:

        The rig body, camera frusta and images hung off the pure-VO pose, so at every tracking
        failure the rig flew away while the optimised line stayed — reading as though the two
        were unrelated. It now takes its translation from `slam_path`, matched per timestamp
        (median divergence 0.589 m, max 5.386 m, so it is not cosmetic). Rotation still comes
        from odometry: `slam_path` carries positions worth more trust, but its orientation is
        not separately validated.

        The closure markers sat ~0.16-0.32 m off both trajectories, because the pose stored with
        a closure is the one current when it FIRED and later optimisations move the trajectory
        under it. `/cuvslam/loop_closures` is now a `nav_msgs/Path` rather than a `PoseArray` —
        PoseArray carries one header stamp for the whole array, so a consumer cannot tell when
        each closure happened and can only guess by nearest-point search. With per-pose stamps
        the viewer places each marker on the optimised trajectory at its own instant: measured
        0.000 m, exact.

  - [x] 1.8 **The panorama prototype was MIRRORED, and the operator caught it from the images.**
  The brick wall sits on the right in `cam1 -45` and `cam2 +45` — both of which point forward,
  az **0.0** and **+1.1 deg**, which is why it appears in both — and the panorama put it on the
  left.

  `pano_maps()` built its column axis as `az = 2*pi*(i+0.5)/out_w - pi`, so azimuth INCREASED
  left to right. The frame is rig FLU (+x forward, **+y left**), so increasing azimuth sweeps
  toward the left and scanning the panorama rightwards turned leftwards: the whole image was
  flipped. Verified per camera — before the fix the two right-pointing carves (`cam2 -45` at
  az -89.1, `cam4 +45` at -89.5) landed in the LEFT half; after it they land at columns 957 and
  958, and the left-pointing pair at 320 and 321.

  Fixed to `az = pi - 2*pi*(i+0.5)/out_w`. `make_panorama.py` imports `pano_maps` so it
  inherits the fix. **The Rerun panorama is pure Python — the C++ node plays no part in it.**

- [ ] 1.9 **`bev_panorama_node.cpp` looks 90 deg out, and this is DEPLOYED code.** Recorded after
  claiming in 64481f7 that "the deployed node was right and the prototype had diverged". Only
  half of that is true: the node's azimuth SWEEP is right, its FRAME is not.

  The node builds `dr = {sin(az)cos(el), cos(az)cos(el), sin(el)}` under the comment
  `ray in rig frame: X=right, Y=forward, Z=up`, then rotates straight into the camera with the
  rotation out of `rig_in_cam1`. But `rig_extrinsics_imx296.yaml` defines that block as poses in
  **cam1's RAW OPTICAL frame — x right, y DOWN, z along the axis** — with cam1 identity by
  construction. So for cam1 the node's centre ray `(0,1,0)` is cam1's **+y, straight DOWN**, and
  `roll180` only turns it into UP. The panorama's horizon band is centred on the vertical axis.

  | | centre ray (az=0, el=0) in cam1 optical | |
  |---|---|---|
  | node | `[0, 1, 0]` | +y = down |
  | python `pano_maps` | `[-0.707, 0, 0.707]` | along the optical axis |
  | | **90 deg apart** | |

  The Python applies `R_rig_cam1` (`ground_plane.yaml`) to get FLU -> cam1 optical first; the
  node has no equivalent step. Note this is a SECOND candidate cause for "the stitch never
  registered" — `panorama_params.yaml` blames the retired IMX219 intrinsics, which were real, but
  a 90 deg frame error would survive fixing them.

  - [ ] 1.9a **Verify on the board before changing anything.** This is static analysis; the node
        needs Argus and has not been run since. Point the rig at something unambiguous, look at
        the published `/bev/panorama`, and settle it the way the mirroring was settled — from
        the image.
  - [ ] 1.9b If confirmed, the fix is to compose `R_rig_cam1` into the node's ray before the
        per-camera rotation, exactly as `pano_maps` does, rather than to re-define its rig frame
        comment to match what it currently computes.

- [ ] 1.7g **The panorama auto-depth follows the map's outliers.** On the full-rate run the
        sphere radius ranged **2.18-23.93 m** (median 2.52). The spikes are
        `scene_radius_near_pose` tracking a landmark cloud that reaches far past the room (5.4:
        310 m in a 14 m log), and where the radius spikes the stitch is effectively at infinity,
        so close scene ghosts. Clamp the radius to the same `--map-radius` the display already
        applies, or use a robust percentile rather than the raw cloud.

## 2. BEV prototype — satisfies `add-bev-ground-stitch` 2.6

- [x] 2.1 **`bev_maps()`: the ground-plane projection the node will implement, in Python.** Output cell
  → point on the plane in rig FLU → `T_cam_rig` → Mei projection → source pixel, per camera, with
  overlap weights. This is the offline mosaic check 2.6 asks for, and it did what 2.6 predicts: a
  wrong frame transform and a stale-image bug were both obvious here and would have been subtle in CUDA.

- [x] 2.2 **Supports a tilted plane, not just a level one.** Grid basis is built from the plane normal
  (`e1` = forward projected onto the plane, `e2 = n × e1`). **Verified**: with `normal = [0,0,1]` it
  reproduces the previous level-plane tables bit-exactly, so tilt support did not perturb the level case.

- [x] 2.3 **Seam sanity check that needs no ground truth**: seams land on forward/left/right/rear, the
  bisectors of ±45°/±135°, confirming `R_rig_cam1` agrees with `rig_in_cam1`. Free, and it is the check
  `add-bev-ground-stitch` 0.2 wants for "wrong file here rotates the entire mosaic".

- [x] 2.4 **Incidence cap instead of frame skipping** (commit `581cc64`). See design Decision 4. A
  ground point at radius `r` is seen at incidence `arctan(r/h)`, so `--bev-max-incidence` (75°) caps
  the painted radius at `3.73 h`. Coverage of a 4 m extent: **96.4 % at h = 1.5 m — identical with the
  cap off, so it is free at working height** — 53.7 % at 0.9 m, 2.5 % at 0.2 m, 0 % at 0.02 m.

- [x] 2.5 **Fixed a user-visible "frozen and stretched" BEV.** Both symptoms were `h → 0`. Frozen: an
  `h < 0.05` skip left the previous frame on screen for **64 of 220 frames**, because Rerun holds the
  last logged image for an entity. Stretched: at h = 0.05–0.15 m the camera is nearly *in* the plane
  and the grid projects to a grazing sliver. **Rule adopted: never suppress bad output by skipping a
  log — log blank.**

- [ ] 2.6 Cross-check the Python mosaic against the CUDA implementation once `ros2/bev_ground/` exists,
  cell for cell on one recorded set. The prototype's value is as a reference; that requires diffing it.

- [ ] 2.7 Characterise above-plane smearing quantitatively (`add-bev-ground-stitch` 4.3). The prototype
  can measure it — an object of known height, and how far its top lands from its base — and no such
  number exists yet.

## 3. Ground plane from VO landmarks — interim, feeds `add-bev-ground-stitch` §1

- [x] 3.1 **Discarded the global fit; it was measuring map drift** (superseded by `a13dc77`). A single
  RANSAC plane fitted over all landmarks in the **odometry frame** reported the height swinging
  **0.03 → 1.56 m** on a walk where the operator states the rig was carried at **constant height on
  their head**. The assumption it breaks is global map consistency: pure VO drifts, so landmarks from
  second 3 and second 25 are in mutually rotated frames and no single plane fits both.

- [x] 3.2 **`plane_near_pose()` — fit locally, in the pose's own frame** (commit `a13dc77`). Landmarks
  within 5 m of the current pose, expressed in that pose's rig frame; height histogrammed into 56 bins
  over (−3.0, −0.2) m; floor taken as the **lowest bin exceeding 25 % of the peak** — not the largest
  consensus set, because indoors a wall is a bigger plane than the visible floor — refined by SVD
  within a 0.12 m band, rejected if the normal is >14° off vertical.
  **Result on the same walk: 1.36 m median, std 0.20 m, range 1.04–1.99 m**, tilt 0.1–0.7° (against
  the global fit's 0.03–1.56 m and 2.95°). 191 distinct tables over 220 frames.

- [x] 3.3 **Cross-check at matched frames**: a seam-NCC sweep peaks at h ≈ 0.90 m; landmark RANSAC
  gives 0.97 m at the same frames — **agreement to 7 cm**. Recorded as a cross-check only: both derive
  from the same extrinsic translations, so this is self-consistency, **not** evidence about metres.

- [x] 3.4 **`ground_plane.yaml` deliberately left `status: unmeasured`, `height_m: null`.** Writing a
  0.20 m-spread estimate with a suspected systematic error into the calibration would launder an
  estimate into a measurement. Tasks `add-bev-ground-stitch` 1.1–1.5 (AprilGrid on the floor) still stand.

- [ ] 3.5 **Resolve the suspected 15–20 % scale underestimate.** The local floor sits **1.36 m** below
  the camera; a rig carried on an adult head should be ~1.60–1.75 m. If real, the error is in the
  extrinsic translations in `config/rig/rig_extrinsics_imx296.yaml`. **Blocked on one input: the
  measured height of the rig when worn.** This is the cheapest scale check available and it has been
  asked for and not yet answered.

## 4. Panorama prototype (commit `97877fa`)

- [x] 4.1 **`pano_maps()` / `render_pano()`: equirectangular stitch of the four raw fisheyes** on the
  *current* rig, which `bev_panorama_node` cannot serve (KB-only, IMX219-configured). 1280×356,
  azimuth 0 = rig forward, +azimuth toward left, ±50° elevation, feathered overlap weights.
  **99 % of the band covered, 85 % of it by 2+ cameras.**

- [x] 4.2 **Bearings verified against the extrinsics, not eyeballed.** Optical-axis azimuths
  **+45.0 / −43.8 / +134.4 / −134.8°**; blend-weight circular-mean centroids
  **+44.6 / −44.0 / +133.7 / −134.6°**. Both match the declared ±45 / ±135 layout.
  **Trap recorded**: the first check used `wt[c].sum(0).argmax()` and reported all four cameras
  mismatched. The feather weight is clipped at 1.0, so there is a broad plateau with no unique maximum
  and `argmax` returns an arbitrary point on it. False alarm; use a circular mean.

- [x] 4.3 **Finite-radius sphere to compensate the baselines.** Measured baselines are
  **0.153 / 0.155 / 0.219 / 0.221 / 0.153 / 0.161 m** and the scene is 2–4 m away, so rotation-only
  stitching ghosted visibly. Rays are cast as `v = (depth·d_cam1 − t_c) @ R_c` (construction borrowed
  from `scripts/calib/pano_tuner.py`). A later `auto` mode tracks the landmark cloud per frame at a
  low percentile (p25 — ghost displacement goes as baseline/depth, so erring near costs far less than
  erring far); measured 2.01–3.87 m, median 2.45 m over the walk.
  **Real but second-order** — see the correction in 4.5. This helps; the blend geometry mattered more.

- [x] 4.4 **Radius selected by measurement — after two metrics gave confidently wrong answers.**
  See design Decision 5. (1) Mean |diff| between overlapping cameras ranked **infinity best** because
  per-camera exposure differs by ~40 grey levels and swamped the geometry. (2) Edge NCC averaged over
  "whichever pairs overlap" was unfair, because the overlap set shrinks with radius (18 vs 12 pairs),
  so radii were scored on different content. (3) Edge NCC over a **fixed** set of the four adjacent
  90° pairs peaks cleanly at **3.0 m: NCC 0.058, all four pairs positive (0.082/0.043/0.046/0.061),
  against −0.007 at infinity.** Confirmed visually against `/tmp/pano_compare.png`.

- [x] 4.5 **CORRECTION — most of the ghosting was the blend geometry, not the method.** This task
  previously read "residual ghosting recorded as method-inherent, not tuned away". That was wrong,
  and the sphere-radius work in 4.3/4.4 was treating a symptom.

  **Root cause.** Weights feathered on absolute angle from each optical axis,
  `w = clip((fov_half − th)/feather, 0, 1)`, which is 1.0 for all `th < 65°`. Adjacent cameras are
  **90° apart**, so at the bisector both sit 45° off axis and **both get weight exactly 1.0**. The
  output was a literal 50/50 double exposure over a ~60° band at each of the four seams.
  **Measured: 65 % of the horizon, 39 % of the whole canvas.** No sphere radius can fix this — the
  radius only makes two views agree at *its* radius, and everything nearer or further doubles at
  full strength across that band.

  **Fix** (commit `22144d5`): rank cameras by angular distance and cross-fade only over
  `--pano-seam` (8°) where two are nearly equidistant, so each pixel comes from the camera looking
  most directly at it. Ranking against the best **valid** camera keeps weight 1.0 where only one
  camera sees, so coverage stays 100 % with no normalisation dip.

  **Detail retained in the overlap region: 0.87 → 1.85** relative to a single camera; equal-blend
  area 39 % → 4 %. Above 1.0 because nearest-axis also selects the *sharper* view — a ray far off
  axis lands in the fisheye's compressed periphery. **For scale, the per-frame auto radius alone
  moved the same number 0.87 → 0.96.** The blend geometry was worth roughly ten times the radius.

  What remains genuinely method-inherent is much smaller: a narrow mis-registered band at each seam
  for scene off the sphere radius, plus the photometric step (4.6).

- [x] 4.8 **The deployed `bev_panorama_node` has the same defect.** `bev_panorama_node.cpp:295`
  computes `fw = (fov_max - th)/feather` — the identical construction. With its shipped parameters
  (`fisheye_fov_half_deg: 80`, `feather_deg: 20`) weight is 1.0 for `th < 60°`, giving a **30°-wide
  50/50 band at each seam, ~33 % of the horizon**. Milder than the viewer's was, same mechanism.
  This matters now because the node has just been ported to Mei/IMX296 (`18319fb`), so it will ghost
  on the TX2 for the same reason. **Not fixed here** — the node is another change's surface.

- [ ] 4.9 Port the nearest-axis seam into `bev_panorama_node`. It is a change to the host-side table
  build only (the CUDA kernel already just weight-blends precomputed maps), so the kernel is untouched.
  Needs a second pass over the cameras to rank them, which the current single-pass loop does not do.

- [ ] 4.6 **Quantify the per-camera brightness step at each seam** — feeds `add-bev-ground-stitch` 5.1,
  which wants exactly this number so "a photometric defect is never later mistaken for a geometric one".
  The mismatch is plainly visible in the render and the ~40 grey-level figure above is an aggregate,
  not a per-seam measurement. Photometric compensation itself stays out of scope (`add-bev-ground-stitch` §5).

- [ ] 4.7 Decide whether the panorama prototype should be retargeted into `bev_panorama_node` (Mei model,
  IMX296 extrinsics, finite depth) or left as a diagnostic. `add-bev-ground-stitch`'s proposal
  explicitly leaves `surround-panorama` alone; this prototype changes the cost of revisiting that.

## 5. Findings that belong to other changes

- [x] 5.1 **Vertical drift is pure-VO drift, and its root cause is structural.** No IMU and no loop
  closure means nothing pins roll and pitch to gravity, so forward motion leaks into apparent vertical
  motion. `bev_imu` + MPU9250 already exist in the repo and are the structural fix. Recorded against
  `retarget-vo-to-imx296-rig` 5.2 (drift).

- [x] 5.2 **But not all of the vertical motion was drift — the operator was right and the fit was
  wrong.** A local floor measurement along each pose's **own** up axis (immune to odometry tilt) puts
  the floor 0.22 m below the camera for t < 6 s and ~1.27 m from t = 7.4 s. The t = 6–9 s segment has
  **dz = +1.22 m with only 0.30 m of horizontal motion**, which drift cannot produce: that is the rig
  being lifted onto the head. Drift is the ±0.4 m wander *after* it. **Operator ground truth beat the
  fit twice in this session; treat it as the reference.**

- [x] 5.3 **One virtual camera is effectively blind in this log.** vcam3 (cam2's +45° carve) yields
  **~12 features/frame against 2402/frame overall** — a person's head occludes it. Relevant to
  `retarget-vo-to-imx296-rig` §5: the run's tracking quality is not evenly sourced, and a repeat should
  keep the operator out of that carve.

- [x] 5.4 **The map has long-range outliers.** `obs_20260903_140714`: 440 odometry poses, 440
  observation sets, 26 759 landmarks, of which **24 234 within 20 m — the map reaches 310 m in a 14 m
  indoor log**. Rerun's 3D auto-framing collapses the room to a few pixels without `--map-radius`.

- [ ] 5.5 **Absolute scale is still unvalidated** — `retarget-vo-to-imx296-rig` 5.1 and
  `add-bev-ground-stitch` 4.1 both need it. Scale enters **only** through the extrinsic translations,
  so the fix on failure is the extrinsics; never post-scale the VO output. In order of value:
  - [ ] 5.5a Tape-measure a 5–10 m straight run and compare reported translation. Free, and the best
        anchor available. Nothing else should be built before this is done.
  - [ ] 5.5b ToF lidar: regress measured range against landmark range over **several** distances —
        slope is the scale error, intercept is a bias. One distance cannot separate the two.

        **Correction (2026-09-06): the channel is not useless, it is intermittently blocked.**
        5.0g called the beam "pointed at the rig or the operator" from its 0.30 m median and
        wrote it off. The operator's account is that a hand covers it part of the time, and
        segmenting on that is what the data supports: readings that are **both** under 0.5 m
        **and** near-constant (rolling std < 5 cm) are **68.6 %** of the run; the other
        **31.4 %** is scene, in three stretches over 1 s — t = 2.7-5.1, **t = 12.6-22.0**
        (9.4 s, 187 readings, 0.04-2.03 m) and t = 53.0-57.7. The middle one sits inside the
        well-exposed room, which makes it the only usable window.

        **But it still cannot anchor scale as recorded.** Inside that stretch |d(range)| per
        50 ms sample is a median 2 cm but p90 24 cm and max 158 cm, implying beam speeds to
        31.6 m/s: the beam is sweeping across objects, not tracking one surface. A regression
        needs the range and the VO to be measuring the same thing.
      - [ ] 5.5e **What a usable range run looks like.** Deliberately: keep the hand off the
            sensor, aim it at one flat wall, and translate along the beam through several
            distances (say 3 m to 0.5 m) with the wall filling the beam throughout. Then
            d(range) is directly comparable to the VO translation projected on the beam, and
            the regression slope is the scale error. Needs the rangefinder extrinsic, or at
            least its axis, which is still unmeasured — with 3+ distances the axis can be
            solved for alongside the scale.
  - [ ] 5.5c T265: the weaker reference. Compare **pairwise pose distances > 2 m apart** so the ~0.2 m
        lever arm between the devices is negligible, rather than aligning trajectories directly.
  - [ ] 5.5d Write the analysis scripts once a bag containing either sensor exists. Offered, not started.
  - [x] 5.5f **Known-object cross-checks from `multicam_full.rrd` (2026-09-07) show a
        position-dependent underestimate, not one defensible global correction.** Each selected Rerun
        feature was matched to the preceding retained observation sample, with the viewer's default
        180-degree upright display rotation undone, then triangulated from all observations of its
        cuVSLAM ID. The resolver is `scripts/vo/measure_replay_anchor.py --upright`.

        | object / physical dimension | reconstructed | error | implied correction |
        |---|---:|---:|---:|
        | Bosch fridge front width, 0.6000 m | 0.5828 m | -2.9% | 1.0295 |
        | Siemens range-hood width, 0.8900 m | 0.8196 m | -7.9% | 1.0859 |
        | black glass height, 0.4000 m | 0.3361 m | -16.0% | 1.1903 |
        | black glass width, 0.4900 m | 0.4104 m | -16.2% | 1.1939 |
        | cabinet height, approximately 2.0500 m | 1.6221 m | -20.9% | 1.2638 |

        The fridge is the cleanest single anchor. The black-glass reconstruction has an 85.1-degree
        corner where the physical rectangle is 90 degrees, and the cabinet height is approximate;
        neither may set the metric scale. The 1.0295--1.2638 range rules out post-scaling the map.
        Keep 5.5 open until a measured straight run or a controlled multi-distance range run validates
        the extrinsic translations independently.

- [x] 5.7 **The rig logger's range channel never worked, and the first run of it found four
  separate faults.** `bev_range` + the `rangelog` service were committed 2026-09-04 (`a870470`)
  and deployed, but had never been run once — `/home/nvidia/logs` was empty. Running it produced
  cameras and IMU with a `range0.csv` containing a correct provenance header and **zero data rows**,
  which is the worst possible failure shape: it reads as "the sensor said nothing" rather than as a
  bug. The sensor was fine throughout — `lidar_present=1`, streaming `!range_cm=N pulses=M` every
  15 pulses exactly.

  1. **The container never started.** `log_rig.sh` launched `rangelog` without `EXPOSURE_US`, and
     `docker compose run <one service>` interpolates the WHOLE file at parse time, so the `capture`
     service's `${EXPOSURE_US:?...}` aborted the command. `imulog` had always passed it. The
     failure was invisible because the call ended in `>/dev/null 2>&1 || true`, and the fallback
     diagnostic asked `docker logs` for a container a failed `run` never created. Same trap as
     `retarget-vo-to-imx296-rig` 5.0c hit with the `shell` service.
  2. **SIGINT never reached the node.** `rangelog` wraps it in `bash -lc`, which does not forward
     signals to a child it is waiting on, so `stop_signal: SIGINT` killed the shell and the node
     was SIGKILLed at the end of the grace period — skipping the destructor that closes the CSV.
     `imulog` has no wrapper and never had this. Fixed with `exec`.
  3. **The CSV buffered its rows** and only flushed in that destructor, so every reading was lost
     with it. Now flushed per row: they arrive at ~2 Hz, so the cost is nothing, and a crash or a
     power cut no longer empties the file.
  4. **A leaked Argus session** failed the next capture with "no session for 0". `log_rig.sh` now
     restarts `nvargus-daemon` in preflight every time, as `retarget-vo-to-imx296-rig` 4.6 asks.

  Verified end to end on `imglog_final_20260905_091337`: 276 sets at 14 us skew, 11 426 IMU
  samples, **17 range readings**, `pulses` advancing exactly +15.

- [x] 5.8 **The range->frame join is NOT the exact integer identity `a870470` claimed.** That commit
  and the node's header both stated `pulses` and the capture side's `seq` were the same counter, so
  "there is nothing here left to solve for". They are not. Both advance one per trigger edge, but
  `seq` is Argus's **per-session** counter (starts near 0) and `pulses` is the MCU's **free-running
  lifetime** counter: on the verification log cam1 `seq` began at 3 while `pulses` was ~60 550.
  A constant integer offset separates them and **no recording writes it down**.

  It is recoverable from the two CLOCK_MONOTONIC columns, but only to about **+-1 frame**, because
  the range `t_mono_ns` is the READ instant and carries 5-20 ms of acquisition plus ~2 ms of UART
  against a 33 ms frame period — which is exactly the ambiguity the pulse counter existed to remove.
  The claim is corrected in the node comment and in the CSV header (`frame_offset = UNMEASURED`).

- [ ] 5.9 **Pin the range->frame offset exactly.** Needs the capture side to see the MCU pulse
  counter, since the range node cannot: it owns the port for the run. Cheapest option is for
  `log_rig.sh` to read the counter at preflight (it already holds the port then) and record it with
  the first frame's `t_sof`. Until this is done, anything derived from a range-to-frame association
  is good to +-1 frame and must say so — including 5.5b, which is the reason it matters.

- [ ] 5.6 Tick or annotate `retarget-vo-to-imx296-rig` 5.0c with 5.2/5.3 above — the replay it records
  is the same log, and the vcam3 occlusion qualifies its tracking result.

## 6. Wrap-up

- [ ] 6.1 Record the frame-convention trap (row-vector `P @ R.T` vs `P @ R`, rig FLU vs the deployed
  node's RFU) in `docs/` where the next person implementing a projection will find it, not only in
  this change.
- [ ] 6.2 Decide the disposition of the panorama prototype (4.7) before archiving.
- [ ] 6.3 Archive this change once §3.5 has an answer and §5.5a has been run — those are the two open
  items that change conclusions rather than add polish.

## Artefacts

Host, all under gitignored `datasets/`:

| path | what |
|---|---|
| `datasets/replay_out/obs_20260903_140714/multicam_bevfit.rrd` | 105 MB — current recording, BEV + panorama panes |
| `datasets/replay_out/obs_20260903_140714/multicam.rrd` | 77 MB — cameras + map only |
| `datasets/replay_out/obs_20260903_140714/multicam_full.rrd` | 275 MB — with fisheye domes |
| `datasets/imglog_vio1_30s_bag` | source images for all of the above |
| `datasets/replay_out/slam_cost_and_map.png` | 1.7l / 1.7m — what the OLD bags could show (replay-rate floored) |
| `datasets/replay_out/slam_backend_sync_rate0.2.png` | 1.7m — **the #77 answer**: backend inline, cost vs map size |
| `datasets/replay_out/slam_backend_sync_rate0.2_timing.csv` | per-set rows behind it (`slam_track_us` is real here) |
| `datasets/replay_out/slam_frontend_async_rate{0.4,1.0}_timing.csv` | the async runs — frontend only; `slam_track_us` is just the enqueue |

Names carry the rate and the regime on purpose: the first pair of figures was called
`slam77_full_04/10`, nobody could tell that meant 0.4x/1.0x, and both were titled as the
"#77 axis" while plotting the frontend. They are deleted rather than kept.

Regenerate the last one (reads the bags directly, no ROS environment needed):

```
python3 scripts/vo/slam_cost_and_map.py
python3 scripts/vo/slam_cost_and_map.py --timing datasets/replay_out/<run>_timing.csv
```

Commits on `feat/imx296-synced-vo`: `b513a25` (world-fixed BEV plane, superseded), `581cc64`
(incidence cap), `a13dc77` (per-pose plane fit), `97877fa` (panorama pane).

Regenerate:

```
python3 scripts/vo/rerun_multicam.py datasets/replay_out/obs_20260903_140714 \
  --images datasets/imglog_vio1_30s_bag --frames 180 \
  --bev-fit-plane --panorama --pano-depth 3.0 \
  --save datasets/replay_out/obs_20260903_140714/multicam_bevfit.rrd
```
