"""Loop closure for slam-rs VIO on the BEV rig.

See openspec/changes/add-slamrs-loop-closure/design.md. The modules, in pipeline order:

  camera     KB4 fisheye model, rig extrinsics, SE(3) helpers
  keyframes  keyframe records taken from the VIO's snapshots, and their export
  features   ORB per camera, landmark association, hash bag-of-words
  retrieval  spatial + appearance candidates, time-gated
  verify     cross-camera matching, two-step multi-camera PnP, cuVSLAM's gates
  posegraph  4-DoF pose graph (position + yaw; roll/pitch from the VIO)
  backend    LoopClosureBackend.on_keyframe: retrieve -> verify -> optimise, bounded
"""
