#!/usr/bin/env python3
"""Close the extrinsic ring: turn four independent pairwise solves into one rigid rig.

WHY. Each pair was solved from its own recording, so nothing forced the four to agree.
Composing them around cam1 -> cam2 -> cam4 -> cam3 -> cam1 left 3.63 deg and 9.2 mm of
residual: the rig they describe is not rigid, and a solver that carries features between
pairs (cuVSLAM Multicamera) would be handed four sub-frustums that disagree about where
they sit relative to one another.

Re-parameterising as three camera poses in cam1's frame (18 dof) instead of four free
edges (24 dof) makes closure structurally impossible to violate.

WHAT THIS DOES NOT DO. It does not make the calibration more accurate. A Monte-Carlo over
each pair's own reported spread says random per-pose error would leave only ~0.5 deg of
loop residual, so the 3.63 deg observed is systematic bias inside each recording - and
averaging biases redistributes them instead of cancelling them. What comes out is a
consistent rig carrying ~1 deg of error spread evenly, not a better one. Removing the bias
needs a recording where three or more cameras see the board at once.

WEIGHTS. Pass a virtual_stereo yaml as argv[3] and each pair is weighted by its MEASURED
epipolar residual (dy_median / focal), which is what its rectification actually delivers.
Without it the weight falls back to the solve's own reported angular spread - and that is
the wrong quantity: internal scatter is near-uniform across the four (0.55-0.68 deg) and
says nothing about bias, so the first run handed the largest correction (1.31 deg) to the
pair with the BEST rectification (right, 0.62 px) and drove it to 2.55 px.

Levenberg-Marquardt, not Gauss-Newton: plain GN diverges here, walking the rig to 70-115
deg per edge while dutifully driving the loop residual to zero. A closed loop is not by
itself evidence of a correct answer, which is why the per-edge corrections are printed.
"""
import sys, numpy as np, yaml

RING = [("front","cam1","cam2"), ("right","cam2","cam4"),
        ("rear","cam4","cam3"),  ("left","cam3","cam1")]
CAMS = ["cam1","cam2","cam4","cam3"]
SIGMA_T = 0.0013                      # observed baseline spread across the four solves

def inv(M):
    A=np.eye(4); A[:3,:3]=M[:3,:3].T; A[:3,3]=-M[:3,:3].T@M[:3,3]; return A
def so3_exp(w):
    th=np.linalg.norm(w)
    if th<1e-14: return np.eye(3)
    k=w/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
def so3_log(R):
    th=np.arccos(np.clip((np.trace(R)-1)/2,-1,1))
    if th<1e-12: return np.zeros(3)
    return th/(2*np.sin(th))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])
def retract(P,d):
    Q=P.copy(); Q[:3,:3]=P[:3,:3]@so3_exp(d[:3]); Q[:3,3]=P[:3,3]+P[:3,:3]@d[3:]; return Q
def loop(T):
    Q=np.eye(4)
    for n,_,_ in RING: Q=Q@inv(T[n])
    return np.degrees(np.arccos(np.clip((Q[:3,:3].trace()-1)/2,-1,1))), 1000*np.linalg.norm(Q[:3,3])

ext = yaml.safe_load(open(sys.argv[1]))
Tm  = {n: np.array(ext[n]["T_to_from"], float) for n,_,_ in RING}
if len(sys.argv) > 3:                       # weight by what each pair actually delivers
    vs = yaml.safe_load(open(sys.argv[3])); f = vs["virtual_pinhole"]["focal_px"]
    sr = {n: vs[n]["quality"]["epipolar_dy_median_px"]/f for n,_,_ in RING}
    print("weighting by measured epipolar residual: %s px"
          % {n: vs[n]["quality"]["epipolar_dy_median_px"] for n,_,_ in RING})
else:
    sr = {n: np.radians(ext[n]["angular_spread_deg"]["median"]) for n,_,_ in RING}

def residual(P):
    r=[]
    for n,a,b in RING:
        e = inv(inv(P[b]) @ P[a]) @ Tm[n]
        r.append(np.concatenate([so3_log(e[:3,:3])/sr[n], e[:3,3]/SIGMA_T]))
    return np.concatenate(r)

P = {"cam1": np.eye(4)}
for n,a,b in RING[:-1]: P[b] = P[a] @ inv(Tm[n])
free, lam = CAMS[1:], 1e-3
print("before closure: %.2f deg, %.1f mm   (weighted cost %.3f)" % (loop(Tm) + (np.linalg.norm(residual(P)),)))
for it in range(60):
    r0 = residual(P); J = np.zeros((len(r0), 6*len(free))); eps = 1e-6
    for i,c in enumerate(free):
        for k in range(6):
            d = np.zeros(6); d[k] = eps
            J[:,6*i+k] = (residual({**P, c: retract(P[c], d)}) - r0)/eps
    H, g = J.T@J, J.T@r0
    for _ in range(30):
        dx = -np.linalg.solve(H + lam*np.diag(np.diag(H)+1e-12), g)
        Pn = {**P}
        for i,c in enumerate(free): Pn[c] = retract(P[c], dx[6*i:6*i+6])
        if np.linalg.norm(residual(Pn)) < np.linalg.norm(r0):
            P, lam = Pn, max(lam*0.3, 1e-12); break
        lam *= 10
    else: break
    if np.linalg.norm(dx) < 1e-13: break
Tc = {n: inv(P[b]) @ P[a] for n,a,b in RING}
print("after  closure: %.4f deg, %.3f mm   (weighted cost %.3f, %d iterations)"
      % (loop(Tc) + (np.linalg.norm(residual(P)), it+1)))
print("\n%-6s %-14s %10s %9s %11s %11s" % ("pair","cams","d_rot deg","d_t mm","base was","base now"))
for n,a,b in RING:
    e = inv(Tm[n]) @ Tc[n]
    print("%-6s %s->%-8s %10.2f %9.2f %11.1f %11.1f" % (n,a,b,
          np.degrees(np.linalg.norm(so3_log(e[:3,:3]))), 1000*np.linalg.norm(e[:3,3]),
          1000*ext[n]["baseline_m"], 1000*np.linalg.norm(Tc[n][:3,3])))

if len(sys.argv) > 2:
    out = {"rig_in_cam1": {c: [[round(float(v),6) for v in row] for row in P[c]] for c in CAMS}}
    for n,a,b in RING:
        e = inv(Tm[n]) @ Tc[n]
        out[n] = {"from": a, "to": b,
                  "simultaneous_poses": ext[n]["simultaneous_poses"],
                  "angular_spread_deg": ext[n]["angular_spread_deg"],
                  "baseline_m": round(float(np.linalg.norm(Tc[n][:3,3])), 4),
                  "baseline_measured_m": ext[n]["baseline_m"],
                  "closure_correction": {"rot_deg": round(float(np.degrees(np.linalg.norm(so3_log(e[:3,:3])))),2),
                                         "t_mm": round(float(1000*np.linalg.norm(e[:3,3])),2)},
                  "T_to_from": [[round(float(v),6) for v in row] for row in Tc[n]],
                  "T_to_from_measured": [[round(float(v),6) for v in row] for row in Tm[n]]}
    yaml.safe_dump(out, open(sys.argv[2],"w"), sort_keys=False, default_flow_style=None, width=100)
    print("\nwrote", sys.argv[2])
