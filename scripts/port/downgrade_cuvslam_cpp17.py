#!/usr/bin/env python3
"""Downgrade cuVSLAM C++17 device-incompatible syntax to C++14 so nvcc 10.2 (TX2,
sm_62) can compile the .cu files. IDEMPOTENT — applied at build time by
scripts/build_cuvslam_tx2gpu.sh; the cuVSLAM submodule stays pinned at v17.0.0.

Transforms (safe in both C++14 and C++17, so applied to all sources):
  1. nested namespace open : `namespace a::b::c {`        -> `namespace a { namespace b { namespace c {`
  2. nested namespace close: `}  // namespace a::b::c`     -> `}}} // namespace a::b::c`  (one } per level)
  3. inline variables      : `inline constexpr ...`        -> `constexpr ...`
       (namespace-scope constexpr already has internal linkage; constexpr fns are implicitly inline)

Usage: downgrade_cuvslam_cpp17.py <cuvslam_src_dir>
"""
import re
import sys
import pathlib

EXTS = {".h", ".hpp", ".hh", ".cuh", ".cu", ".cpp", ".cc", ".cxx"}
OPEN_RE = re.compile(r'^(\s*)namespace\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\s*\{\s*$')
# Closes use a comment convention but with variants: '// namespace a::b',
# '// end namespace a::b', '// end of namespace a::b'. Match any text before
# 'namespace <nested::path>'. (Already-converted '}}  //' lines won't match the
# leading single-} pattern, so this stays idempotent.)
CLOSE_RE = re.compile(r'^(\s*)\}\s*//.*?\bnamespace\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)')
INLINE_RE = re.compile(r'\binline\s+constexpr\b')


def convert(text: str):
    out, changed = [], False
    for line in text.split("\n"):
        mo = OPEN_RE.match(line)
        if mo:
            indent, path = mo.group(1), mo.group(2)
            out.append(indent + " ".join(f"namespace {p} {{" for p in path.split("::")))
            changed = True
            continue
        mc = CLOSE_RE.match(line)
        if mc:
            indent, path = mc.group(1), mc.group(2)
            out.append(indent + "}" * len(path.split("::")) + f"  // namespace {path}")
            changed = True
            continue
        out.append(line)
    new = "\n".join(out)
    new2 = INLINE_RE.sub("constexpr", new)
    return new2, (changed or new2 != new)


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    n = 0
    for p in root.rglob("*"):
        if p.suffix in EXTS and p.is_file() and "/build" not in str(p):
            txt = p.read_text(encoding="utf-8", errors="replace")
            new, ch = convert(txt)
            if ch:
                p.write_text(new, encoding="utf-8")
                n += 1
    print(f"downgrade_cuvslam_cpp17: rewrote {n} files under {root}")


if __name__ == "__main__":
    main()
