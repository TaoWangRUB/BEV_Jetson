#!/usr/bin/env python3
"""Downgrade cuVSLAM C++17 device-incompatible syntax to C++14 so nvcc 10.2 (TX2,
sm_62) can compile the .cu files. IDEMPOTENT — applied at build time by
scripts/build_cuvslam_tx2gpu.sh; the cuVSLAM submodule stays pinned at v17.0.0.

Transforms (safe in both C++14 and C++17, so applied to all sources):
  1. nested namespace open : `namespace a::b::c {`        -> `namespace a { namespace b { namespace c {`
  2. nested namespace close: the MATCHING brace, found by brace counting (not by
     the trailing comment -- a bare `}` closes one such namespace in v17)
  3. inline variables      : `inline constexpr ...`        -> `constexpr ...`
       (namespace-scope constexpr already has internal linkage; constexpr fns are implicitly inline)

Usage: downgrade_cuvslam_cpp17.py <cuvslam_src_dir>
"""
import re
import sys
import pathlib

EXTS = {".h", ".hpp", ".hh", ".cuh", ".cu", ".cpp", ".cc", ".cxx"}
OPEN_RE = re.compile(r'^(\s*)namespace\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)+)\s*\{\s*$')
# Closes are found by BRACE MATCHING, not by their trailing comment. Matching on
# '// namespace a::b' silently misses a bare '}' -- cuVSLAM v17's
# libs/sof/sof_mono_interface.h closes a forward-declaration namespace with one --
# which leaves the namespace open, so everything after it nests one level deeper
# ('cuvslam::cuvslam::sof') and the file fails to compile far from the real cause.
INLINE_RE = re.compile(r'\binline\s+constexpr\b')


def _strip_literals(line: str, in_block: bool):
    """Blank out //-comments, /*..*/ comments and string/char literals so brace
    counting sees only real code. Returns (code_only_line, still_in_block)."""
    out, i, n = [], 0, len(line)
    while i < n:
        c = line[i]
        if in_block:
            if c == "*" and i + 1 < n and line[i + 1] == "/":
                in_block = False
                i += 2
            else:
                i += 1
            continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break
        if c == "/" and i + 1 < n and line[i + 1] == "*":
            in_block = True
            i += 2
            continue
        if c in "\"'":
            quote, i = c, i + 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out), in_block


def convert(text: str):
    lines = text.split("\n")
    extra_closes = {}        # line index -> how many '}' to append
    pending = []             # (depth_before_open, extra) for each converted open
    depth, in_block, changed = 0, False, False

    for idx, line in enumerate(lines):
        mo = OPEN_RE.match(line)
        if mo:
            indent, path = mo.group(1), mo.group(2)
            parts = path.split("::")
            lines[idx] = indent + " ".join(f"namespace {p} {{" for p in parts)
            pending.append((depth, len(parts) - 1))
            depth += 1       # the source line contributed exactly one level
            changed = True
            continue
        code, in_block = _strip_literals(line, in_block)
        for c in code:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                # Closing back to the level an converted open started at: this is
                # its matching brace, so the extra levels close here too.
                if pending and depth == pending[-1][0]:
                    _, extra = pending.pop()
                    extra_closes[idx] = extra_closes.get(idx, 0) + extra

    if pending:
        raise SystemExit(
            "downgrade_cuvslam_cpp17: unbalanced namespace braces; "
            f"{len(pending)} nested namespace(s) never closed"
        )
    for idx, extra in extra_closes.items():
        stripped = lines[idx].rstrip()
        # Append after the closing brace, before any trailing comment.
        pos = stripped.rfind("}")
        lines[idx] = stripped[: pos + 1] + "}" * extra + stripped[pos + 1 :]

    new = "\n".join(lines)
    new2 = INLINE_RE.sub("constexpr", new)
    if changed or extra_closes:
        # A conversion must never change the file's overall brace balance.
        if (text.count("{") - text.count("}")) != (new2.count("{") - new2.count("}")):
            raise SystemExit("downgrade_cuvslam_cpp17: conversion changed brace balance")
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
