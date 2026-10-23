#!/usr/bin/env python3
"""Cross-compile the production NEON transpose, test in QEMU, and model with MCA.

Requires Python 3, the aarch64-unknown-linux-musl Rust target, qemu-aarch64,
and LLVM's llvm-mca. Uses cached Cargo dependencies without network access.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TARGET = "aarch64-unknown-linux-musl"
CPUS = ("cortex-a55", "neoverse-n1", "neoverse-n2")


def extract_region(assembly, variant, region):
    match = re.search(
        rf"^([^\s:]*{variant}[^\s:]*probe_{region}[^\s:]*):\n(.*?)^\.Lfunc_end",
        assembly,
        re.M | re.S,
    )
    if not match:
        raise RuntimeError(f"Missing assembly for {variant}/{region}")
    body = match[2].split("\tret\n", 1)[0]
    if region == "fft64":
        # Analyze the successful straight-line FFT body, including all spills.
        # Exclude length checks, panic paths and the callee-save pro/epilogue.
        body = body[body.rindex("\tb.ne"):].split("\n", 1)[1]
        body = body[:body.rindex("\tadd\tsp, sp, #")]
    lines = [line for line in body.splitlines()
             if line.startswith("\t") and not line.startswith("\t.")]
    body = "\n".join(lines) + "\n"
    if re.search(r"^\s*(?:b(?:\.\w+)?|bl|cbz|cbnz|tbz|tbnz|ret)\s", body, re.M):
        raise RuntimeError("Region is no longer straight-line; review extraction")
    # These are immutable input/stack bases. All other inputs are defined by
    # loads or constant setup within the region, so iterations do not inherit
    # vector results from the previous iteration.
    for line in lines:
        if re.match(r"\s*(?!str\b|stur\b|stp\b)\w+\s+(?:[xw][012]|sp|[xw]29),", line):
            raise RuntimeError(f"Input base overwritten; review dependencies: {line}")
        if re.search(r"\[(?:x[012]|sp|x29)[^\]]*\](?:!|,)", line):
            raise RuntimeError(f"Input base uses writeback: {line}")
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llvm-mca", default="llvm-mca")
    parser.add_argument("--qemu", default="qemu-aarch64")
    parser.add_argument("--out-dir", type=Path, default=REPO / "target/neon-transpose")
    parser.add_argument("--check-only", action="store_true",
                        help="Cross-compile and run correctness checks without MCA")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    src = out / "src"
    results = out / "results"
    src.mkdir(parents=True, exist_ok=True)
    results.mkdir(exist_ok=True)

    # Match the repository's resolved dependency, rather than silently modeling
    # a different fearless_simd release in the temporary probe crate.
    versions = re.findall(r'name = "fearless_simd"\nversion = "([^"]+)"',
                          (REPO / "Cargo.lock").read_text())
    if len(versions) != 1:
        raise RuntimeError("Expected one fearless_simd version in Cargo.lock")
    simd_version = versions[0]
    (out / "Cargo.toml").write_text(f'''[package]
name = "phastft-neon-probe"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
fearless_simd = "={simd_version}"
num-traits = "0.2.19"
phastft = {{ path = {json.dumps(str(REPO))}, default-features = false }}
''')
    source = (REPO / "src/kernels/small.rs").read_text()
    probes = (HERE / "probes.rs").read_text()
    dispatch = "return transpose8_neon(neon, input)"
    if source.count(dispatch) != 1:
        raise RuntimeError("NEON dispatch changed; review generic-baseline substitution")
    for variant, implementation in (("generic", "transpose8_generic_neon"),
                                    ("dedicated", "transpose8_neon")):
        # Only the transpose differs, including when inlined into the FFT64
        # probe. Both variants are otherwise the current production source.
        body = source.replace(dispatch, f"return {implementation}(neon, input)")
        (src / f"{variant}.rs").write_text(
            body + "\n" + probes.replace("TRANSPOSE_IMPL", implementation)
        )
    shutil.copyfile(HERE / "harness.rs", src / "main.rs")

    rust_version = subprocess.check_output(["rustc", "-Vv"], text=True)
    host = re.search(r"^host: (.+)$", rust_version, re.M)[1]
    sysroot = Path(subprocess.check_output(["rustc", "--print", "sysroot"], text=True).strip())
    linker = sysroot / f"lib/rustlib/{host}/bin/rust-lld"
    env = dict(os.environ, CARGO_TARGET_DIR=str(out / "target"),
               CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=str(linker),
               RUSTFLAGS="-C target-cpu=generic")
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    subprocess.run([
        "cargo", "rustc", "--manifest-path", str(out / "Cargo.toml"), "--offline",
        "--release", "--target", TARGET, "--", "--emit=asm,link",
    ], env=env, check=True)
    release = out / "target" / TARGET / "release"
    subprocess.run([args.qemu, str(release / "phastft-neon-probe")], check=True)
    if args.check_only:
        return
    assembly_path = max((release / "deps").glob("phastft_neon_probe-*.s"),
                        key=lambda path: path.stat().st_mtime)
    assembly = assembly_path.read_text()
    shutil.copyfile(assembly_path, results / "probes.s")
    mca_version = subprocess.check_output([args.llvm_mca, "--version"], text=True)
    (results / "versions.txt").write_text(rust_version + "\n" + mca_version
                                         + f"\nfearless_simd {simd_version}\n")
    summaries = []
    for variant in ("generic", "dedicated"):
        for region in ("transpose", "fft64"):
            body = extract_region(assembly, variant, region)
            asm_path = results / f"{variant}-{region}.s"
            asm_path.write_text(body)
            # noalias=false is a deliberately pessimistic sensitivity check
            # for FFT scratch dependencies; MCA does not know actual addresses.
            for noalias in ((True, False) if region == "fft64" else (True,)):
                for cpu in CPUS:
                    for iterations in (1000, 10000):
                        run = subprocess.run([
                            args.llvm_mca, "-mtriple=aarch64-unknown-linux-gnu",
                            f"-mcpu={cpu}", f"-iterations={iterations}",
                            f"-noalias={str(noalias).lower()}", str(asm_path),
                        ], text=True, capture_output=True, check=True)
                        name = f"{variant}-{region}-{cpu}-{noalias}-{iterations}"
                        (results / f"{name}.txt").write_text(run.stdout + run.stderr)
                        row = dict(variant=variant, region=region, cpu=cpu,
                                   noalias=noalias, iterations=iterations)
                        for label in ("Instructions", "Total Cycles", "Total uOps",
                                      "Dispatch Width", "Block RThroughput"):
                            row[label] = float(re.search(
                                rf"^{label}:\s*([\d.]+)", run.stdout, re.M)[1])
                        row["cycles_per_iteration"] = row["Total Cycles"] / iterations
                        summaries.append(row)
                        if iterations == 10000:
                            print(f"{variant:9} {region:9} {cpu:12} noalias={noalias!s:5} "
                                  f"{row['cycles_per_iteration']:.3f} cycles", flush=True)
    (results / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Assembly, complete resource reports and summaries: {results}")


if __name__ == "__main__":
    main()
