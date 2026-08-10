"""Repair LightGBM's OpenMP linkage on macOS without Homebrew.

Why this exists
---------------
The LightGBM macOS wheel links against ``@rpath/libomp.dylib`` but bakes in only two
rpath entries: ``/opt/homebrew/opt/libomp/lib`` and ``/opt/local/lib/libomp``. Both are
package-manager locations. On a machine without Homebrew or MacPorts, neither exists, so
``import lightgbm`` dies with::

    OSError: dlopen(... lib_lightgbm.dylib): Library not loaded: @rpath/libomp.dylib

The usual advice is ``brew install libomp``, which is not an option when Homebrew is not
installed and we do not want to require it.

What this does
--------------
scikit-learn and PyTorch already ship their own ``libomp.dylib`` inside the virtualenv. We
reuse one of those copies:

1. Copy ``libomp.dylib`` next to ``lib_lightgbm.dylib``.
2. Set that copy's install name to ``@rpath/libomp.dylib`` so it satisfies the dependency.
3. Add ``@loader_path`` as an rpath on ``lib_lightgbm.dylib`` so dyld looks beside itself.
4. Re-sign both (ad-hoc); editing a Mach-O invalidates its signature, and arm64 macOS
   refuses to load unsigned dylibs.

The operation is idempotent and confined to the virtualenv, which is disposable. Re-run it
after any ``uv sync`` that reinstalls LightGBM. Non-macOS platforms are a no-op.

Note on OpenMP duplication: loading two OpenMP runtimes into one process can be unsafe.
Here we deliberately reuse the *same* libomp image that scikit-learn already loads rather
than introducing a second one, which is why we copy from site-packages rather than
downloading an independent build.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


class LibompRepairError(RuntimeError):
    """Raised when the repair cannot be completed."""


def _site_packages() -> Path:
    path = sysconfig.get_paths().get("purelib")
    if path is None:  # pragma: no cover - defensive
        raise LibompRepairError("cannot locate site-packages")
    return Path(path)


def find_donor_libomp(site_packages: Path) -> Path | None:
    """Find a libomp.dylib already vendored by another installed package."""
    candidates = [
        site_packages / "sklearn" / ".dylibs" / "libomp.dylib",
        site_packages / "torch" / "lib" / "libomp.dylib",
    ]
    candidates.extend(sorted(site_packages.glob("*/.dylibs/libomp.dylib")))
    return next((c for c in candidates if c.exists()), None)


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    # install_name_tool warns that it invalidates the signature; we re-sign afterwards.
    if result.returncode != 0 and "invalidate the code signature" not in result.stderr:
        raise LibompRepairError(f"{' '.join(cmd[:2])} failed: {result.stderr.strip()}")


def _has_loader_path_rpath(lib: Path) -> bool:
    out = subprocess.run(["otool", "-l", str(lib)], capture_output=True, text=True).stdout
    return "@loader_path" in out


def lightgbm_imports() -> bool:
    """Probe LightGBM in a subprocess so a failed dlopen cannot poison this process."""
    probe = subprocess.run(
        [sys.executable, "-c", "import lightgbm"], capture_output=True, text=True
    )
    return probe.returncode == 0


def repair(verbose: bool = True) -> bool:
    """Repair LightGBM's libomp linkage. Returns True if LightGBM imports afterwards."""

    def say(msg: str) -> None:
        if verbose:
            print(msg)

    if platform.system() != "Darwin":
        say("Not macOS; no repair needed.")
        return lightgbm_imports()

    if lightgbm_imports():
        say("LightGBM already imports cleanly; nothing to do.")
        return True

    site_packages = _site_packages()
    lgb_lib_dir = site_packages / "lightgbm" / "lib"
    lgb_dylib = lgb_lib_dir / "lib_lightgbm.dylib"
    if not lgb_dylib.exists():
        raise LibompRepairError(
            f"lib_lightgbm.dylib not found at {lgb_dylib}. Is the 'ml' extra installed?"
        )

    donor = find_donor_libomp(site_packages)
    if donor is None:
        raise LibompRepairError(
            "No vendored libomp.dylib found in site-packages (checked scikit-learn and "
            "torch). Install the 'ml' extra, or install libomp system-wide."
        )
    say(f"Using donor libomp: {donor}")

    target = lgb_lib_dir / "libomp.dylib"
    shutil.copy2(donor, target)
    _run(["install_name_tool", "-id", "@rpath/libomp.dylib", str(target)])

    if not _has_loader_path_rpath(lgb_dylib):
        _run(["install_name_tool", "-add_rpath", "@loader_path", str(lgb_dylib)])

    # Editing a Mach-O invalidates its signature; arm64 macOS will not load it unsigned.
    for lib in (target, lgb_dylib):
        _run(["codesign", "-f", "-s", "-", str(lib)])

    ok = lightgbm_imports()
    say("LightGBM imports successfully." if ok else "Repair applied but LightGBM still fails.")
    return ok


def main() -> int:
    try:
        return 0 if repair() else 1
    except LibompRepairError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
