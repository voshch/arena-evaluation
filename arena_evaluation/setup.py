"""Package metadata and the C++17 solver build.

The two ctypes solvers (`processing/acoustics/solver.cpp`, `processing/path/solver.cpp`)
are not CPython extensions, but they are compiled artefacts the package needs before it
can compute an acoustic metric or a route. `arena build` installs this package through
`setup.py develop`, so the `build_ext` command below is the hook that compiles them, and
`CMakeLists.txt` is the only place their flags and output paths live.
"""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext

package_name = "arena_evaluation"
package_root = Path(__file__).resolve().parent

# Directories below the Python package that hold a `solver.cpp` the package loads
# through ctypes as `solver.so` next to the importing module.
solver_modules = ("processing/acoustics", "processing/path")
solver_file_name = "solver.so"
solver_cxx_standard = "17"
solver_build_hint = (
    "install cmake and a C++17 compiler (`apt install cmake g++`) or build the library "
    "with `cmake -S arena_evaluation -B build -DARENA_SOLVERS_ONLY=ON && cmake --build build`"
)


def _walk_data_files(*roots):
    # os.path.isfile filters out dangling symlinks. colcon --symlink-install
    # populates the build dir with per-file symlinks into source and does not
    # prune them when source files are deleted, so os.walk would otherwise
    # hand setuptools broken symlinks and the copy step would abort.
    for root in roots:
        for base, _dirs, files in os.walk(root):
            kept = [
                os.path.join(base, f)
                for f in files
                if os.path.isfile(os.path.join(base, f))
            ]
            if kept:
                yield (os.path.join("share", package_name, base), kept)


def _solver_path(relative_module):
    """Where the loader of `relative_module` looks for the built solver library."""
    return package_root / package_name / relative_module / solver_file_name


def _build_solvers(build_temp):
    """Compile both solver libraries through CMakeLists.txt, in its ARENA_SOLVERS_ONLY mode."""
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit(
            f"arena_evaluation: building the C++{solver_cxx_standard} solver libraries needs cmake: {solver_build_hint}"
        )
    cxx = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
    if cxx is None:
        raise SystemExit(
            f"arena_evaluation: building the C++{solver_cxx_standard} solver libraries needs a C++ compiler: {solver_build_hint}"
        )
    cmake_build_dir = Path(build_temp) / "arena_solvers"
    configure = [
        cmake,
        "-S",
        str(package_root),
        "-B",
        str(cmake_build_dir),
        "-DARENA_SOLVERS_ONLY=ON",
        f"-DARENA_SOLVER_OUTPUT_DIR={package_root / package_name}",
        f"-DCMAKE_CXX_STANDARD={solver_cxx_standard}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={cxx}",
    ]
    subprocess.run(configure, check=True)
    subprocess.run([cmake, "--build", str(cmake_build_dir), "--parallel"], check=True)
    missing = [
        str(path)
        for path in (_solver_path(module) for module in solver_modules)
        if not path.is_file()
    ]
    if missing:
        raise SystemExit(
            f"arena_evaluation: the solver build produced no {solver_file_name}: {', '.join(missing)}"
        )


class SolverBuildExt(build_ext):
    """Build the ctypes solver libraries for `setup.py develop` (what `arena build` runs)."""

    def run(self):
        _build_solvers(self.build_temp)
        if not self.inplace:
            self._copy_into_build_lib()

    def _copy_into_build_lib(self):
        """Place the libraries in the build tree an install copies from."""
        for module in solver_modules:
            destination = Path(self.build_lib) / package_name / module
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_solver_path(module), destination / solver_file_name)


class SolverBuild(build):
    """Build the ctypes solver libraries before `setup.py build` and `setup.py install`."""

    def run(self):
        self.run_command("build_ext")
        super().run()


setup(
    name=package_name,
    packages=find_packages(where=".", include=[f"{package_name}*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        *_walk_data_files("config", "configs"),
    ],
    zip_safe=False,
    maintainer="NelsonHecker",
    maintainer_email="heckernelson@gmail.com",
    description="Record, evaluate, and plot navigational metrics to evaluate ROS navigation planners",
    license="BSD",
    entry_points={
        "console_scripts": [
            "record = arena_evaluation.ingestion.recorder:main",
            "evaluation = arena_evaluation.cli:main",
            "benchmark = arena_evaluation.benchmark.runner:cli_main",
            "evaluation_cli = arena_evaluation.benchmark.cli:main",
        ],
        "arena_mcp.domains": [
            "evaluation = arena_evaluation.mcp:domain",
        ],
    },
    cmdclass={"build_ext": SolverBuildExt, "build": SolverBuild},
    package_data={
        "arena_evaluation.processing.acoustics": ["solver.cpp"],
        "arena_evaluation.processing.path": ["solver.cpp"],
        "arena_evaluation.presentation": ["*.j2", "*.html.j2"],
        "arena_evaluation.benchmark": ["dashboard.html"],
    },
)
