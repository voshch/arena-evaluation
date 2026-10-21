import os

from setuptools import find_packages, setup

package_name = 'arena_evaluation'


def _walk_data_files(*roots):
    # os.path.isfile filters out dangling symlinks. colcon --symlink-install
    # populates the build dir with per-file symlinks into source and does not
    # prune them when source files are deleted, so os.walk would otherwise
    # hand setuptools broken symlinks and the copy step would abort.
    for root in roots:
        for base, _dirs, files in os.walk(root):
            kept = [os.path.join(base, f) for f in files if os.path.isfile(os.path.join(base, f))]
            if kept:
                yield (os.path.join('share', package_name, base), kept)


setup(
    name=package_name,
    packages=find_packages(where='.', include=[f'{package_name}*']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *_walk_data_files('config', 'configs'),
    ],
    zip_safe=True,
    maintainer='NelsonHecker',
    maintainer_email='heckernelson@gmail.com',
    description='Record, evaluate, and plot navigational metrics to evaluate ROS navigation planners',
    license='BSD',
    entry_points={
        'console_scripts': [
            'record = arena_evaluation.ingestion.recorder:main',
            'evaluation = arena_evaluation.cli:main',
            'benchmark = arena_evaluation.benchmark.runner:cli_main',
            'evaluation_cli = arena_evaluation.benchmark.cli:main',
        ],
    },
    package_data={
        'arena_evaluation.processing.acoustics': ['solver.cpp'],
        'arena_evaluation.presentation': ['*.j2', '*.html.j2'],
    },
)
