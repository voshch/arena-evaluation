import os

from setuptools import setup

package_name = 'arena_evaluation_mcp'

setup(
    name=package_name,
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
    ],
    zip_safe=True,
    author='NelsonHecker',
    author_email='heckernelson@gmail.com',
    maintainer='NelsonHecker',
    maintainer_email='heckernelson@gmail.com',
    description='MCP server giving agents full control over the Arena evaluation pipeline: '
                'discover, configure, run benchmarks, process metrics, create report manifests, '
                'analyze results, and inject insights into reports.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'arena_evaluation_mcp = arena_evaluation_mcp.server:main',
        ],
    },
)
