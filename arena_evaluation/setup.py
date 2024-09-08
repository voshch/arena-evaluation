from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'arena_evaluation'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')), # Include non-Python files (like launch files, configuration files, or other resources) in the package's installation
                                                                                # Include the configuration file in the install directory
    ],                                                                          
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NamTruongTran',
    maintainer_email='trannamtruong98@gmail.com',
    description='Record, evaluate, and plot navigational metrics to evaluate ROS navigation planners',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'record = arena_evaluation.data_recorder_node:main',
        'metrics = arena_evaluation.get_metrics:main',
        ],
    },
)
