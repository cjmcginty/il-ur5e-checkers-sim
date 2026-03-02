from setuptools import setup
from glob import glob
import os

package_name = 'ur5e_checkers_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('ur5e_checkers_bringup/launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('ur5e_checkers_bringup/config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('ur5e_checkers_bringup/urdf/*')),
        (os.path.join('share', package_name, 'worlds'), glob('ur5e_checkers_bringup/worlds/*')),
        (os.path.join('share', package_name, 'models', 'checkers_board'),
         glob('ur5e_checkers_bringup/models/checkers_board/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nat',
    maintainer_email='nat@example.com',
    description='UR5e checkers simulation bringup (Gazebo Sim + gz_ros2_control)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
