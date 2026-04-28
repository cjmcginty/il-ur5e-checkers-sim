from setuptools import setup
import os

package_name = 'ur5e_checkers_bringup'


def package_files(source_dirs):
    data_files = []

    for source_dir in source_dirs:
        for path, _, filenames in os.walk(source_dir):
            if not filenames:
                continue

            rel_path = os.path.relpath(path, os.path.join(package_name))
            install_path = os.path.join('share', package_name, rel_path)
            file_paths = [os.path.join(path, f) for f in filenames]
            data_files.append((install_path, file_paths))

    return data_files


data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

data_files += package_files([
    'ur5e_checkers_bringup/launch',
    'ur5e_checkers_bringup/config',
    'ur5e_checkers_bringup/urdf',
    'ur5e_checkers_bringup/worlds',
    'ur5e_checkers_bringup/models',
])

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Natalie Essig',
    maintainer_email='nnessig@wpi.edu',
    description='UR5e checkers simulation bringup (Gazebo Sim + gz_ros2_control)',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'checkers_game_node = ur5e_checkers_bringup.checkers_game_node:main',
            'test_checkers = ur5e_checkers_bringup.test_checkers:main',
            "checkers_piece_manager = ur5e_checkers_bringup.checkers_piece_manager:main",
            'dqn_policy_node = ur5e_checkers_bringup.dqn_policy_node:main',
            "move_target_node = ur5e_checkers_bringup.move_target_node:main",
            "player_move_helper_node = ur5e_checkers_bringup.player_move_helper_node:main",
            "magic_piece_mover_node = ur5e_checkers_bringup.magic_piece_mover_node:main",
        ],
    },
)