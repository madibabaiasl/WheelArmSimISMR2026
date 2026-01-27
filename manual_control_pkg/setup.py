from setuptools import find_packages, setup

package_name = 'manual_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pynput', 'tf_transformations', 'ikpy'],
    zip_safe=True,
    maintainer='group2',
    maintainer_email='group2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'manual_control_node = manual_control_pkg.manual_control_node:main',
        'inverse_kinematics_node = manual_control_pkg.ik_node:main',
        'data_processing_node = manual_control_pkg.data_processing:main',
        'ikpy_node = manual_control_pkg.ikpy:main',
        ],
    },
)
