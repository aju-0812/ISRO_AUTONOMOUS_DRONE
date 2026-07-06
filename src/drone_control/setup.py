from setuptools import find_packages, setup

package_name = 'drone_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='orinnano',
    maintainer_email='orinnano@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'takeoff_node = drone_control.takeoff_node:main',
            'landing_node = drone_control.landing_node:main',
	    'flight_manager=drone_control.flight_manager:main',
	    'landing_manager=drone_control.landing_manager:main',
	    'navigation_manager=drone_control.navigation_manager:main',
	    'lawnmower=drone_control.lawnmower:main',
	    'image_loop_logger = drone_control.image_loop_logger:main',
	    'flight_ctrl=drone_control.flight_ctrl:main',
	    'tag_detect=drone_control.tag_dect:main',
        ],
    },
)

