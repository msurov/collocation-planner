from setuptools import find_packages, setup


def get_version(path_to_version: str):
    with open(path_to_version, "r") as version:
        return version.readline()


setup(
    name="collocation-demo",
    version=get_version("VERSION"),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "tk",
        "scipy",
        "wheel",
        "casadi",
        "svgpathtools"
    ],
    packages=find_packages(where="."),
    package_dir={"": "."},
    entry_points={
        "console_scripts": [
            "run_trajectory_trcking_test=trajectory_tracking.run_sim:main",
            "gen_sample_trajectories=routes.gen_sample_trajectories:main",
            "make_trajectories_from_route=routes.make_trajectories_from_route:main",
            "make_trajectory_from_waypoints=routes.make_trajectory_from_waypoints:main",
            "show_trajectory=routes.show_trajectory:main"
        ]
    },
    author="Maksim Surov",
    author_email="maxim.surov@evocargo.com",
    description="Control system simulator",
)
