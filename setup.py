import sys

from setuptools import find_packages, setup


def get_version(filename):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError("No version found in %r." % filename)
    if version is None:
        raise ValueError(filename)
    return version

install_requires = [
    "gym>=0.17.1",
    "numpy>=1.10.0,<=1.20.0",
    "pyglet",
    # 'pyglet',
    "pyzmq>=16.0.0",
    "opencv-python>=3.4",
    "PyYAML>=3.11",
    f"duckietown-world-daffy",
    "PyGeometry-z6",
    "carnivalmirror==0.6.2",
    "stable-baselines3==1.3.0",
    "zuper-commons-z6",
    "typing_extensions",
    "Pillow",
]

system_version = tuple(sys.version_info)[:3]

if system_version < (3, 7):
    install_requires.append("dataclasses")


setup(
    name=f"sim2sim2real",
    zip_safe=False,
    version="0.0.1",
    keywords="duckietown, environment, agent, rl, openaigym, openai-gym, gym",
    include_package_data=True,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "dt-check-gpu=duckietown.check_hw:main",
        ],
    },
)