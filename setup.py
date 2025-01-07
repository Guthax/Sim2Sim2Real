import sys

from setuptools import find_packages, setup
import platform


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


CUDA = False


torch_install_urls = {
    "TORCH_WIN_CUDA": "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp37-cp37m-win_amd64.whl",
    "TORCH_WIN_CPU": "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp37-cp37m-win_amd64.whl",
    "TORCH_LINUX_CUDA": "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp37-cp37m-linux_x86_64.whl",
    "TORCH_LINUX_CPU": "https://download.pytorch.org/whl/cpu/torch-1.13.1%2Bcpu-cp37-cp37m-linux_x86_64.whl",
    "TORCHVIS_WIN_CUDA": "https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp37-cp37m-win_amd64.whl",
    "TORCHVIS_WIN_CPU": "https://download.pytorch.org/whl/cpu/torchvision-0.14.1%2Bcpu-cp37-cp37m-win_amd64.whl",
    "TORCHVIS_LINUX_CUDA": "https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp37-cp37m-linux_x86_64.whl",
    "TORCHVIS_LINUX_CPU": "https://download.pytorch.org/whl/cpu/torchvision-0.14.1%2Bcpu-cp37-cp37m-linux_x86_64.whl"
}
torch_url = ""
torchvis_url = ""
if platform.system() == "Windows":
    torch_url = torch_install_urls["TORCH_WIN_CUDA"] if CUDA else torch_install_urls["TORCH_WIN_CPU"]
    torchvis_url = torch_install_urls["TORCHVIS_WIN_CUDA"] if CUDA else torch_install_urls["TORCHVIS_WIN_CPU"]
else:
    torch_url = torch_install_urls["TORCH_LINUX_CUDA"] if CUDA else torch_install_urls["TORCH_LINUX_CPU"]
    torchvis_url = torch_install_urls["TORCHVIS_LINUX_CUDA"] if CUDA else torch_install_urls["TORCHVIS_LINUX_CPU"]
print(torch_url, torchvis_url)
install_requires = [
    "gym>=0.17.1",
    "numpy>=1.10.0",
    "pyglet<=1.5.0",
    "pygame==2.1.2",
    "pyzmq>=16.0.0",
    "opencv-python==4.3.0.36",
    "PyYAML>=3.11",
    f"duckietown-world-daffy",
    "PyGeometry-z6",
    "carnivalmirror==0.6.2",
    "stable-baselines3==1.3.0",
    "zuper-commons-z6",
    "typing_extensions",
    "Pillow",
    "pipdeptree",
    "typing-extensions",
    f"torch@{torch_url}",
    f"torchvision@{torchvis_url}",
    "torchsummary==1.5.1",
    "carla==0.9.13",
    "tensorboard",
    "chardet"
]

system_version = tuple(sys.version_info)[:3]

if system_version < (3, 7):
    install_requires.append("dataclasses")


setup(
    name=f"sim2sim2real",
    zip_safe=False,
    version="0.0.1",
    keywords="duckietown, environment, agent, rl, openaigym, openai-gym, gym",
    packages=find_packages(include=["envs*", "simulators*"]),  # Only include these
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=[
        'https://download.pytorch.org/whl/cu116'
    ],
    entry_points={
        "console_scripts": [
            "dt-check-gpu=duckietown.check_hw:main",
        ],
    },
)
