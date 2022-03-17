#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name="motlib",
    version="0.1",
    author="3dlg-hcvc",
    url="https://github.com/3dlg-hcvc/2DMotion",
    description="Code for 2D motion prediction project",
    packages=find_packages(exclude=("configs", "tests")),
    dependency_links=[
        "https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html",
        "https://download.pytorch.org/whl/torch_stable.html"
    ],
    install_requires=[
        "detectron2==0.3",
        "opencv-python==4.5.1.48",
        "torch==1.7.1+cu110",
        "torchvision==0.8.2+cu110",
    ],
)
