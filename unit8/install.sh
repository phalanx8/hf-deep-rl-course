#!/bin/bash
apt update
apt install -y python-opengl ffmpeg xvfb swig cmake git-lfs
pip insatll testresources
pip install --upgrade pip
pip install setuptools==65.5.0
