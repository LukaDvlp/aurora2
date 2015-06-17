# Overview
#### Aurora: Adaptive Unified Recognition for On-site Rover Autonomy

Aurora is a library for enabling rover autonomy in unknown environments written in Python/C++. It includes several vision algorithms as well as providing interfaces to drive Micro-series rovers.

#### Contributors (as of June 2015)

- Kyohei Otsu <kyon@ac.jaxa.jp>
- Taiki Mashimo <mashimo.taiki@ac.jaxa.jp>

# Modules

Aurora has a modular structure based on functionalities:
* core. The core functionality
* loc. Vision-based localization
* geom. Geometric analysis of terrain
* nongeom. Non-geometric analsys of terrain
* planning. Path planning and command generation
* viz. Visualization
* hw. Drivers for hardwares


# Dependency

- OpenCV (2.4.11 tested)

- Python (2.7.5 tested)
- NumPy
- Scipy (0.15.1 tested)
- Scikit-learn
- Scikit-image

- FLASK (0.10.1 tested)
http://flask.pocoo.org/

- Colorcorrect (0.0.5 tested)
https://pypi.python.org/pypi/colorcorrect
