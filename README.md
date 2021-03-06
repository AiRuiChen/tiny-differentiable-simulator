# Tiny Differentiable Simulator

Tiny Differentiable Simulator is a header-only C++ physics library with zero dependencies.

It currently implements various rigid-body dynamics algorithms, including forward and inverse dynamics, as well as contact models based on impulse-level LCP and force-based nonlinear spring-dampers. Actuator models for motors, servos, and Series-Elastic Actuator (SEA) dynamics are implemented.

The entire codebase is templatized so you can use forward- and reverse-mode automatic differentiation
scalar types, such as CppAD, Stan Math fvar and ceres::Jet. The library can also be used with
regular float or double precision values. Another option is to use the included
fix-point integer math, that provide cross-platform deterministic computation.

Multiple visualizers are available, see below.

![](https://github.com/erwincoumans/tiny-differentiable-simulator/blob/master/data/trb_meshcat.jpg)

## Related Papers
* "NeuralSim: Augmenting Differentiable Simulators with Neural Networks", Eric Heiden, David Millard, Erwin Coumans, Yizhou Sheng, Gaurav S. Sukhatme. [PDF on Arxiv](https://arxiv.org/abs/2011.04217)
* “Augmenting Differentiable Simulators with Neural Networks to Close the Sim2Real Gap”, RSS 2020 sim-to-real workshop, Eric Heiden, David Millard, Erwin Coumans, Gaurav Sukhatme. [PDF on Arxiv](https://sim2real.github.io/assets/papers/2020/heiden.pdf) and [video](https://www.youtube.com/watch?v=awhkI5xtFa0)
* "Interactive Differentiable Simulation", 2020, Eric Heiden, David Millard, Hejia Zhang, Gaurav S. Sukhatme. [PDF on Arxiv](https://arxiv.org/abs/1905.10706)

## Getting started
The open-source version builds using CMake and requires a compiler with C++17 support.

```
mkdir build
cd build
cmake ..
make -j
```

## Examples

For visualization, two options are supported:

### OpenGL 3+ Visualization

* tiny_opengl3_app, an OpenGL3 visualizer

This visualizer is native part of this library under src/visualizer/opengl

### MeshCat Visualization

* [MeshCat](https://github.com/rdeits/meshcat), a web-based visualizer that uses WebGL

A C++ ZMQ interface is provided.

Before running the example, install python, pip and meshcat, run the meshcat-server 
and open the web browser (Chrome is recommended for a good three.js experience.)
```
pip install meshcat
meshcat-server --open
This should open Chrome at http://localhost:7000/static/
Then compile and run tiny_urdf_parser_meshcat_example in optimized/release build.
```

URDF files can be loaded using a provided parser based on TinyXML2.

All dependencies for meshcat visualization are included in third_party.

---

*Disclaimer: This is not an official Google product.*

