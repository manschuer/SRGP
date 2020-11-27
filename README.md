## Stochastic Recursive Gaussian Process Regression

Implements the "Stochastic Recursive Gaussian Process" (SRGP) Regression algorithm from the paper "Recursive Estimation for Sparse Gaussian Process Regression" by Manuel Schuerch, Dario Azzimonti, Alessio Benavoli and Marco Zaffalon.


## Usage

We provide the code for SRGP and an example to demonstrate the usage.
The main code is in RECC.py and the example in the jupyter notebook example.ipynb.
In order to run the algorithm, you need [GPy](https://github.com/SheffieldML/GPy) (tested up to version 1.9.8) since we use their implementation of the kernels.
At the moment, the implemented stationary kernels from GPy are compatible with our implementation and it could be easily extended for instance to sum- and product kernels.


## Contributors

Schuerch, M. and Azzimonti, D. and Benavoli A. and Zaffalon M.

## Reference

```
@article{schurch2020recursive,
  title={Recursive estimation for sparse Gaussian process regression},
  author={Sch{\"u}rch, Manuel and Azzimonti, Dario and Benavoli, Alessio and Zaffalon, Marco},
  journal={Automatica},
  volume={120},
  pages={109127},
  year={2020},
  publisher={Elsevier}
}
```


