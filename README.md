## Stochastic Recursive Gaussian Process Regression

Implements the "Stochastic Recursive Gaussian Process" (SRGP) Regression algorithm from the paper "Recursive Estimation for Sparse Gaussian Process Regression" by Manuel Schuerch, Dario Azzimonti, Alessio Beanvoli and Marco Zaffalon.


## Usage

We provide the code for SRGP and an example to demonstrate the usage.
The main code is in RECC.py and the example in the jupyter notebook example.ipynb.
In order to run the algorithm, you need [GPy](https://github.com/SheffieldML/GPy) since we use their implementation of the kernels.
At the moment, the implemented stationary kernels from GPy are compatible with our implementation and it could be easily extended for instance to sum- and product kernels.


## Contributors

Schuerch, M. and Azzimonti, D. and Benavoli A. and Zaffalon M.

## Reference

```
@paper{SABZ,
  title =  {Recursive {E}stimation for {S}parse {G}aussian {P}rocess {R}egression},
  author =   {Schuerch, M. and Azzimonti, D. and Benavoli A. and Zaffalon M.},
  year =   {2019}
}

```


