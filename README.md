# LSTM-Scratch
By Antony Sagayaraj

This is a Python 3 implementation of the Long-Short Term Memory (LSTM) network by Hochreiter and Schmidhuber 1997 (http://www.bioinf.jku.at/publications/older/2604.pdf). The only required package is Numpy

LSTM_Cell.py contains the class for a single LSTM cell and the data feeder

Rebber_Grammar.py contains the functions used to generate embedded Rebber Grammar, a pseudo-random sequence used in the paper as a toy problem to test the LSTM network’s functionality.

Neural_Network.py uses the previous two scripts to generate the training, validation, and test data files, and trains an LSTM network of predefined size