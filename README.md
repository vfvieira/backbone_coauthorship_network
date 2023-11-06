# Backbone of coauthorship networks

## Introduction
This project contains a Python implementation of the experiments performed for the manuscript entitled "A Network-Driven Study of Hyperprolific Authors in Computer Science", by Vinícius da F. Vieira, Carlos H. G. Ferreira, Jussara M. Almeida, Edré Moreira, Alberto H. F. Laender, Wagner Meira Jr. and Marcos A. Gonçalves, still under review to be considered for publication in Scientometrics.

## Dependencies
The code has been executed using Python 3.10.12 and the following libraries:
- networkx (version 2.8.8)
- numpy (version 1.21.5)
- scipy (versin 1.9.3)
- rbo (version 0.1.2)
- matplotlib (version 3.6.2)

All libraries has been installed using pip, with the following command:
- pip3 install <library>

## Execution
The code is executed with the following command:
- python3 dblp_backbone.py

## Organization of the code
The code is organized in the following python scripts:
- dblp_backbone.py: Main script that organizes the execution flow and calls the other scripts in order to perform the methodology as described in the manuscript.
- dblp_aux.py: Auxiliary functions such as data input and output and pre-processing steps.
- BackboneExtractor.py: Backbone extraction, devbeloped by Coscia, Michele, and Frank MH Neffke. "Network backboning with noisy data." 2017 IEEE 33rd International Conference on Data Engineering (ICDE). IEEE, 2017
- dblp_backbone_analysis.py: The analysis of the backbone of the network and the probability of presence (essentially Section 4.2).
- dblp_correlation.py: The analysis of the static and dynamic correlation as described in the manuscript (essentially Section 4.3).
- dblp_generate_rank.py: Implementation of the hyperprolific rank generation.
- dblp_network_analysis.py: Analysis of the network topology that allows the investigation of the correlations (Section 4.3) and other network analysis (Section 4.1).
- dblp_plot.py: Implementation of functions for plotting the results.

## General flow
The main script (dblp_backbone.py) reads the input data and executes the methodology for the network creation and backbone extraction, as described in the manuscript. Then it performs the experiments following the organization described in the manuscript. Some information are displayed in the standard output. The figures used in the manuscript are stored in the 'results' folder.

## Parameterization
