.. ITMAL Project documentation master file, created by
   sphinx-quickstart on Thu Apr 22 19:03:10 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ITMAL Project's documentation!
=========================================
The objective of the second assignment in the fifth lesson is to get started on the final project for the course. The final project is load-bearing element which will be worked on parallel to the topics learned in the lessons. The main objective of the final project is to get a better understanding of the principles of Machine Learning, while using the knowledge gained from the assignments executed in the individual lessons, when determining an appropriate algorithm and executing it on the final project data set. The final project data set is based on a machine learning problem in which the individual group discovers with their own research and analysis and there are no restrictions on the selected final project data set. However, it is recommended in discovering an already existing machine learning problem which has an understandable description of the data set, a method of collecting data and a problem definition.

Acoustic signal processing will be the focus of Group 28 when implementing a machine learning algorithm on the final project. The data set chosen by group 28 is a sound data set for malfunctioning industrial machine investigation and inspection (MIMII) - found here https://zenodo.org/record/3384388. The MIMII data set consists of sounds from four different industrial sources and they are valves, pumps, fans and slide rails - where group 28 chose to elaborate on the pump sound data set. There are seven different product models included in the pump data set and each model contains normal and abnormal sounds, where the normal sound is the pump itself and the abnormal sound are assorted background noises. Each data set was recorded with an eight-channel microphone array.

The focus of the application of the data set for Group 28, when analyzing and interpreting the MIMII data set, is to construct a machine learning algorithm that will perform a classification analysis on the individual pumps. The classification will be divided into two different categories, where the first category will be normally functioning pumps, or approved pumps and the other category will be failure pumps. Features are normally taken into consideration when implementing an algorithm on data sets, but considering the type of data set to be analyzed, more knowledge would be gained by analyzing the individual data sets with spectrograms. A spectrogram plots the spectrum of frequencies as they vary with time in a more understandable visual presentation. In the event that the spectrogram analysis indicates inconclusive results, then a Fourier transform analysis has been in consideration.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
