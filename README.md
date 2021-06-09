# ml4comm-icc21
Code for IEEE ICC'2021 Tutorial 14: Machine Learning for MIMO Systems with Large Arrays

Aldebaro Klautau (UFPA), Nuria Gonzalez-Prelcic (NCSU) and Robert W. Heath Jr. (NCSU)

June 14, 2021

In order to run the beam_selection.ipynb notebook, you need to download the input and output data for the neural network.
The files can be obtained at https://nextcloud.lasseufpa.org/s/mrzEiQXE83YE3kg
where you can find folders with the three distinct sets of input parameters: coord, image, lidar. You can use only one, all 3 or any combination.
You also need the correct output labels, which is the file beams_output.npz in the folder beam_output.
After downloading the datasets, save them in the folder specified in the notebook or change the code to point to your folder. The default folder name is data.
