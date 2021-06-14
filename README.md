# Machine Learning for MIMO Systems with Large Arrays

Code for IEEE ICC'2021 Tutorial 14 

Aldebaro Klautau (UFPA), Nuria Gonzalez-Prelcic (NCSU) and Robert W. Heath Jr. (NCSU)

June 14, 2021

# Installation of Jupyter notebooks for 16QAM classification

The notebooks *lstm_time_variant_channel.ipynb* and *qam_classifiers.ipynb* generate themselves the used data. The main dependencies are listed in requirements.txt. You can use, for instance:

```pip install -r requirements.txt```

They can also be executed in Google's Colab.

# Installation of the beam_selection notebook

The notebook *beam_selection.ipynb* requires downloading the input and output data for the neural network.
The files can be obtained at https://nextcloud.lasseufpa.org/s/mrzEiQXE83YE3kg
where you can find folders with the three distinct sets of input parameters: coord, image, lidar. You can use only one, all 3 or any combination.
You also need the correct output labels, which is the file beams_output.npz in the folder beam_output.
After downloading the datasets, save them in the folder specified in the notebook or change the code to point to your folder. The default folder name is data.

In case you intend to use all three sets of input parameters, you will end up with the files in the following folders:
* Contents of: your_folder\ml4comm-icc21\data
  *                                          beam_output
  *                                          coord_input
  *                                          image_input
  *                                          lidar_input
* Contents of: your_folder\ml4comm-icc21\data\beam_output
 *                                            45,850,920 beams_output.npz  
 *                                            45,850,920 beams_output_no_ori.npz (we will not use this one)
* Contents of: your_folder\ml4comm-icc21\data\coord_input
 *                                            179,380 coord_input.npz
* Contents of: your_folder\ml4comm-icc21\data\image_input
 *                                            43,522,538 img_input_20.npz
* Contents of: your_folder\ml4comm-icc21\data\lidar_input
 *                                            447,760,264 lidar_input.npz
