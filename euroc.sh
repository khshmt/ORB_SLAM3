#!/bin/bash
pathDatasetEuroc='/home/dz/Documents/rosbags/EuRoC' #Example, it is necesary to change it by the dataset path


#------------------------------------
# Stereo-Inertial Examples
echo "Launching MH01 with Stereo-Inertial sensor"
./Examples/Stereo-Inertial/stereo_inertial_euroc ./Vocabulary/ORBvoc.bin ./Examples/Stereo-Inertial/EuRoC.yaml "$pathDatasetEuroc"/MH05 ./Examples/Stereo-Inertial/EuRoC_TimeStamps/MH05.txt dataset-MH05_stereoi



