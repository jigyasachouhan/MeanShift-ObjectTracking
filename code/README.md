# MeanShift-ObjectTracking
CS 663: Course Project

start.ipynb: This was just the inital Proof of Concept Notebook

meanshift.py: The first working model - in which the window size does not change, but using the mean shift algorithm we track it.

camshift_resize.py : Added the feature that the window size changes - based on the pixels seen inside.

camshift_rot.py: Added the feature of displaying a rotated object window - on the basis of our estimate of the orientation of the object of interest.

We have also provided another file, cv_camshift.py - which is just CAMShift using OpenCV's functions for the same. This is just to serve as a comparison.

Usage: python3 <algo_file_name> <video_name>

Options for video names are: case, rubics, walk, walksit, test_person, chainsnatch, raghav, tiger, attacktiger, lioness.
These are located in the ./videos/ directory.

