# Face_Swap

In this project we implement an end-to-end pipeline to swap faces. We achieve the task using two approaches: (a) Classical Computer Vision and (b) Deep Learning Approach. 

In part 'a', we use two methods namely Delaunay Triangulation and Thin Plate Splines for face warping.

In part 'b', we use the pre trianed PRNet model for face swap. 

Outputs obtained from all the methods are compared.

## Authors

 1. Sri Manika Makam
 2. Akanksha Patel

## Instructions to run

Download the folder. To run the program, you have to give two command line arguments, video and method. If your video has only one face, give the argument 'video' as 0 and if it has more than one faces, give it as 1. You give 'method' as 1 for Triangulation output, 2 for TPS ouput and 3 for PRNet output.

Download the folder. Go to the directory where Wrapper.py is present.
For example if you want get the Triangulation output for a video with one face, run the following command 

```
python Wrapper.py --video 0 --method 1
```
By deafult, the video with one face is given as Data1 (present in Data folder) and the default picture with which the face in Data1 is swapped with, is Selena Gomez (sg.jpg in Data folder). The default video with multiple faces is given as Data2. 

If you want to run the program on new videos, download the video to Data folder and change the path name in Wrapper.py

## Output

The Triangulation, TPS and PRNet outputs for Data1 and Data can be found in Data folder. The outputs for Test Sets can be found in Data/TestSetOutputs folder. 
