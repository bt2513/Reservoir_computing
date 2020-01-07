# Reservoir Computing Project

Authors: *Bertrand Thia-Thiong-Fat, Yassine Kamri*

# Directories and files

`ActivationFunction`: contains the datapoints of the different activation functions that one team has been able to produce - using the analog board. In this folder, we visualize them. We will assess the performances of our reservoir-computing following the activation function used on two tasks (digit spoken recognition and creation of sinus exponential).

`AuditoryToolbox`: Matlab modul used to preprocess our data for the digit spoken recognition task.

`Example`: One simple example that can be runned to observe our results on the didigit spoken recognition task.

`preprocessing`: Code used to preprocess our data for the digit spoken recognition task - based on the cochlea technique.

`references-papers`: References and papers used during our work

`report.pdf`: Report of our work.

`SinuxExpo`: Code used to assess our reservoir-computer model on the creation of sinus exponential task.

`Spoken_Digit_Recognition`: Code used to assess our reservoir-computer model on the digit spoken recognition task. Folders containing the study of our model on different parameters can be found here - such as the performances with respect to the datasize, sparsity, number of neurons, activation functions, etc.


# The Objectives

In this study, we will present the results of our work on the implementation of a brain-inspired computing system. The objective is to simulate a reservoir-computer that can be programmed on the DC RASP3.0A FPAA platform from Georgia Tech Institute to validate various experiences before reproducing them on the board in practice. We will assess the performances of our simulator and validate its functioning and ability to simulate machine learning tasks.

# The data

Unfortunately, the datasets used were too heavy to be uploaded on the git. We used the dataset **TI20** for spoken digit recognition.

# Quantitative results

See the codes in the different folders. They unravel our workflow, from problem definition to our results, limitations and future works. To observe our results, the code to run is always named `main.py`.
