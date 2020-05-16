# Reservoir Computing Project

Authors: *Bertrand Thia-Thiong-Fat, Yassine Kamri*


This repository presents the results of our work on the implementation of deep learning tasks on a FPAA (Field-Programmable Analog Array). We simulated a reservoir-computer based on the DC RASP3.0A FPAA platform of Georgia Tech Institute to validate various experiences - before reproducing them on the board in practice. The performances of our simulator on two deep learning tasks can be found here: the sine-wave generator and spoken digit recognition tasks.

# Directories and files

`ActivationFunction`: contains the datapoints and visualizations of the different activation functions the team produced using the analog board (with resistors, capacitors and op-amps).

`AuditoryToolbox`: Matlab module used to preprocess our data for the digit spoken recognition task.

`Example`: One simple example that can be run to observe our results on the digit spoken recognition task.

`preprocessing`: Code used to preprocess our data for the digit spoken recognition task using the cochlea technique.

`references-papers`: References and papers used during our work.

`report.pdf`: Report of our work.

`SinuxExpo`: Code used to test our reservoir-computer model on the sine-wave generator task.

`Spoken_Digit_Recognition`: Code used to assess our reservoir-computer model on the digit spoken recognition task. Folders containing the study of our model on different hyperparameters can be found - such as the performances with respect to the datasize, sparsity of the network, number of neurons, activation functions, etc.

# The data

We used the dataset **TI20** for the spoken digit recognition task.

# Quantitative results

Please refer to the report. It unravels our workflow, from problem definition to the results, limitations and future works. To observe the results obtained in a given folder, please download it and run the file `main.py`.
