


plotSingleCaseTS3ph.py:
	Use this script to plot any case (using any file) in TS3ph. All the relevant petsc options need to be specified here

PFORNLSim folder:
	Contains the results of N-2 F simulations on the pf_ornl raw file using PSSE (detailed in N_2Framework.py)

PFORNLTS3phScripts folder:
	Contains scripts which run pf_ornl simulations in TS3ph in parallel.
	
LSTM Prediction folder:
	Contains examples of one-step ahead and multi-time step ahead predictions of time series using LSTM.
	Tried to apply on the NREL PMU angle data, but the results mostly sucked.

IIT PMU data:
	Contains anomaly detection experiments on the IIT PMU data

comed PMU data:
	Contains anomaly detection experiments on the comed data

	
	
TSIRegressor.py:
	build a regressor which can output the transient stability index by looking at all the generator 
	angles after a fault
	tries various regressors	

New framework for getting N-2 F simulation data from PSSE as well as TS3ph:
	Totally explained in N_2Framework.py
	generateSimBatches.py contains scripts to generate scripts for sim in PSSE and TS3ph (sequential as well as parallel)
Framework developed for testing three class angle stability classifier:
	See or run ThreeClassClassifierFramework.py.
	Note that the angle separation events and the oscillation threshold has been chosen from analyses carried out 
	in 	analyzeOsc.py and analyzeN_2FAngles.py respectively.
		
Current way of extracting info from the large N-2 plus fault dataset (using all the scaled loads): 
	gatherN_2FData.py: Gather all the data (voltage, angle, frequency) from the small csv chunks and make it one large file.
	visualizeN_2F.py: To visualize any event at a bus, just provide the event key
	gatherGenSteadyAngles.py: 
		organizes all the generator angle data during (apparent) steady state, this helps in determining if there is any angle oscillation
		has easy modifications to also get the whole angle data (from start to beginning) for all the buses in one row
		has also been used to gather all the gen trip scenarios for visualization of angles
	analyzeOsc.py: 
		script which contains experiments to determine the relative angle locus and also to determine angle separation (in which events and when)
		has scripts to sort all the angle abnormalities, determine which cases need generator tripping
		has a section which visualizes the angles for any given simulation.
	PSSE Simulation Scripts/simGenTrip.py:
		Uses 'GenTripData.txt' (generated from analyzeOsc.py) to simulate those cases which need generator tripping while running the N-2 plus fault cases.
		These modified simulations are then saved into new sets of csv files which will be integrated later on into the machine learning analysis workspace.
	integrateN_2VA.py:
		outputs a new vN_2F and aN_2F csv file which contains steady state (last 120) values of all the events, with each row corresponding to an event.
		The corresponding event list and bus list are also saved.
		Same thing is done to get the transient samples (a predefined number of samples ) after the final line is outaged.
	
	analyzeVoltageN_2FLarge.py:
		Looks at the steady state voltage (bus wise) and set thresholds to classify oscillation
		Tries out different classifiers and gets the precision, recall, f1 score and accuracies on the dataset.

	visualizeN_2FSteadyAngles.py:
		Use to visualize the steady state gen angles at all buses for a given event.
	analyzeN_2FAngles.py:
		analyze the steady state angles, set targets and classify oscillation vs non-oscillation
		tests a lot of different classifier models
		Also implements PCA based event extraction
	
	
clusterOScGenAngle.py:
	apply clustering algorithms (K Means and Hierarchical clustering) on the generator angle data (N-2 F) and see what happens
	
Old way of extracting info from the large N-2 plus fault dataset (using all the scaled loads):
	The voltage, angle, frequency of N-2 plus fault studies with different load levels are saved in Event0.pkl to Event8.pkl. The object organization is given in PSSE Sim Script Trials/simPSSEBatchxxx.py scripts
	Since i cannot load all the objects together, this is what i do currently to get all the voltage data into arrays:
		pickleTohdf5.py: Load each object separately and then make lists out of the voltage data, and corresponding event keys are saved in keyxxx.pkl files
						 It also saves the time list into a separate pkl file
		saveVhf5.py: Load each voltage list into a separate array with keys in the hf5 file object.
		compileVDatah5f.py then generate input arrays, target arrays (for voltage oscillation) and corresponding key lists. The arrays are saved to .h5 files while the list is saved to a text file

		


ECV2FGLTAllBus.py:
	Event (fault, gen outage, line outage, tf outage) classification using three phase voltage and angles of all buses together
	Noise, smoothing, time shifts are there, but no time shifts as of yet
	5 PMUs are available only	
ECV2FGLT1Busv2.py:
	Event (fault, gen outage, line outage, tf outage) classification using three phase voltage of one bus as a sample
	Noise, smoothing, time shifts are all there
	5 PMUs are available only
testFaultClassifierLISingleBus.py:
	Just a fault classifier using seq data and single three phase bus as input.
		
		
testMDSXAng.py:
	Gets the co-ord map using branch impedance data
	Also contains MDS experiments using the angles
EventClassifier1BuswSNTS.py:
	Improvement from EventClassifier1BuswSNTS.py: added motor start
		
EventClassifier1BuswSNTS.py:
	Improvement from EventClassifier1BuswSN.py: added time shift
		
EventClassifier1BuswSN.py:
	Improvement from EventClassifier1Bus.py: added noise and smoothing



	
EventClassifier1Bus.py:
	Event (fault and gen outage) classification using three phase voltage of one bus as a sample
		
testEventClassifierAllBusesv4.py:
	has scripts which puts all the fault and gen outage data into an array for testing with LSTM and SVM.
	Changes from v3:
		Line outage is not considered anymore
		Provisions to time shift the data
		
testEventClassifierAllBusesv3.py:
	Fault, gen and line outage classification, no time shift for evaluation
	SVM and LSTM performance can be evaluated.
	
plotSingleCase.py:
	Use this file to generate all volt, freq and angle plots for different type of events
	Currently supports line outages and gen outages
		
		
testFaultClassifierAllVPMUMimic.py:
	Improves upon testFaultClassifierAllEventDataShifted.py by adding a 6 cycle filter and noise to mimic PMU data
	Also tests xgboost and lstm on the dataset
		
		
simDist.py:
	Used to simulate generate gen outages and line outages on savnw_conp and its scaled up loads 

		
testMDS.py:
	Implement multi-dimensional scaling to try to get co-ordinates of different buses using 
	voltage drop difference during faults
		
clusterAngle.py:
	Initial experimentation with the synchrophasor data
		
clustMultCont.py: 
	used to show the operation of the hierarchical clustering algorithm in cases where mutliple events (N-2 plus fault) occur in a timeframe
		
testFaultClassifierAllEventDataShifted.py:
	improvement from testFaultClassifierAllEventData: the input data is also shifted by half the time window
	Also has feature to output the voltage data in ranked order according to min voltage recorded, essentially saying which buses are closest to fault
		
		
testFaultClassifierAllEventData.py:
	here the classifier only outputs the fault type using the voltage data of all the buses at the time of fault
		
testFaultClassifierLIFTv2.py:
	Fault classification based on type as well as location (here vs elsewhere)
		
		
testFaultClassifierLIFT.py:
	Fault classification based on type only
		
EventDetectPCAFn.py:
	Function which implements the PCA based event detection proposed in GFKA2015. Given x (time), y (signal) and the steady state time window, it returns a list of 
	time indexes when the difference between the predicted steady state signal and the actual signal deviates beyond a certain threshold.
avgFilterFn.py:
			Implements a multi-cycle filter
		
		
testFaultClassifier.py:
	Script to get the voltage csv file and use it to train a classifier which can classify:
	Class 0: Steady state
	Class 1: Three phase fault at bus
	Class 2: SLG fault at bus
	Class 3: Something happened, not a fault at this bus

testFaultClassifierv2.py:
	Same as testFaultClassifier.py with the improvement that the classifier also tries to categorize SLG phase B and C as class 2

testFaultClassifierv3.py:
	Same as testFaultClassifierv2.py except:
		SLG B has class 3
		SLG C has class 4
		everything else has class 5
	Added functionality to see output when the results are shifted.
	
testFaultClassifierv4.py:
	Improvements from v3:
		The event data are phase shifted upto 5 timesteps and added to the data matrix
		Also add the standard deviation of each phase data at the end
convRawPconToZcon.py:
	converts constant power loads to constant impedance loads and saves the raw file
		
TS3phSim3phFaults.py:
	Script to apply three phase fault and phase A line to ground fault to all the buses in the raw file and save the data to csv files
TS3phSim3phFaultsL.py:	
	Script to apply three phase fault and PALG fault to all buses for multiple raw files (with scaled loads)
	Also put functionality to check the output of the classifier to a voltage stream

TS3phSim3phFaultsLI.py:	
	Similar to TS3phSim3phFaultsL.py, the difference is that here we put fault impedances = very low, impedance and 
	impedance/2 of all branches connected to current fault bus
	
runSimFn3ph.py:
	Function to get three phase voltage and angles for all buses while simulating an event, and also write these stuff to csv file as well as 
	the event id to a separate text file. This fn is used by TS3phSim3phFaults.py

runSimFn3phL.py::
	Same as runSimFn3ph.py, just built for TS3phSim3phFaultsL.py
runSimFn3phLI.py::
	Same as runSimFn3phL.py, just built for TS3phSim3phFaultsLI.py
	
	
testLROscLarge.py:
	Runs various performance tests with the LR model as a classifier for voltage oscillation on the large dataset (100% to 106% load)
	Generates the templates for class 0 and class 1
	Evaluates performance using templates and similarity thresholds and maximum voltage after fault clearance.
	Also has script to do cross-validation and feature scaling (mean 0, variance 1)
testRNN.py:
	Builds a Sequential RNN model from keras library
	Has following features:
		Test any case
		Generate plots of performance wrt the number of time steps in input
		Generate plots of performance wrt the number of epochs
		Generate plots of performance wrt the test/train ratio
		Get the average accuracy, false positives and false negatives over a 100 trials

testLRPerformanceOsc.py:
	Script to rigorously test the LR classifier on the voltage oscillation data
	The features tested include:
		The raw time series data 
		A combination of the following features: similarity thresholds to the templates, peak voltage overshoot, gen ratio at a depth of one to the fault bus, rise time
	Also contains code to generate 3d plot of the accuracy wrt no. of timesteps and test/train ratio. Can save the 3d plot to be loaded interactively

load3DPlot.py:
	Loads a 3d plot to view interactively
AbnormalVoltClassifierSimple.py: 
	Uses an LR model to classify low (or high) voltage vs normal voltage
	Uses a bunch of features (such as voltage recovery value, prefault voltage, load ratio, distance to fault) as inputs
	Has functionality of showing or saving histograms of the input distribution
	Carries out performance analysis using a combination of features to get which feature has what impact
	Also carries out 3d surface plots of the performance accuracy wrt any variable you choose
	
OscillationClassifierSimple.py:
	Same functionality as AbnormalVoltClassifierSimple.py except for the feature which is being classified here is the oscillation in voltage.

voltage_template.py:
	# load the voltage data and get the voltage templates from the class 0 and class 1 oscillation data
	# then test some of the samples using similarity indices
	# implements some of the ideas from RGNCT2010.pdf
	
TS3phSimN_2FaultDataSave.py: Runs N-2 (fault in between) simulations in TS3ph in which there are no topological inconsistencies and saves the voltage and angle data in separate 
pickle objects
VoltageAnalysisN_1.py: Script to carry out N-1 line outages and organize the average voltages in the 10 time steps after the event, lowest voltage first
VoltageAnalysisN_2.py: Script to carry out N-2 line outages (which do not cause any topology inconsistencies) 
and organize the average voltages in the 10 time steps after the event, lowest voltage first
getFlows.py: # get the branch and tf flows from the bus report in descending order
runSimFn.py: Function which executes the simulation and returns a structure containing relevant results. Currently, its only voltage magnitude and angle, but probably we will add more signals
later on
input_data.py: Looks at the input file and harnesses the data to be used by mainFile.py
mainFile.py: Main script which runs the simulations, generates the results and tries to reconstruct the real world data. Currently, it compares real world sim with TS3ph simulation and gathers new 
events by comparing voltage magnitude data. Every time a new event is detected, TS3ph simulation is run again.

changeLoadFn.py:	Changes load and generator dispatch and saves the new raw file
changeLoadFnv2.py:  The only difference here is that the input raw file's load is converted to constant power.
runSimPSSE.py: Function to run N-2 + fault contingencies in PSSE
			   currently saves the voltage, angle and frequency of the bus

fix_indent_problem.py: Script to fix the mixed tab indentation problem
getROCFn.py: Function to calculate the time rate of change of a variable
readMACINFO.py: Function to organize the read mac info data

			   
Old Files:
batch_plot_v_0922.py: File from which batch_get_v.py is derived (copied from 'CAPE-TS Simulations' project)
batch_get_v.py: Original file to get all the complex voltage info (mag and angle) under a structure with the bus number as the key
batch_get_v_fault.py: Basically tries to simulate the real world cases, where some event happens. (derived from batch_get_v.py)
batch_get_v_nodist.py: Tries to simulate TS3ph running, so no disturbance (derived from batch_get_v.py)
