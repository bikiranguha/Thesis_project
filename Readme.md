Current way of extracting info from large datasets:
	The voltage, angle, frequency of N-2 plus fault studies with different load levels are saved in Event0.pkl to Event8.pkl. The object organization is given in PSSE Sim Script Trials/simPSSEBatchxxx.py scripts
	Since i cannot load all the objects together, this is what i do currently to get all the voltage data into arrays:
		pickleTohdf5.py: Load each object separately and then make lists out of the voltage data, and corresponding event keys are saved in keyxxx.pkl files
						 It also saves the time list into a separate pkl file
		saveVhf5.py: Load each voltage list into a separate array with keys in the hf5 file object.
		compileVDatah5f.py then generate input arrays, target arrays (for voltage oscillation) and corresponding key lists. The arrays are saved to .h5 files while the list is saved to a pickle object
		
		


testLROscLarge.py:
	Runs various performance tests with the LR model as a classifier for voltage oscillation on the large dataset (100% to 106% load)
	Generates the templates for class 0 and class 1
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
