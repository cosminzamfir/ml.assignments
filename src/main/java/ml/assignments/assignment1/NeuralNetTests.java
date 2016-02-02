package ml.assignments.assignment1;

import java.util.ArrayList;
import java.util.List;

import ml.assignments.CommandLineOptions;
import ml.assignments.GeneralChart;
import ml.assignments.MLAssignmentUtils;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

/**
 * Plot the error in the training set and the error in the test set as a function of the training set.
 * @author eh2zamf
 *
 */
public class NeuralNetTests {

	static int initialHiddenLayers = 1;
	static int runs = 20;
	static double step = 1;
	static int trainingSize = 1000;
	static int testSize = 1000;
	static double[][] accuracies = new double[runs][2];
	static String dataSetName = "robot-moves.txt";

	public static void main(String[] args) throws Exception {
		CommandLineOptions options = CommandLineOptions.newInstance(args);
		MultilayerPerceptron classifier = MLAssignmentUtils.buildNeuralNet(options);
		ClassifierRunner runner = new ClassifierRunner(classifier);
		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource(options.getDataSetName(dataSetName));
		dataSet = MLAssignmentUtils.shufle(dataSet);
		int size = dataSet.size();
		
		Instances trainingDataSet = new Instances(dataSet, 0, options.getTrainingSize(trainingSize));
		Instances testSet = new Instances(dataSet, size - options.getTestSize(options.getTestSize(testSize)), options.getTestSize(testSize));
		for (int i = 0; i < options.getRuns(runs); i ++) {
			int hiddentUnits = options.getInitialSize(initialHiddenLayers) + i;
			classifier.setHiddenLayers(String.valueOf(hiddentUnits));
			runner.run(trainingDataSet, testSet);
			accuracies[i][0] = hiddentUnits;
			accuracies[i][1] = runner.getEvaluationOnTrainingSet().pctIncorrect();
		}
		List<double[][]> data = new ArrayList<>();
		List<String> titles = new ArrayList<>();
		data.add(accuracies);
		titles.add("Error rate as function of hidden units");
		String title = classifier.getClass().getSimpleName() + " - error rates versus hidden units : " + options.getDataSetName(dataSetName);
		new GeneralChart(title, data, titles, "Hidden Units", "Accuracy");
	}
}
