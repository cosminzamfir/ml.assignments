package ml.assignments.assignment1;

import java.util.ArrayList;
import java.util.List;

import ml.assignments.CommandLineOptions;
import ml.assignments.GeneralChart;
import ml.assignments.MLAssignmentUtils;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

/**
 * Plot the error in the training set and the error in the test set as a function of the training set.
 * @author eh2zamf
 *
 */
public class NeuralNetTests {

	static int initialHiddenLayers = 5;
	static int runs = 1;
	static double step = 1;
	static int trainingSize = 1000;
	static int testSize = 1000;
	static double[][] accuracies = new double[runs][2];
	static double[][] crossValidatedAccuracies = new double[runs][2];
	static String dataSetName = "robot-moves.arff";

	public static void main(String[] args) throws Exception {
		CommandLineOptions options = CommandLineOptions.instance(args);
		MultilayerPerceptron classifier = MLAssignmentUtils.buildNeuralNet(options);
		ClassifierRunner runner = new ClassifierRunner(classifier, options);
		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource(options.getDataSetName(dataSetName));
		dataSet = MLAssignmentUtils.shufle(dataSet);
		int size = dataSet.size();
		
		Instances training = new Instances(dataSet, 0, options.getTrainingSize(trainingSize));
		Instances test = new Instances(dataSet, size - options.getTestSize(options.getTestSize(testSize)), options.getTestSize(testSize));
		for (int i = 0; i < options.getRuns(runs); i ++) {
			int hiddentUnits = options.getInitialSize(initialHiddenLayers) + i;
			classifier.setHiddenLayers(String.valueOf(hiddentUnits));
			runner.buildModel(training);
			Evaluation testSetEvaluation = runner.evaluateModel(training, test);
			accuracies[i][0] = hiddentUnits;
			accuracies[i][1] = testSetEvaluation.pctIncorrect();
			
			Evaluation crossValidation = runner.crossValidate(training);
			crossValidatedAccuracies[i][0] = hiddentUnits;
			crossValidatedAccuracies[i][1] = crossValidation.pctIncorrect();
			
		}
		List<double[][]> data = new ArrayList<>();
		List<String> titles = new ArrayList<>();
		data.add(accuracies);
		data.add(crossValidatedAccuracies);
		titles.add("Test error rate as function of hidden units");
		titles.add("Cross-validation error rate as function of hidden units");
		String title = classifier.getClass().getSimpleName() + " - error rates versus hidden units : " + options.getDataSetName(dataSetName);
		new GeneralChart(title, data, titles, "Hidden Units", "Accuracy");
	}
}
