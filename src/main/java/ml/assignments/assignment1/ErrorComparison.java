package ml.assignments.assignment1;

import java.util.ArrayList;
import java.util.List;

import ml.assignments.CommandLineOptions;
import ml.assignments.GeneralChart;
import ml.assignments.MLAssignmentUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 * Plot the error in the training set and the error in the test set as a function of the training set.
 * @author eh2zamf
 *
 */
public class ErrorComparison {

	static int runs = 110;
	static int initialTrainingSize = 100;
	static int step = 40;
	static int testSize = 1000;
	static double[][] trainingErrorRateVersusTrainingSize = new double[runs][2];
	static double[][] testErrorRateVersusTrainingSize = new double[runs][2];
	static String fileName = "robot-moves.arff";

	public static void main(String[] args) throws Exception {
		CommandLineOptions options = CommandLineOptions.instance(args);
		Classifier classifier = options.getClassifier();
		
		ClassifierRunner runner = new ClassifierRunner(classifier, options);
		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource(options.getDataSetName(fileName));
		dataSet = MLAssignmentUtils.shufle(dataSet);
		MLAssignmentUtils.write(fileName + "_shuffled.arff", dataSet);

		int size = dataSet.size();
		Instances test = new Instances(dataSet, size - options.getTestSize(testSize), options.getTestSize(testSize));

		for (int i = 0; i < options.getRuns(runs); i++) {
			int trainingSize = options.getInitialSize(initialTrainingSize) + i * options.getStepSize(step);
			Instances training = new Instances(dataSet, 0, trainingSize);
			runner.buildModel(training);
			Evaluation testSetEvalution = runner.evaluateModel(training, test);
			Evaluation trainingSetEvaluation = runner.evaluateModel(training, training);
			trainingErrorRateVersusTrainingSize[i][0] = trainingSize;
			trainingErrorRateVersusTrainingSize[i][1] = trainingSetEvaluation.pctIncorrect();

			testErrorRateVersusTrainingSize[i][0] = trainingSize;
			testErrorRateVersusTrainingSize[i][1] = testSetEvalution.pctIncorrect();
		}
		List<double[][]> data = new ArrayList<>();
		List<String> titles = new ArrayList<>();
		data.add(trainingErrorRateVersusTrainingSize);
		data.add(testErrorRateVersusTrainingSize);
		titles.add("Training error rate");
		titles.add("Test error rate");
		String title = MLAssignmentUtils.toString(classifier) + " - error rates versus training size: " + fileName;
		new GeneralChart(title, data, titles, "Training size", "Error rate");
	}
}
