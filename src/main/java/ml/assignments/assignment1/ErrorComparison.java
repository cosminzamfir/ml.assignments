package ml.assignments.assignment1;

import java.util.ArrayList;
import java.util.List;

import ml.assignments.GeneralChart;
import ml.assignments.MLAssignmentUtils;
import ml.assignments.assignment1.SVMTests.KernelFunction;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Plot the error in the training set and the error in the test set as a function of the training set.
 * @author eh2zamf
 *
 */
public class ErrorComparison {

	static int runs = 40;
	static int initialTrainingSize = 100;
	static int step = 100;
	static int testSize = 1000;
	static double[][] trainingErrorRateVersusTrainingSize = new double[runs][2];
	static double[][] testErrorRateVersusTrainingSize = new double[runs][2];
	static String fileName = "magic-gama-telescope.arff";

	public static void main(String[] args) throws Exception {
		
		Classifier classifier = MLAssignmentUtils.buildKNearestNeibor();
		ClassifierRunner runner = new ClassifierRunner(classifier);
		
		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource(fileName);
		dataSet = MLAssignmentUtils.shufle(dataSet);
		int size = dataSet.size();
		Instances testSet = new Instances(dataSet, size - 1000, 1000);
		
		for (int i = 0; i < runs; i ++) {
			int trainingSize = initialTrainingSize + i*step;
			Instances trainingDataSet = new Instances(dataSet, 0, trainingSize);
			runner.run(trainingDataSet, testSet);
			trainingErrorRateVersusTrainingSize[i][0] = trainingSize;
			trainingErrorRateVersusTrainingSize[i][1] = runner.getEvaluationOnTrainingSet().pctIncorrect();

			testErrorRateVersusTrainingSize[i][0] = trainingSize;
			testErrorRateVersusTrainingSize[i][1] = runner.getEvaluationOnTestSet().pctIncorrect();
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
