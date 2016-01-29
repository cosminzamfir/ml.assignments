package ml.assignments.assignment1;

import java.util.ArrayList;
import java.util.List;

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

	static double initialMomentum = 0;
	static int runs = 20;
	static double step = 0.05;
	static int trainingSize = 1000;
	static int testSize = 1000;
	static double[][] accuracyVersusMomentum = new double[runs][2];

	public static void main(String[] args) throws Exception {
		MultilayerPerceptron classifier = MLAssignmentUtils.buildNeuralNet();
		ClassifierRunner runner = new ClassifierRunner(classifier);
		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource("magic-gama-telescope.arff");
		dataSet = MLAssignmentUtils.shufle(dataSet);
		int size = dataSet.size();
		
		Instances trainingDataSet = new Instances(dataSet, 0, trainingSize);
		Instances testSet = new Instances(dataSet, size - 1000, 1000);
		for (int i = 0; i < runs ; i ++) {
			double momentum = initialMomentum + step * i;
			classifier.setMomentum(momentum);
			runner.run(trainingDataSet, testSet);
			accuracyVersusMomentum[i][0] = momentum;
			accuracyVersusMomentum[i][1] = runner.getEvaluationOnTrainingSet().pctIncorrect();
		}
		List<double[][]> data = new ArrayList<>();
		List<String> titles = new ArrayList<>();
		data.add(accuracyVersusMomentum);
		titles.add("Error rate as function of momentum");
		String title = classifier.getClass().getSimpleName() + " - error rates versus momentum: magic-gama-telescope.arff";
		new GeneralChart(title, data, titles, "Momentum", "Accuracy");
	}
}
