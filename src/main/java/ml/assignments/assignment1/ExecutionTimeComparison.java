package ml.assignments.assignment1;

import java.util.ArrayList;
import java.util.List;

import ml.assignments.GeneralChart;
import ml.assignments.MLAssignmentUtils;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Plot the execution time versus training set in building and evaluating the classifier
 * @author eh2zamf
 *
 */
public class ExecutionTimeComparison {

	static int runs = 40;
	static int initialSize = 1;
	static int step = 100;
	static List<double[][]> buildingTimes = new ArrayList<>();
	static List<double[][]> evaluationTimes = new ArrayList<>();
	static List<Classifier> classifiers = new ArrayList<>();
	static String buildingTimeTitle = "Classifier building time (ms)";
	static String evaluationTimeTitle = "Classifier evaluation time (ms)";

	public static void main(String[] args) throws Exception {
		long start;
		long end;

		classifiers = MLAssignmentUtils.buildClassifiers();
		buildArrays();
		
		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource("magic-gama-telescope.arff");
		//Instances dataSet = MLAssignmentUtils.buildInstancesFromResource("robot-moves-binary.arff");
		
		dataSet = MLAssignmentUtils.shufle(dataSet);
		
		for (int i = 0; i < runs; i++) {
			int size = initialSize + i * step;
			Instances training = new Instances(dataSet, 0, size);
			for (int j = 0; j < classifiers.size(); j++) {
				Classifier classifier = classifiers.get(j);
				ClassifierRunner runner = new ClassifierRunner(classifier);
				
				start = System.currentTimeMillis();
				runner.buildModel(training);
				end = System.currentTimeMillis();
				buildingTimes.get(j)[i][0] = size;
				buildingTimes.get(j)[i][1] = (end - start);
				
				start = System.currentTimeMillis();
				runner.evaluateModel(training, training);
				end = System.currentTimeMillis();
				evaluationTimes.get(j)[i][0] = size;
				evaluationTimes.get(j)[i][1] = (end - start);
			}
		}
		new GeneralChart(buildingTimeTitle, buildingTimes, asStrings(classifiers), "Training size", "Classifier building time");
		new GeneralChart(evaluationTimeTitle, evaluationTimes, asStrings(classifiers), "Testing size", "Classification time");
	}

	private static List<String> asStrings(List<Classifier> classifiers2) {
		List<String> res = new ArrayList<>();
		for (Classifier classifier : classifiers) {
			res.add(classifier.getClass().getSimpleName());
		}
		return res;
	}

	private static void buildArrays() {
		for (Classifier classifier : classifiers) {
			buildingTimes.add(new double[runs][2]);
			evaluationTimes.add(new double[runs][2]);
		}
	}
}
