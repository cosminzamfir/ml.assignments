package ml.assignments.assignment1;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import ml.assignments.CommandLineOptions;
import ml.assignments.MLAssignmentUtils;

public class Main {

	public static void main(String[] args) throws Exception {
		CommandLineOptions options = CommandLineOptions.newInstance(args);
		String dataSetName = options.getDataSetName();
		int trainingSize = options.getTrainingSize();
		int testSize = options.getTestSize(1000);
		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource(dataSetName);
		MLAssignmentUtils.shufle(dataSet);
		Instances training = new Instances(dataSet, 0, trainingSize);
		Instances test = new Instances(dataSet, dataSet.size() - testSize, testSize);
		
		
		ClassifierRunner runner = new ClassifierRunner(options.getClassifier());
		runner.buildModel(training);
		Classifier classifier = runner.getClassifier();
		System.out.println(classifier);
		System.out.println("=================================================================");
		Evaluation eval = runner.evaluateModel(training, test);
		System.out.println(eval.toSummaryString(true));
		System.out.println("=================================================================");
		System.out.println(eval.toMatrixString());
	}
}
