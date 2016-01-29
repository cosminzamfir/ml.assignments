package ml.assignments.assignment1;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ml.assignments.MLAssignmentUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;

public class Assignment1 {

	private static String dataSetName = "adult.arff";
	private static int trainingSize = 100;
	private static int testingSize = 1000;
	private static Map<Classifier, Map<Integer,Evaluation>> evaluations = new LinkedHashMap<>();
	private static List<Classifier> classifiers = new ArrayList<>();
	
	public static void main(String[] args) throws Exception {
		Instances dataSet  = MLAssignmentUtils.buildInstancesFromResource(dataSetName);
		dataSet = MLAssignmentUtils.shufle(dataSet);
		buildClassifiers();
		initializeEvaluationMap();
		for (Classifier classifier : classifiers) {
			ClassifierRunner runner = new ClassifierRunner(classifier);
			for (int i = trainingSize; i < 200; i=i+10) {
				runner.run(dataSet, i, testingSize);
				evaluations.get(classifier).put(i, runner.getEvaluationOnTestSet());
			}
		}
		printStatistics();
	}
	
	private static void initializeEvaluationMap() {
		for (Classifier classifier : classifiers) {
			Map<Integer, Evaluation> m1 = new LinkedHashMap<>();
			evaluations.put(classifier, m1);
		}
	}

	private static void printStatistics() {
		System.out.println("DataSet: " + dataSetName);
		System.out.println("Training size: " + trainingSize);
		System.out.println("Accuaracy of evaluated algorihms:");
		for (Classifier algo : evaluations.keySet()) {
			System.out.println("  - " + algo.getClass().getSimpleName() + ":");
			Map<Integer, Evaluation> eval = evaluations.get(algo);
			for (Integer trainingSize : eval.keySet()) {
				System.out.println("     - " + trainingSize + " : " + eval.get(trainingSize).pctCorrect());
			}
		}
	}

	private static void buildClassifiers() {
		classifiers.add(MLAssignmentUtils.buildDecisionTree(true));
		classifiers.add(MLAssignmentUtils.buildKNearestNeibor());
		classifiers.add(MLAssignmentUtils.buildNeuralNet());
		classifiers.add(MLAssignmentUtils.buildBoosting());
	}
}
