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
 * Plot the execution time versus training set in building and evaluating the classifier
 * @author eh2zamf
 *
 */
public class StatIndicatorsComparison {

	static int runs = 20;
	static int initialTrainingSize = 100;
	static int step = 100;
	static int testSize = 1000;
	static List<double[][]> accuracies = new ArrayList<>();
	static List<double[][]> precisions = new ArrayList<>();
	static List<double[][]> recalls = new ArrayList<>();
	static List<double[][]> fMeasures = new ArrayList<>();
	static String dataSetName = "magic-gama-telescope.arff";

	static String accuraciesTitle = "Accuracy (%)" ;
	static String precisionsTitle = "Precision (%)";
	static String recalsTitle = "Recal (%)";
	static String fMeasuresTitle = "F-Measure";

	static List<Classifier> classifiers = new ArrayList<>();

	public static void main(String[] args) throws Exception {
		CommandLineOptions options = CommandLineOptions.instance(args);
		classifiers = options.getClassifiers(MLAssignmentUtils.buildClassifiers(options));
		buildArrays();

		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource(options.getDataSetName(dataSetName));
		dataSet = MLAssignmentUtils.shufle(dataSet);

		for (int i = 0; i < options.getRuns(runs); i++) {
			int size = options.getInitialSize(initialTrainingSize)+ i * options.getStepSize(step);
			Instances training = new Instances(dataSet, 0, size);
			
			Instances test = new Instances(dataSet, dataSet.size() - options.getTestSize(testSize), options.getTestSize(testSize));
			for (int j = 0; j < classifiers.size(); j++) {
				Classifier classifier = classifiers.get(j);
				ClassifierRunner runner = new ClassifierRunner(classifier, options);
				runner.buildModel(training);
				Evaluation eval = runner.evaluateModel(training, test);
				accuracies.get(j)[i][0] = size;
				accuracies.get(j)[i][1] = eval.pctCorrect();

				precisions.get(j)[i][0] = size;
				precisions.get(j)[i][1] = MLAssignmentUtils.getAveragePrecision(eval, test);
				
				recalls.get(j)[i][0] = size;
				recalls.get(j)[i][1] = MLAssignmentUtils.getAverageRecall(eval, test);
				
				fMeasures.get(j)[i][0] = size;
				fMeasures.get(j)[i][1] = MLAssignmentUtils.getAverageFMeasure(eval, test);
			}
		}
		new GeneralChart(accuraciesTitle, accuracies, asStrings(classifiers), "Training Size", "Accuracy");
		new GeneralChart(precisionsTitle, precisions, asStrings(classifiers), "Training Size", "Precision");
		new GeneralChart(recalsTitle, recalls, asStrings(classifiers), "Training Size", "Recall");
		new GeneralChart(fMeasuresTitle, fMeasures, asStrings(classifiers), "Training Size", "F-measure");
	}

	private static List<String> asStrings(List<Classifier> classifiers2) {
		List<String> res = new ArrayList<>();
		for (Classifier classifier : classifiers) {
			res.add(classifier.getClass().getSimpleName());
		}
		return res;
	}

	private static void buildArrays() {
		for (int i = 0; i < classifiers.size(); i++) {
			accuracies.add(new double[runs][2]);
			precisions.add(new double[runs][2]);
			recalls.add(new double[runs][2]);
			fMeasures.add(new double[runs][2]);
		}
	}
}
