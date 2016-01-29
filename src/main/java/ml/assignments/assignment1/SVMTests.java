package ml.assignments.assignment1;

import java.util.ArrayList;
import java.util.List;

import ml.assignments.GeneralChart;
import ml.assignments.MLAssignmentUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * Plot the error in the training set and the error in the test set as a function of the training set.
 * @author eh2zamf
 *
 */
public class SVMTests {

	public static enum KernelFunction {
		Liniar, Quadratic, Cubic, _4GradePolynomial, _5GradePolynomial, _6GradePolynomial, Radial, Sigmoid
	}
	static int initialTrainingSize = 100;
	static int runs = 20;
	static int step = 100;
	static int testSize = 800;
	static boolean useLibSVM = false;
	static List<double[][]> accuracyVersusTrainingSizeByKernel = new ArrayList<>();
	static List<KernelFunction> kernelFunctions = new ArrayList<>();
	static List<String> titles = new ArrayList<>();
	static String dataSetName = "robot-moves-binary.arff";
	
	static {
		kernelFunctions.add(KernelFunction.Liniar);
		kernelFunctions.add(KernelFunction.Quadratic);
		kernelFunctions.add(KernelFunction.Cubic);
		kernelFunctions.add(KernelFunction._4GradePolynomial);
		kernelFunctions.add(KernelFunction._5GradePolynomial);
		kernelFunctions.add(KernelFunction._6GradePolynomial);
		kernelFunctions.add(KernelFunction.Radial);
		for (KernelFunction function : kernelFunctions) {
			titles.add(function.toString());
		}
	}

	public static void main(String[] args) throws Exception {

		initialize();
		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource(dataSetName);
		dataSet = MLAssignmentUtils.shufle(dataSet);

		for (int k = 0 ; k < kernelFunctions.size(); k++ ) {
			KernelFunction function = kernelFunctions.get(k);
			ClassifierRunner runner = new ClassifierRunner(useLibSVM ? MLAssignmentUtils.buildLibSVM(function): MLAssignmentUtils.buildSMOSVM(function));
			for (int i = 0; i < runs; i++) {
				int trainingSize = initialTrainingSize + i * step;
				Instances training = new Instances(dataSet, 0 , trainingSize);
				Instances test = new Instances(dataSet, dataSet.size() - testSize, testSize);
				runner.buildModel(training);
				Evaluation eval = runner.evaluateModel(training, test);
				accuracyVersusTrainingSizeByKernel.get(k)[i][0] = trainingSize; 
				accuracyVersusTrainingSizeByKernel.get(k)[i][1] = eval.pctCorrect();
			}
		}
		new GeneralChart("Accuracy versus Training Size by kernel", accuracyVersusTrainingSizeByKernel, titles, "Training size", "Accuracy");

	}

	private static void initialize() {
		for (int i = 0; i < kernelFunctions.size(); i++) {
			accuracyVersusTrainingSizeByKernel.add(new double[runs][2]);
		}
	}
}
