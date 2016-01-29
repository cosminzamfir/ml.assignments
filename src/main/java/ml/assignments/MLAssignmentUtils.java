package ml.assignments;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

import ml.assignments.assignment1.SVMTests.KernelFunction;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.pmml.consumer.SupportVectorMachineModel;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.Saver;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class MLAssignmentUtils {

	public static Instances buildInstancesFromResource(String resourceName) {
		Reader r;
		try {
			r = new java.io.BufferedReader(new java.io.InputStreamReader(Thread.currentThread().getContextClassLoader().getResourceAsStream(resourceName)));
			return new Instances(r);
		} catch (Exception e) {
			throw new RuntimeException("", e);
		}
	}

	public static Instances buildInstancesFromFileName(String fileName) {
		Reader r;
		try {
			r = new java.io.BufferedReader(new java.io.FileReader(fileName));
			return new Instances(r);
		} catch (Exception e) {
			throw new RuntimeException("", e);
		}
	}

	public static void split(String arffFileName, double trainPecentage) throws Exception {
		Instances trainingDataSet = MLAssignmentUtils.buildInstancesFromFileName(arffFileName);
		Filter filter = new Randomize();
		filter.setInputFormat(trainingDataSet);
		Instances filtered = Filter.useFilter(trainingDataSet, filter);

		ArffSaver trainSaver = new ArffSaver();
		ArffSaver testSaver = new ArffSaver();
		trainSaver.setFile(new File(arffFileName.replace(".arff", "_train.arff")));
		testSaver.setFile(new File(arffFileName.replace(".arff", "_test.arff")));
		testSaver.setRetrieval(Saver.INCREMENTAL);
		trainSaver.setRetrieval(Saver.INCREMENTAL);
		testSaver.setStructure(filtered);
		trainSaver.setStructure(filtered);

		int border = (int) (filtered.size() * trainPecentage);
		for (int i = 0; i < border; i++) {
			trainSaver.writeIncremental(filtered.get(i));
		}
		for (int i = border; i < filtered.size(); i++) {
			testSaver.writeIncremental(filtered.get(i));
		}
	}

	public static void write(String fileName, Instances dataSet) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setFile(new File("C:/work/data/workspace/ml.assignments/src/main/resources/" + fileName));
		saver.setRetrieval(Saver.BATCH);
		saver.setInstances(dataSet);
		saver.writeBatch();
	}

	public static Instances shufle(Instances input) throws Exception {
		Filter randomizer = new Randomize();
		randomizer.setInputFormat(input);
		return Filter.useFilter(input, randomizer);
	}

	public static Instances loadCSV(String resourceName) throws IOException {
		CSVLoader loader = new CSVLoader();
		loader.setSource(Thread.currentThread().getContextClassLoader().getResourceAsStream(resourceName));
		Instances res = loader.getDataSet();
		return res;
	}

	public static void main(String[] args) throws Exception {
		Instances ds = loadCSV("prudential-life-insurance.csv");
		write("prudential-life-insurance.arff", ds);
		//		for (int i = 0; i < ds.numAttributes(); i++) {
		//			Attribute attr = ds.attribute(i);
		//			if (ds.attributeStats(i).distinctCount != 2) {
		//				System.out.println("@attribute " + attr.name() + " " + Attribute.typeToString(attr));
		//			} else {
		//				System.out.println("@attribute " + attr.name() + " {" + ds.attributeStats(i).numericStats.min + ", " + ds.attributeStats(i).numericStats.max
		//						+ "}");
		//			}
		//		}
	}

	public static J48 buildDecisionTree(boolean usePruning) {
		J48 res = new J48();
		res.setUnpruned(!usePruning);
		return res;
	}

	public static IBk buildKNearestNeibor() {
		IBk res = new IBk();
		res.setKNN(10);
		res.setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING));
		return res;
	}

	public static AdaBoostM1 buildBoosting() {
		AdaBoostM1 res = new AdaBoostM1();
		res.setClassifier(new J48());
		return res;
	}

	public static MultilayerPerceptron buildNeuralNet() {
		MultilayerPerceptron res = new MultilayerPerceptron();
		return res;
	}

	public static LibSVM buildLibSVM(KernelFunction function) {
		LibSVM svm = new LibSVM();
		if (function == KernelFunction.Liniar) {
			svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_LINEAR, LibSVM.TAGS_KERNELTYPE));
		} else if (function == KernelFunction.Quadratic) {
			svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE));
			svm.setDegree(2);
		} else if (function == KernelFunction.Quadratic) {
			svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE));
			svm.setDegree(2);
		} else if (function == KernelFunction.Cubic) {
			svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE));
			svm.setDegree(3);
		} else if (function == KernelFunction._4GradePolynomial) {
			svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE));
			svm.setDegree(4);
		} else if (function == KernelFunction._5GradePolynomial) {
			svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE));
			svm.setDegree(5);
		} else if (function == KernelFunction._6GradePolynomial) {
			svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_POLYNOMIAL, LibSVM.TAGS_KERNELTYPE));
			svm.setDegree(6);
		} else if (function == KernelFunction.Radial) {
			svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_RBF, LibSVM.TAGS_KERNELTYPE));
		} else if (function == KernelFunction.Sigmoid) {
			svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_SIGMOID, LibSVM.TAGS_KERNELTYPE));
		}
		return svm;
	}

	public static SMO buildSMOSVM(KernelFunction function) {
		SMO smo = new SMO();
		if (function == KernelFunction.Liniar) {
			PolyKernel kernel = new PolyKernel();
			kernel.setExponent(1);
			smo.setKernel(kernel);
		} else if (function == KernelFunction.Quadratic) {
			PolyKernel kernel = new PolyKernel();
			kernel.setExponent(2);
			smo.setKernel(kernel);
		} else if (function == KernelFunction.Cubic) {
			PolyKernel kernel = new PolyKernel();
			kernel.setExponent(3);
			smo.setKernel(kernel);
		} else if (function == KernelFunction._4GradePolynomial) {
			PolyKernel kernel = new PolyKernel();
			kernel.setExponent(4);
			smo.setKernel(kernel);
		} else if (function == KernelFunction._5GradePolynomial) {
			PolyKernel kernel = new PolyKernel();
			kernel.setExponent(5);
			smo.setKernel(kernel);
		} else if (function == KernelFunction._6GradePolynomial) {
			PolyKernel kernel = new PolyKernel();
			kernel.setExponent(6);
			smo.setKernel(kernel);
		} else if (function == KernelFunction.Radial) {
			RBFKernel kernel = new RBFKernel();
			smo.setKernel(kernel);
		} else if (function == KernelFunction.Sigmoid) {
			throw new RuntimeException(KernelFunction.Sigmoid + " not supported by " + smo.getClass().getName());
		}
		return smo;
	}

	public static String toString(J48 classifier) {
		return classifier.getClass().getSimpleName() + " -pruned=" + !classifier.getUnpruned();
	}

	public static String toString(IBk classifier) {
		return classifier.getClass().getSimpleName() + " - k=" + classifier.getKNN() + " - weighting:" + classifier.getDistanceWeighting();
	}

	public static String toString(AdaBoostM1 classifier) {
		return classifier.getClass().getSimpleName() + " - maxIterations=" + classifier.getNumIterations() + " - baseLearner="
				+ classifier.getClassifier().getClass().getSimpleName();
	}

	public static String toString(MultilayerPerceptron classifier) {
		return classifier.getClass().getSimpleName() + " - learningRate=" + classifier.getLearningRate() + " - momentum=" + classifier.getMomentum()
				+ " - hiddenUnits=" + classifier.getHiddenLayers();
	}

	public static String toString(SMO classifier) {
		return classifier.getClass().getSimpleName() + " - kernel=" + classifier.getKernel().getClass().getSimpleName();
	}

	public static String toString(Classifier classifier) {
		return classifier.getClass().getSimpleName();
	}

	/** The average precision over all classes */
	public static double getAveragePrecision(Evaluation eval, Instances test) {
		double res = 0;
		int classes = test.classAttribute().numValues();
		for (int i = 0; i < classes; i++) {
			res += eval.precision(i);
		}
		return res * 100 / classes;
	}

	/** The average recall over all classes */
	public static double getAverageRecall(Evaluation eval, Instances test) {
		double res = 0;
		int classes = test.classAttribute().numValues();
		for (int i = 0; i < classes; i++) {
			res += eval.recall(i);
		}
		return res * 100 / classes;
	}

	/** The average recall over all classes */
	public static double getAverageFMeasure(Evaluation eval, Instances test) {
		double res = 0;
		int classes = test.classAttribute().numValues();
		for (int i = 0; i < classes; i++) {
			res += eval.fMeasure(i);
		}
		return res * 100 / classes;
	}

	public static List<Classifier> buildClassifiers() {
		List<Classifier> classifiers = new ArrayList<>();
		classifiers.add(MLAssignmentUtils.buildDecisionTree(true));
		classifiers.add(MLAssignmentUtils.buildBoosting());
		classifiers.add(MLAssignmentUtils.buildNeuralNet());
		classifiers.add(MLAssignmentUtils.buildKNearestNeibor());
		//classifiers.add(MLAssignmentUtils.buildLibSVM(KernelFunction.Quadratic));
		classifiers.add(MLAssignmentUtils.buildSMOSVM(KernelFunction.Quadratic));
		return classifiers;
	}

	public static Instances removeUnwantedClasses(Instances dataSet, String nominalIndexesToKeep) throws Exception {
		RemoveWithValues filter = new RemoveWithValues();
		filter.setInputFormat(dataSet);
		filter.setAttributeIndex(String.valueOf(dataSet.numAttributes() - 1));
		filter.setInvertSelection(true);
		return Filter.useFilter(dataSet, filter);

	}

}
