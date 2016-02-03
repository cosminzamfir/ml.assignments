package ml.assignments;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.ObjectInputStream.GetField;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import ml.assignments.assignment1.SVMTests.KernelFunction;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.neural.NeuralConnection;
import weka.classifiers.functions.supportVector.CachedKernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.Saver;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemoveWithValues;
import weka.gui.SimpleCLIPanel.CommandlineCompletion;

public class MLAssignmentUtils {

	public static Instances buildInstancesFromResource(String resourceName) {
		Reader r;
		try {
			r = new java.io.BufferedReader(new java.io.InputStreamReader(Thread.currentThread().getContextClassLoader().getResourceAsStream(resourceName)));
			Instances res = new Instances(r);
			setClassIndex(res);
			return res;
		} catch (Exception e) {
			throw new RuntimeException("", e);
		}
	}

	public static Instances buildInstancesFromFileName(String fileName) {
		Reader r;
		try {
			r = new java.io.BufferedReader(new java.io.FileReader(fileName));
			Instances res = new Instances(r);
			setClassIndex(res); 
			return res;
		} catch (Exception e) {
			throw new RuntimeException("", e);
		}
	}

	private static void setClassIndex(Instances res) {
		if(CommandLineOptions.getInstance() != null && CommandLineOptions.getInstance().getClassIndex().equals("first")) {
			res.setClassIndex(0);
		} else {
			res.setClassIndex(res.numAttributes() - 1);
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

	public static void write(String fileName, Instances dataSet) {
		try {
			ArffSaver saver = new ArffSaver();
			saver.setFile(new File("c:/data/dropbox/dropbox/omcs/ml/projects/ml.assignments/src/main/resources/" + fileName));
			saver.setRetrieval(Saver.BATCH);
			saver.setInstances(dataSet);
			saver.writeBatch();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
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

	public static J48 buildDecisionTree(CommandLineOptions options) {
		J48 res = new J48();
		boolean usePruning = options.isPruning(true);
		res.setUnpruned(!usePruning);
		return res;
	}

	public static IBk buildKNearestNeibor(CommandLineOptions options) {
		IBk res = new IBk();
		res.setKNN(options.getKNeibors(10));
		int distanceWeight = options.getDistanceWeight(IBk.WEIGHT_INVERSE);
		res.setDistanceWeighting(new SelectedTag(distanceWeight, IBk.TAGS_WEIGHTING));
		
		return res;
	}

	public static AdaBoostM1 buildBoosting(CommandLineOptions options) {
		AdaBoostM1 res = new AdaBoostM1();
		res.setClassifier(buildDecisionTree(options));
		return res;
	}

	public static MultilayerPerceptron buildNeuralNet(CommandLineOptions options) {
		MultilayerPerceptron res = new MultilayerPerceptron();
		res.setHiddenLayers(options.getHiddenUnits("a"));
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

	public static SMO buildSMOSVM(KernelFunction function, CommandLineOptions options) {
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
			kernel.setGamma(options.getGamma(1));
			smo.setKernel(kernel);
		} else if (function == KernelFunction.Sigmoid) {
			throw new RuntimeException(KernelFunction.Sigmoid + " not supported by " + smo.getClass().getName());
		}
		((CachedKernel) smo.getKernel()).setCacheSize(1);
		return smo;
	}

	public static String toString(J48 classifier) {
		return classifier.getClass().getSimpleName() + " - pruned=" + !classifier.getUnpruned();
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
				+ " - hiddenUnits=" + ((NeuralConnection[]) getField("m_neuralNodes", classifier)).length;
	}

	private static Object getField(String fieldName, Object o) {
		try {
			Field field = o.getClass().getDeclaredField(fieldName);
			field.setAccessible(true);
			return field.get(o);
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public static String toString(SMO classifier) {
		return classifier.getClass().getSimpleName() + " - kernel=" + classifier.getKernel();
	}

	public static String toString(LibSVM classifier) {
		return classifier.getClass().getSimpleName() + " - kernel=" + classifier.getKernelType().getSelectedTag().getReadable();
	}

	public static String toString(Classifier classifier) {
		if (classifier instanceof J48) {
			return toString((J48) classifier);
		}
		if (classifier instanceof MultilayerPerceptron) {
			return toString((MultilayerPerceptron) classifier);
		}
		if (classifier instanceof IBk) {
			return toString((IBk) classifier);
		}
		if (classifier instanceof LibSVM) {
			return toString((LibSVM) classifier);
		}
		if (classifier instanceof SMO) {
			return toString((SMO) classifier);
		}
		if (classifier instanceof AdaBoostM1) {
			return toString((AdaBoostM1) classifier);
		}

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

	public static List<Classifier> buildClassifiers(CommandLineOptions options) {
		List<Classifier> classifiers = new ArrayList<>();
		classifiers.add(MLAssignmentUtils.buildDecisionTree(options));
		classifiers.add(MLAssignmentUtils.buildBoosting(options));
		classifiers.add(MLAssignmentUtils.buildNeuralNet(options));
		classifiers.add(MLAssignmentUtils.buildKNearestNeibor(options));
		classifiers.add(MLAssignmentUtils.buildLibSVM(KernelFunction.Quadratic));
		classifiers.add(MLAssignmentUtils.buildSMOSVM(KernelFunction.Quadratic, options));
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
