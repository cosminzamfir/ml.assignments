package ml.assignments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.lang.reflect.Field;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;

import shared.Instance;
import ml.assignments.CommandLineOptions.Option;
import ml.assignments.assignment1.SVMTests.KernelFunction;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.MultilayerPerceptronTanh;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.neural.NeuralConnection;
import weka.classifiers.functions.supportVector.CachedKernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.Saver;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class MLAssignmentUtils {

	public static final String ROOT_FOLDER = "C:/work/data/workspace/ml.assignments/";

	public static Instances buildInstancesFromResource(String resourceName) {
		Reader r;
		try {
			InputStream is = Thread.currentThread().getContextClassLoader().getResourceAsStream(resourceName);
			if(is == null) {
				throw new RuntimeException("Non existing resource file: " + resourceName);
			}
			r = new java.io.BufferedReader(new java.io.InputStreamReader(is));
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
		if (CommandLineOptions.getInstance() != null && CommandLineOptions.getInstance().getClassIndex().equals("first")) {
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
			saver.setFile(new File(ROOT_FOLDER + "src/main/resources/" + fileName));
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
		res.setConfidenceFactor(options.getConfidenceFactor());
		return res;
	}

	public static IBk buildKNearestNeibor(CommandLineOptions options) {
		IBk res = new IBk();
		res.setKNN(options.getKNeibors());
		int distanceWeight = options.getDistanceWeight();
		res.setDistanceWeighting(new SelectedTag(distanceWeight, IBk.TAGS_WEIGHTING));
		return res;
	}

	public static AdaBoostM1 buildBoosting(CommandLineOptions options) {
		AdaBoostM1 res = new AdaBoostM1();
		res.setClassifier(options.getBaseLearner());
		return res;
	}

	public static MultilayerPerceptron buildNeuralNet(CommandLineOptions options) {
		String hiddenUnits = options.getHiddenUnits();
		String activationFunction = options.getActivationFunction();
		if (activationFunction.equalsIgnoreCase("sigmoid")) {
			MultilayerPerceptron res = new MultilayerPerceptron();
			res.setHiddenLayers(hiddenUnits);
			res.setLearningRate(options.getLearningRate());
			res.setMomentum(options.getMomentum());
			return res;
		}
		if (activationFunction.equalsIgnoreCase("tanh")) {
			MultilayerPerceptronTanh res = new MultilayerPerceptronTanh();
			res.setHiddenLayers(hiddenUnits);
			res.setLearningRate(options.getLearningRate());
			res.setMomentum(options.getMomentum());
			return res;
		}
		throw new RuntimeException("Invalid activation function value. Only " + Option.ACTIVATION_FUNCTION.usage() + " are supported");
	}

	public static LibSVM buildLibSVM(KernelFunction function, CommandLineOptions options) {
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
			svm.setGamma(options.getGamma());
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
			kernel.setGamma(options.getGamma());
			smo.setKernel(kernel);
		} else if (function == KernelFunction.Sigmoid) {
			throw new RuntimeException(KernelFunction.Sigmoid + " not supported by " + smo.getClass().getName());
		}
		return smo;
	}

	public static LibSVM buildLibSVM(CommandLineOptions options) {
		return buildLibSVM(KernelFunction.Quadratic, options);
	}

	public static SMO buildSMOSVM(CommandLineOptions options) {
		return buildSMOSVM(KernelFunction.Quadratic, options);
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

	public static String toString(MultilayerPerceptronTanh classifier) {
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
		if (classifier instanceof MultilayerPerceptronTanh) {
			return toString((MultilayerPerceptronTanh) classifier);
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
		classifiers.add(MLAssignmentUtils.buildLibSVM(options));
		classifiers.add(MLAssignmentUtils.buildSMOSVM(options));
		return classifiers;
	}

	public static Instances removeUnwantedClasses(Instances dataSet, String nominalIndexesToKeep) throws Exception {
		RemoveWithValues filter = new RemoveWithValues();
		filter.setInputFormat(dataSet);
		filter.setAttributeIndex(String.valueOf(dataSet.numAttributes() - 1));
		filter.setInvertSelection(true);
		return Filter.useFilter(dataSet, filter);

	}

	public static DecisionStump buildDecisionStump(CommandLineOptions commandLineOptions) {
		return new DecisionStump();
	}

	public static Instances keepOnlyAttributes(Instances dataSet, String attributesToKeep) {
		try {
			Remove filter = new Remove();
			filter.setInvertSelection(true);
			filter.setAttributeIndices(attributesToKeep);
			filter.setInputFormat(dataSet);
			return Filter.useFilter(dataSet, filter);
		} catch (Exception e) {
			throw new RuntimeException("Error keeping attributes " + attributesToKeep, e);
		}
	}

	public static Instances removeAttributes(Instances dataSet, String attributesToRemove) {
		try {
			Remove filter = new Remove();
			filter.setInvertSelection(false);
			filter.setAttributeIndices(attributesToRemove);
			filter.setInputFormat(dataSet);
			return Filter.useFilter(dataSet, filter);
		} catch (Exception e) {
			throw new RuntimeException("Error removing attributes " + attributesToRemove, e);
		}
	}

	public static void writeToFile(String fileName, String s, boolean append) {
		try {
			File file = new File(fileName);
			file.createNewFile();
			FileWriter writer = new FileWriter(new File(fileName), append);
			writer.write(s + "\n");
			writer.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}

	public static int getStartN() {
		String s = System.getProperty("startN");
		return s == null ? 10: Integer.valueOf(s);
	}

	public static int getEndN() {
		String s = System.getProperty("endN");
		return s == null ? 100 : Integer.valueOf(s);
	}

	public static int getStep() {
		String s = System.getProperty("step");
		return s == null ? 5 : Integer.valueOf(s);
	}

	public static int getTrials() {
		String s = System.getProperty("trials");
		return s == null ? 3 : Integer.valueOf(s);
	}
	
	public static String toString(double[] array) {
		StringBuilder sb = new StringBuilder();
		for (double d : array) {
			sb.append(NumberFormat.getInstance().format(d)).append(" ");
		}
		return sb.toString();
	}

	public static double average(HashMap<String, Double> map) {
		double res = 0;
		for (Double d: map.values()) {
			res += d;
		}
		return res/map.size();
	}

	public static double countIf(int[] clusterAssignments, int equalTo) {
		int res = 0;
		for (int i : clusterAssignments) {
			if(i == equalTo) {
				res ++;
			}
		}
		return res;
	}
	
	
    public static Instance[] initializeInstances(int size, String filePath, int nrAttributes, List<String> labels ) {
    	
    	int nrLabels = labels.size();
        double[][][] attributes = new double[size][][];

        try {
        	InputStream is = Thread.currentThread().getContextClassLoader().getResourceAsStream(filePath);
            BufferedReader br = new BufferedReader(new InputStreamReader(is));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[nrAttributes]; 
                attributes[i][1] = new double[nrLabels];

                for(int j = 0; j < nrAttributes; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                String label = scan.next();
                for (int j = 0; j < nrLabels; j++) {
                	if (j == labels.indexOf(label)) {
                		attributes[i][1][j] = 1;
                	} else {
                		attributes[i][1][j] = 0;
                	}
                }
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1]));
        }

        return instances;
    }


}
