package ml.assignments;


import java.util.ArrayList;
import java.util.List;

import ml.assignments.assignment1.SVMTests.KernelFunction;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;

public class CommandLineOptions {

	private static enum Option {
		
		CLASS_INDEX("-classIndex", "last|first", "What attribute is the class index?"),
		CLASSIIFIER("-c", "Classifier", "<dt|knn|ann|libsvm|smo|boost>"),
		CLASSIIFIERS("-cs", "Classifiers", "one or more of <dt|knn|ann|libsvm|smo|boost>"),
		BASE_LEARNER("-baseLearner", "The base learner for Boosting", "<dt|knn|ann|libsvm|smo|boost>"),
		PRUNING("-pruning", "Use pruning for decision tree?", "<true|false>"),
		LEARNING_RATE("-learningRate", "For neural nets", "numeric"),
		MOMENTUM("-momentum", "For neural nets", "numeric"),
		KERNEL_FUNCTION("-kernel", "The kernel function", "<Liniar|Quadratic|Cubic|Radial|Sigmoid>"),
		MAX_ITERATINS("-maxIterations", "For boosting", "numeric"),
		HIDDEN_UNITS("-hiddenUnits", "For neural nets", "Comma delimited string of numbers. Each number -> one hidden layer with so many nodes"),
		K_NEIGHBORS("-kn", "The size of k-neighbours", "numeric"),
		RUNS("-runs", "The number of runs, for looping, e.g. for different training size", "numeric"),
		INITIAL_SIZE("-initialSize", "The initial size, for looping, e.g. for different training size", "numeric"),
		STEP_SIZE("-stepSize", "The step size, for looping, e.g. for different training size", "numeric"),
		DATA_SET_FILE("-dataSet", "The name of the data set file. Must be in the classpath", "string"),
		TEST_SIZE("-testSize", "The size of the test data set", "numeric"),
		TRAINING_SIZE("-trainingSize", "The size of the training data set", "numeric"),
		DISTANCE_WEIGHT("-distanceWeight", "The distance weighting function for KNN", "<" + IBk.WEIGHT_NONE + "(none)|" + IBk.WEIGHT_INVERSE + "(inverse)|" + IBk.WEIGHT_SIMILARITY+ "(similarity)|"),
		CROSS_VALIDATE("-crossValidate", "Perform cross validation?. Default true", "true|false"),
		HELP("-help", "Help !", "no params");

		String key;
		String description;
		String usage;
		
		private Option(String key, String description, String values) {
			this.key = key;
			this.description = description;
			this.usage = values;
		}
		
		public String key() {
			return key;
		}
		
		public String description() {
			return description;
		}
		
		public String usage() {
			return usage;
		}
		
	}
	public static boolean PRUNING_DEF = true;
	public static double LEARNING_RATE_DEF = 0.3;
	public static double MOMENTUM_DEF = 0.02;
	public static KernelFunction KERNEL_FUNCTION_DEF = KernelFunction.Quadratic;
	public static int MAX_ITERATINS_DEF = 10;
	public static String HIDDEN_UNITS_DEF = "10";
	public static int K_NEIGHBORS_DEF = 10;

	public static int RUNS_DEF = 50;
	public static int INITIAL_SIZSE_DEF = 100;
	public static int STEP_SIZE_DEF = 40;
	public static String DATA_SET_FILE_DEF = "robot-moves.arff";

	private static CommandLineOptions instance = null;
	private List<KeyValue> options = new ArrayList<>();

	private static class KeyValue {

		public KeyValue(String key, String value) {
			super();
			this.key = key;
			this.value = value;
		}

		private String key;
		private String value;
	}

	public static CommandLineOptions instance(String[] args) {
		if(instance != null) {
			return instance;
		}
		if(args.length == 1 && args[0].equals(Option.HELP.key())) {
			printUsage();
			System.exit(0);
		}
		if (args.length % 2 == 1) {
			throw new RuntimeException("key - value pairs expected. This requires even number of args");
		}
		CommandLineOptions res = new CommandLineOptions();
		for (int i = 0; i < args.length; i = i + 2) {
			String key = args[i];
			String value = args[i+1];
			checkKey(key);
			res.options.add(new KeyValue(key, value));
		}
		instance = res;
		return instance;
	}
	
	public static CommandLineOptions getInstance() {
		return instance;
	}

	private static void checkKey(String key) {
		for (Option option : Option.values()) {
			if(option.key().equalsIgnoreCase(key)) {
				return;
			}
		}
		throw new RuntimeException("Unknown option: " + key);
	}

	public boolean hasOption(String key) {
		for (KeyValue keyValue : options) {
			if (keyValue.key.equals(key)) {
				return true;
			}
		}
		return false;
	}

	public String getValue(String key) {
		for (KeyValue keyValue : options) {
			if (keyValue.key.equals(key)) {
				return keyValue.value;
			}
		}
		throw new RuntimeException("Option " + key + " not present");
	}

	public String getValue(String key, String defaultValue) {
		for (KeyValue keyValue : options) {
			if (keyValue.key.equals(key)) {
				return keyValue.value;
			}
		}
		return defaultValue;
	}

	public Integer getIntValue(String key) {
		for (KeyValue keyValue : options) {
			if (keyValue.key.equals(key)) {
				try {
					return Integer.valueOf(keyValue.value);
				} catch (Exception e) {
					throw new RuntimeException("Invalid format for integer option " + key + ": " + keyValue.value);
				}
			}
		}
		throw new RuntimeException("Option " + key + " not present");
	}

	public Integer getIntValue(String key, int defaultValue) {
		for (KeyValue keyValue : options) {
			if (keyValue.key.equals(key)) {
				try {
					return Integer.valueOf(keyValue.value);
				} catch (Exception e) {
					throw new RuntimeException("Invalid format for integer option " + key + ": " + keyValue.value);
				}
			}
		}
		return defaultValue;
	}

	public Double getDoubleValue(String key) {
		for (KeyValue keyValue : options) {
			if (keyValue.key.equals(key)) {
				try {
					return Double.valueOf(keyValue.value);
				} catch (Exception e) {
					throw new RuntimeException("Invalid format for double option " + key + ": " + keyValue.value);
				}
			}
		}
		throw new RuntimeException("Option " + key + " not present");
	}

	public Double getDoubleValue(String key, double defaultValue) {
		for (KeyValue keyValue : options) {
			if (keyValue.key.equals(key)) {
				try {
					return Double.valueOf(keyValue.value);
				} catch (Exception e) {
					throw new RuntimeException("Invalid format for double option " + key + ": " + keyValue.value);
				}
			}
		}
		return defaultValue;
	}

	public boolean getBooleanValue(String key) {
		for (KeyValue keyValue : options) {
			if (keyValue.key.equals(key)) {
				try {
					return Boolean.valueOf(keyValue.value);
				} catch (Exception e) {
					throw new RuntimeException("Invalid format for boolean option " + key + ": " + keyValue.value);
				}
			}
		}
		throw new RuntimeException("Option " + key + " not present");
	}

	public boolean getBooleanValue(String key, boolean defaultValue) {
		for (KeyValue keyValue : options) {
			if (keyValue.key.equals(key)) {
				try {
					return Boolean.valueOf(keyValue.value);
				} catch (Exception e) {
					throw new RuntimeException("Invalid format for boolean option " + key + ": " + keyValue.value);
				}
			}
		}
		return defaultValue;
	}

	public Classifier getClassifier() {
		String value = getValue("-c");
		return builClassifier(value);
	}
	
	public List<Classifier> getClassifiers(List<Classifier> defaultValue) {
		if(!hasOption(Option.CLASSIIFIERS.key())) {
			return defaultValue;
		}
		String s = getValue(Option.CLASSIIFIERS.key());
		String[] tokens = s.split(",");
		List<Classifier> res = new ArrayList<>();
		for (String token : tokens) {
			res.add(builClassifier(token));
		}
		return res;
	}

	public Classifier getBaseLearner(Classifier defaultValue) {
		if (!hasOption(Option.BASE_LEARNER.key())) {
			return defaultValue;
		}
		String value = getValue("-bl");
		return builClassifier(value);
	}

	private Classifier builClassifier(String value) {
		if (value.equalsIgnoreCase("dt")) {
			return MLAssignmentUtils.buildDecisionTree(this);
		}
		if (value.equalsIgnoreCase("knn")) {
			return MLAssignmentUtils.buildKNearestNeibor(this);
		}
		if (value.equalsIgnoreCase("ann")) {
			return MLAssignmentUtils.buildNeuralNet(this);
		}
		if (value.equalsIgnoreCase("libsvn")) {
			return MLAssignmentUtils.buildLibSVM(getKernelFunction(KernelFunction.Quadratic));
		}
		if (value.equalsIgnoreCase("smo")) {
			return MLAssignmentUtils.buildSMOSVM(getKernelFunction(KernelFunction.Quadratic));
		}
		if (value.equalsIgnoreCase("boost")) {
			return MLAssignmentUtils.buildBoosting(this);
		}
		throw new RuntimeException("Unknown classifier symbol. Supported: dt,knn,ann,libsvn,smo,boost");
	}

	public int getKNeibors(int defaultValue) {
		return getIntValue(Option.K_NEIGHBORS.key(), defaultValue);
	}

	public double getLearningRate(double defaultValue) {
		return getDoubleValue(Option.LEARNING_RATE.key(), defaultValue);
	}

	public String getHiddenUnits(String defaultValue) {
		return getValue(Option.HIDDEN_UNITS.key(), defaultValue);
	}

	public int getRuns(int defaultValue) {
		return getIntValue(Option.RUNS.key(), defaultValue);
	}

	public int getStepSize(int defaultValue) {
		return getIntValue(Option.STEP_SIZE.key(), defaultValue);
	}

	public int getInitialSize(int defaultValue) {
		return getIntValue(Option.INITIAL_SIZE.key(), defaultValue);
	}

	public int getTestSize(int defaultValue) {
		return getIntValue(Option.TEST_SIZE.key(), defaultValue);
	}

	public int getTrainingSize() {
		return getIntValue(Option.TRAINING_SIZE.key());
	}

	public int getTrainingSize(int defaultValue) {
		return getIntValue(Option.TRAINING_SIZE.key(), defaultValue);
	}

	public String getDataSetName() {
		return getValue(Option.DATA_SET_FILE.key());
	}

	public String getDataSetName(String defaultValue) {
		return getValue(Option.DATA_SET_FILE.key(), defaultValue);
	}

	public boolean isPruning(boolean defaultValue) {
		return getBooleanValue(Option.PRUNING.key(), defaultValue);
	}

	public int getDistanceWeight(int defaultValue) {
		return getIntValue(Option.DISTANCE_WEIGHT.key(), defaultValue);
	}
	
	public boolean crossValidate()	{
		return getBooleanValue(Option.CROSS_VALIDATE.key(), true);
	}
	
	public KernelFunction getKernelFunction(KernelFunction defaultValue) {
		if(!hasOption(Option.KERNEL_FUNCTION.key())) {
			return KernelFunction.Quadratic;
		} 
		
		String value = getValue(Option.KERNEL_FUNCTION.key());
		if(KernelFunction.valueOf(value) == null) {
			throw new RuntimeException("Unknown kernel function. Supported values: " + KernelFunction.values());
		}
		return KernelFunction.valueOf(getValue(Option.KERNEL_FUNCTION.key()));
	}

	public static void printUsage() {
		System.out.println("Command line options:");
		for (Option option : Option.values()) {
			p(option.key(), option.description(), option.usage());
		}
	}
	
	public String getClassIndex() {
		return getValue(Option.CLASS_INDEX.key(), "last");
	}

	private static void p(String option, String description, String values) {
		System.out.println("        " + option + " - " + description + " - " + values);
	}
}
