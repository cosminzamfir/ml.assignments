package ml.assignments;

import java.util.ArrayList;
import java.util.List;

import ml.assignments.assignment1.SVMTests.KernelFunction;
import weka.classifiers.Classifier;

public class CommandLineOptions {

	public static String CLASSIIFIER = "-c";
	public static String PRUNING = "-pruning";
	public static String LEARNING_RATE = "-learningRate";
	public static String MOMENTUM = "-momentum";
	public static String KERNEL_FUNCTION = "-kernel";
	public static String MAX_ITERATINS = "-maxIterations";
	public static String HIDDEN_UNITS = "-hiddenUnits";
	public static String K_NEIGHBORS = "-k";
	public static String RUNS= "-runs";
	public static String INITIAL_SIZSE = "-initialSize";
	public static String STEP_SIZE = "-stepSize";
	public static String DATA_SET_FILE = "-dataSet";

	
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

	public static CommandLineOptions newInstance(String[] args) {
		if (args.length % 2 == 1) {
			throw new RuntimeException("key - value pairs expected. This requires even number of args");
		}
		CommandLineOptions res = new CommandLineOptions();
		for (int i = 0; i < args.length; i = i + 2) {
			res.options.add(new KeyValue(args[i], args[i + 1]));
		}
		return res;
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
	
	public Classifier getClassifier() {
		String value = getValue("-c");
		if(value.equalsIgnoreCase("dt")) {
			return MLAssignmentUtils.buildDecisionTree(true);
		} 
		if(value.equalsIgnoreCase("knn")) {
			return MLAssignmentUtils.buildKNearestNeibor();
		}
		if(value.equalsIgnoreCase("ann")) {
			return MLAssignmentUtils.buildNeuralNet();
		}
		if(value.equalsIgnoreCase("libsvn")) {
			return MLAssignmentUtils.buildLibSVM(KernelFunction.Quadratic);
		}
		if(value.equalsIgnoreCase("smo")) {
			return MLAssignmentUtils.buildSMOSVM(KernelFunction.Quadratic);
		}
		if(value.equalsIgnoreCase("boost")) {
			return MLAssignmentUtils.buildBoosting();
		}
		throw new RuntimeException("Unknown classifier symbol. Supported: dt,knn,ann,libsvn,smo,boost");
	}
	
	public int getKNeibors() {
		return getIntValue(K_NEIGHBORS, K_NEIGHBORS_DEF);
	}
	
	public double getLearningRate() {
		return getDoubleValue(LEARNING_RATE, LEARNING_RATE_DEF);
	}
	
	public String getHiddenUnits() {
		return getValue(HIDDEN_UNITS, HIDDEN_UNITS_DEF);
	}
	
	public int getRuns(int defaultValue) {
		return getIntValue(RUNS, defaultValue);
	}
	public int getStepSize(int defaultValue) {
		return getIntValue(STEP_SIZE, defaultValue);
	}
	public int getInitialSize(int defaultValue) {
		return getIntValue(INITIAL_SIZSE, defaultValue);
	}
	
	public String getDataSetName(String defaultValue) {
		return getValue(DATA_SET_FILE, defaultValue);
	}


}
