package ml.assignments.assignment1;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import ml.assignments.CommandLineOptions;
import ml.assignments.MLAssignmentUtils;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class ClassifierTuner {

    protected double trainingPerc = 0.7;
    protected double testPec = 0.3;
    protected Instances dataSet;
    protected Instances training;
    protected Instances test;
    protected Classifier classifier;
    protected TreeMap<Double, String> results = new TreeMap<>(); //map errorRate to Description
    
    public void initialize(CommandLineOptions options) throws Exception {
            dataSet = MLAssignmentUtils.buildInstancesFromResource(options.getDataSetName());
            MLAssignmentUtils.shufle(dataSet);
        //            training = new Instances(dataSet, 0, (int) (dataSet.size() * trainingPerc));
        //            test = new Instances(dataSet, training.size(), dataSet.size() - training.size());
            training = new Instances(dataSet, 0, options.getTrainingSize());
            test = new Instances(dataSet, dataSet.size() - options.getTestSize(), options.getTestSize());
    }
    
    public double singleRun(String description) throws Exception {
        classifier.buildClassifier(training);
        Evaluation evaluation = new Evaluation(training);
        evaluation.evaluateModel(classifier, test);
        double errorRate = evaluation.pctIncorrect();
        System.out.println(classifier.getClass().getSimpleName() + ":" + description  + " - errorRate:" + errorRate);
        results.put(errorRate, description);
        return errorRate;
    }
    
    public Entry<Double, String> getBestResult() {
        return results.firstEntry();
    }
    
}
