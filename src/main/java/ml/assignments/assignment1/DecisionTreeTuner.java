package ml.assignments.assignment1;

import ml.assignments.CommandLineOptions;

import java.util.Map.Entry;

import weka.classifiers.trees.J48;

public class DecisionTreeTuner extends ClassifierTuner {

    public J48 j48;

    public DecisionTreeTuner() {
        classifier = new J48();
        j48 = (J48) classifier;
    }

    public void run(CommandLineOptions options) throws Exception {

        initialize(options);
        boolean pruning = false;
        j48.setUnpruned(!pruning);
        singleRun("pruning=false");

        //lower confidence, more pruning: http://ww.samdrazin.com/classes/een548/project2report.pdf. Default is 0.25
        pruning = true;
        j48.setUnpruned(!pruning);
        float confidence = 0;
        for (int i = 1; i < 5; i++) {
            confidence = (float) (i * 0.1);
            j48.setConfidenceFactor(confidence);
            singleRun("pruning=true;confidenceFactor=" + confidence);
        }
    }

    public static void main(String[] args) throws Exception {
        DecisionTreeTuner tuner = new DecisionTreeTuner();
        CommandLineOptions options = CommandLineOptions.instance(args);
        tuner.run(options);
        Entry<Double, String> best = tuner.getBestResult();
        System.out.println("Best result: " + best.getKey() + " - " + best.getValue());
    }
}
