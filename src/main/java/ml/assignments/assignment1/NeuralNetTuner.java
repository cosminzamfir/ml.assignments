package ml.assignments.assignment1;

import ml.assignments.CommandLineOptions;
import weka.classifiers.functions.MultilayerPerceptron;

import java.util.Map.Entry;

public class NeuralNetTuner extends ClassifierTuner {

    public MultilayerPerceptron ann;

    public NeuralNetTuner() {
        classifier = new MultilayerPerceptron();
        ann = (MultilayerPerceptron) classifier;
    }

    public void run(CommandLineOptions options) throws Exception {
        initialize(options);
        int hiddenUnits = 1;
        double momentum = 0;
        double learningRate = 0;

        for (int huIndex = 5; huIndex < dataSet.numAttributes(); huIndex++) {
            for (int momIndex = 0; momIndex < 5; momIndex++) {
                for (int lrIndex = 1; lrIndex < 5; lrIndex++) {
                    hiddenUnits = huIndex;
                    momentum = momIndex * 0.1;
                    learningRate = lrIndex * 0.1;
                    ann.setHiddenLayers(String.valueOf(hiddenUnits));
                    ann.setMomentum(momentum);
                    ann.setLearningRate(learningRate);
                    singleRun("hiddenUnits=" + hiddenUnits + ";momentum=" + momentum + ";learningRate=" + learningRate);
                }
            }
        }

    }

    public static void main(String[] args) throws Exception {
        NeuralNetTuner tuner = new NeuralNetTuner();
        CommandLineOptions options = CommandLineOptions.instance(args);
        tuner.run(options);
        Entry<Double, String> best = tuner.getBestResult();
        System.out.println("Best result: " + best.getKey() + " - " + best.getValue());
    }
}
