package ml.assignments.assignment1;

import java.util.Date;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

/**
 * Run a classifier algo on a given data set, given trainingSize and testSize and return the {@link Evaluation}
 * @author eh2zamf
 *
 */
public class ClassifierRunner {

	private Classifier classifier;
	private Evaluation evaluationOnTestSet;
	private Evaluation evaluationOnTrainingSet;

	public ClassifierRunner(Classifier classifier) {
		super();
		this.classifier = classifier;
	}
	
	public void run(Instances trainingDataSet, Instances testDataSet) throws Exception {
		buildModel(trainingDataSet);
		evaluationOnTestSet = evaluateModel(trainingDataSet, testDataSet);
		evaluationOnTrainingSet = evaluateModel(trainingDataSet, trainingDataSet);
	}
	
	public void buildModel(Instances trainingDataSet) throws Exception {
		trainingDataSet.setClassIndex(trainingDataSet.numAttributes() - 1);
		System.out.println(new Date() + ":" + "Buillding " + classifier.getClass().getName() + " classifier. Training size: " + trainingDataSet.size());
		classifier.buildClassifier(trainingDataSet);
		System.out.println(new Date() + ":" + "Done");
	}
	
	public Evaluation evaluateModel(Instances training, Instances test) throws Exception {
		System.out.println(new Date() + ":" + "Validating " + classifier.getClass().getName() + " classifier.");
		test.setClassIndex(test.numAttributes() - 1);
		Evaluation evaluation = new Evaluation(training);
		evaluation.evaluateModel(classifier, test);
		System.out.println(new Date() + ":" + "Done");
		return evaluation;
		
	}

	/**
	 * @param dataSet
	 * @param trainingSize - select first trainingSize examples from dataSet
	 * @param testingSize - select next testingSize examples, starting at trainingSize+1 position
	 * @throws Exception
	 */
	public void run(Instances dataSet, int trainingSize, int testingSize) throws Exception {
		Instances training = new Instances(dataSet, 0, trainingSize);
		Instances test = new Instances(dataSet, trainingSize + 1, trainingSize + 1 + testingSize);
		run(training, test);
	}

	public Evaluation getEvaluationOnTestSet() {
		return evaluationOnTestSet;
	}
	
	public Evaluation getEvaluationOnTrainingSet() {
		return evaluationOnTrainingSet;
	}
	
	public Classifier getClassifier() {
		return classifier;
	}
}
