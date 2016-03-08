package ml.assignments.assignment2.tsp;

import java.util.Random;

import ml.assignments.assignment2.ExecutionResult;
import ml.assignments.assignment2.ExecutionResults;
import ml.assignments.assignment2.MaxIterationsTrainer;
import opt.OptimizationAlgorithm;

public abstract class TSPSingleRun {

	private static final int startN = 5;
	private static final int endN = 150;
	private static final int step = 5;
	private static final int TRIALS = 3;

	public void run() {
		System.out.println(getDescription());
		System.out.println("N,BestDistance,TimeToBestDistance,IterationsToBestDistance");
		for (int n = startN; n <= endN; n = n + step) {
			ExecutionResults results = new ExecutionResults();
			for (int i = 0; i < TRIALS; i++) {
				results.addResult(runOnce(n));
			}
			System.out.println(n + "," + 1 / results.getOptimalFitness() + "," + results.getTimeToBestIteration() + "," + results.getBestIteration());
		}
	}

	public abstract OptimizationAlgorithm getOptimizationAlgorithm(double[][] points);

	public String getDescription() {
		return getOptimizationAlgorithm(generatePoints(10)).getClass().getSimpleName();
	}

	public ExecutionResult runOnce(int n) {
		double[][] points = generatePoints(n);
		OptimizationAlgorithm oa = getOptimizationAlgorithm(points);
		MaxIterationsTrainer trainer = new MaxIterationsTrainer(oa, 0, 0, 0);
		trainer.train();
		return new ExecutionResult(trainer);
	}

	private double[][] generatePoints(int n) {
		Random random = new Random();
		// create the random points
		double[][] points = new double[n][2];
		for (int i = 0; i < points.length; i++) {
			points[i][0] = random.nextDouble();
			points[i][1] = random.nextDouble();
		}
		return points;
	}
}
