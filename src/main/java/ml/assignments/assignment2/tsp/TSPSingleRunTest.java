package ml.assignments.assignment2.tsp;

import opt.OptimizationAlgorithm;

public class TSPSingleRunTest {

	public static void main(String[] args) {

		new TSPSingleRun() {

			@Override
			public OptimizationAlgorithm getOptimizationAlgorithm(double[][] points) {
				return TravelingSalesmanUtils.buildRHC(points);
			}
		}.run();
	
		new TSPSingleRun() {
		
			@Override
			public OptimizationAlgorithm getOptimizationAlgorithm(double[][] points) {
				return TravelingSalesmanUtils.buildSA(points);
			}
		}.run();

		
		new TSPSingleRun() {
			
			@Override
			public OptimizationAlgorithm getOptimizationAlgorithm(double[][] points) {
				return TravelingSalesmanUtils.buildGA(points);
			}
		}.run();

		
		new TSPSingleRun() {
			
			@Override
			public OptimizationAlgorithm getOptimizationAlgorithm(double[][] points) {
				return TravelingSalesmanUtils.buildMIMIC(points);
			}
		}.run();

	}



}
