package ml.assignments.assignment2.tsp;

import opt.OptimizationAlgorithm;

public class TSPMaxAllocatedTimeTest {

	static boolean RHC = false;
	static boolean SA = false;
	static boolean GA = false;
	static boolean MIMIC = true;

	public static void main(String[] args) {

		if(RHC)
		new TSPMaxAllocatedTime() {

			@Override
			public OptimizationAlgorithm getOptimizationAlgorithm(double[][] points) {
				return TravelingSalesmanUtils.buildRHC(points);
			}
		}.run();
	
		if(SA)
		new TSPMaxAllocatedTime() {
		
			@Override
			public OptimizationAlgorithm getOptimizationAlgorithm(double[][] points) {
				return TravelingSalesmanUtils.buildSA(points);
			}
		}.run();

		
		if(GA)
		new TSPMaxAllocatedTime() {
			
			@Override
			public OptimizationAlgorithm getOptimizationAlgorithm(double[][] points) {
				return TravelingSalesmanUtils.buildGA(points);
			}
		}.run();

		
		if(MIMIC)
		new TSPMaxAllocatedTime() {
			
			@Override
			public OptimizationAlgorithm getOptimizationAlgorithm(double[][] points) {
				return TravelingSalesmanUtils.buildMIMIC(points);
			}
		}.run();

	}



}
