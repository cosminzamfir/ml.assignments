package ml.assignments.assignment3;

import func.KMeansClusterer;
import shared.DataSet;
import shared.Instance;
import ml.assignments.MLAssignmentUtils;

public class AbagailKMeansTest {

	public static void main(String[] args) {
		Instance[] instances = MLAssignmentUtils.initializeRobotDataSet(5400);
		KMeansClusterer kmeans = new KMeansClusterer();
		DataSet dataSet = new DataSet(instances);
		kmeans.estimate(dataSet);
		System.out.println(kmeans);
	}
}
