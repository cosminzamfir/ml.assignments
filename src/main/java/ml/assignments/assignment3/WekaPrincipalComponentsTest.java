package ml.assignments.assignment3;

import ml.assignments.CommandLineOptions;
import ml.assignments.MLAssignmentUtils;
import weka.attributeSelection.PrincipalComponents;
import weka.core.Instance;
import weka.core.Instances;

public class WekaPrincipalComponentsTest {

	public static void main(String[] args) throws Exception {
		CommandLineOptions options = CommandLineOptions.instance(args);
		PrincipalComponents pc = new PrincipalComponents();
		pc.setCenterData(true);
		Instances dataSet = MLAssignmentUtils.buildInstancesFromResource(options.getDataSetName()); 
		pc.setTransformBackToOriginal(true);
		pc.buildEvaluator(dataSet);
		System.out.println(pc);
		Instance instance = dataSet.get(0);
		System.out.println(instance);
		System.out.println(pc.convertInstance(instance));
	}
}
