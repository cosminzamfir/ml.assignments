package ml.assignments.assignment3;

import java.util.ArrayList;
import java.util.List;

import ml.assignments.MLAssignmentUtils;
import shared.DataSet;
import shared.Instance;
import shared.filt.PrincipalComponentAnalysis;
import util.linalg.Matrix;

/**
 * A class for testing
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class PrincipalComponentAnalysisTest {
    
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        List<String> labels = new ArrayList<String>();
        labels.add("Slight-Right-Turn");
        labels.add("Sharp-Right-Turn");
        labels.add("Move-Forward");
        labels.add("Slight-Left-Turn");
    	Instance[] instances =  MLAssignmentUtils.initializeInstances(4000, "robot-moves.txt", 24, labels);
        DataSet set = new DataSet(instances);
        System.out.println("Before PCA: the data set");
        //System.out.println(set);
        PrincipalComponentAnalysis filter = new PrincipalComponentAnalysis(set, 3);
        
        System.out.println("===============================================");
        System.out.println("Eigenvalues");
        System.out.println(filter.getEigenValues());
        
        
        System.out.println("===============================================");
        System.out.println("The projection matrix");
        System.out.println(filter.getProjection().transpose());
        
        filter.filter(set);
        System.out.println("===============================================");
        System.out.println("After PCA: the data set");
        //System.out.println(set);
        Matrix reverse = filter.getProjection().transpose();
        for (int i = 0; i < set.size(); i++) {
            Instance instance = set.get(i);
            instance.setData(reverse.times(instance.getData()).plus(filter.getMean()));
        }
        System.out.println("===============================================");
        System.out.println("After reconstructing");
        //System.out.println(set);
        
    }

}
