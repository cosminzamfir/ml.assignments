package ml.assignments;

import java.awt.BasicStroke;
import java.awt.Color;
import java.util.List;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

public class GeneralChart extends ApplicationFrame {

	private static final long serialVersionUID = 1L;

	/**
	 * @param title the chart title
	 * @param observations List of 'observations'. Each 'observations' consists of n arrays of [x,y]. For each 'observations' there is one graph line
	 * @param titles the titles of each observation 
	 */
	public GeneralChart(String title, List<double[][]> observations, List<String> titles, String xAxis, String yAxis) {
		super(title);

		final XYSeriesCollection dataSet = new XYSeriesCollection();
		for (int i = 0; i < observations.size(); i++) {
			final XYSeries series = new XYSeries(titles.get(i));
			double[][] points = observations.get(i);
			for (int j = 0; j < points.length; j++) {
				series.add(points[j][0], points[j][1]);
			}
			dataSet.addSeries(series);
		}

		//the chart
		final JFreeChart chart = ChartFactory.createXYLineChart(title, xAxis, yAxis, dataSet);
		final ChartPanel chartPanel = new ChartPanel(chart);
		customize(chart);
		chartPanel.setPreferredSize(new java.awt.Dimension(700, 370));
		setContentPane(chartPanel);
		pack();
		RefineryUtilities.centerFrameOnScreen(this);
		setVisible(true);
	}

	private void customize(JFreeChart chart) {
		chart.setBackgroundPaint(Color.white);

		//final StandardLegend legend = (StandardLegend) chart.getLegend();
		//legend.setDisplaySeriesShapes(true);

		// get a reference to the plot for further customisation...
		final XYPlot plot = chart.getXYPlot();
		plot.setBackgroundPaint(Color.lightGray);
		plot.setDomainGridlinePaint(Color.white);
		plot.setRangeGridlinePaint(Color.white);

		final XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		renderer.setBaseStroke(new BasicStroke(4));
		plot.setRenderer(renderer);

		// change the auto tick unit selection to integer units only...
		final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
		rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
		// OPTIONAL CUSTOMISATION COMPLETED.
	}
}
