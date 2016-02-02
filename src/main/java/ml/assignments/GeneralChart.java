package ml.assignments;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.util.List;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.FastScatterPlot;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.chart.title.TextTitle;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;
import org.jfree.util.ShapeUtilities;

import weka.core.Instances;

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
		Font titleFont = JFreeChart.DEFAULT_TITLE_FONT;
		titleFont = new Font(titleFont.getName(), titleFont.getStyle(), 14);
		chart.setTitle(new TextTitle(title, titleFont));
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
		for (int i = 0; i < plot.getSeriesCount(); i++) {
			renderer.setSeriesShapesVisible(i, false);
		}

		// change the auto tick unit selection to integer units only...
		final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
		rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
		// OPTIONAL CUSTOMISATION COMPLETED.
	}

	/** ================================================================================================== */

	/**
	 * Only data set with 2 dimensional real input values
	 * @param title
	 * @param instances
	 */
	public GeneralChart(String title, Instances instances) {
		super(title);

		final XYSeriesCollection dataSet = new XYSeriesCollection();

		final XYSeries series0 = new XYSeries("Class0");
		addToSeries(series0, instances, "0");

		final XYSeries series1 = new XYSeries("Class1");
		addToSeries(series1, instances, "1");

		dataSet.addSeries(series0);
		dataSet.addSeries(series1);

		//the chart
		final JFreeChart chart = ChartFactory.createXYLineChart(title, "X", "Y", dataSet);
		Font titleFont = JFreeChart.DEFAULT_TITLE_FONT;
		titleFont = new Font(titleFont.getName(), titleFont.getStyle(), 14);
		chart.setTitle(new TextTitle(title, titleFont));
		final ChartPanel chartPanel = new ChartPanel(chart);
		customizeScatered(chart);
		chartPanel.setPreferredSize(new java.awt.Dimension(500, 500));
		setContentPane(chartPanel);
		pack();
		RefineryUtilities.centerFrameOnScreen(this);
		setVisible(true);
	}

	private void customizeScatered(JFreeChart chart) {
		chart.setBackgroundPaint(Color.white);

		//final StandardLegend legend = (StandardLegend) chart.getLegend();
		//legend.setDisplaySeriesShapes(true);

		// get a reference to the plot for further customisation...
		final XYPlot plot = chart.getXYPlot();
		plot.setBackgroundPaint(Color.lightGray);
		plot.setDomainGridlinePaint(Color.white);
		plot.setRangeGridlinePaint(Color.white);

		final XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer(false, true);
		plot.setRenderer(renderer);
		renderer.setSeriesShape(0, ShapeUtilities.createDiamond(1));
		renderer.setSeriesShape(1, ShapeUtilities.createDiamond(1));

		// change the auto tick unit selection to integer units only...
		final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
		rangeAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
		// OPTIONAL CUSTOMISATION COMPLETED.
	}

	private void addToSeries(XYSeries series, Instances instances, String filter) {
		for (int i = 0; i < instances.size(); i++) {
			String classValue = instances.get(i).classValue() == 0 ? "0" : "1";
			if (!classValue.equals(filter)) {
				continue;
			}
			series.add(instances.get(i).value(0), instances.get(i).value(1));
		}
	}

}
