package ml.assignments;

import ml.model.function.MultivariableFunction;

public class Function1 extends MultivariableFunction {

	@Override
	public double evaluate(double... x) {
		return Math.pow(x[0], 2) + Math.pow(x[1], 2);
		//return Math.sin(10 * x[0]) + Math.cos(10 * x[1]);
	}
}
