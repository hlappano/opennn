//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//   Author: Harleen Lappano
//
//   TIME SERIES Example
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a time series example.

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

void RunSeries(std::string NameOfSeries)
{
	std::string DataPath("../data");

	std::cout << "OpenNN. " << NameOfSeries << " Example." << std::endl;
	std::srand(static_cast<unsigned>(std::time(nullptr)));

	// Data Set (no delimiter and no column names)
	OpenNN::DataSet DataSet(DataPath + "/" + NameOfSeries + ".csv", ',', false);
	Vector<Histogram> columns_histograms = DataSet.calculate_columns_histograms();


	std::cout << "Converting to time series" << std::endl;
	DataSet.set_lags_number(1);
	DataSet.set_steps_ahead_number(1);
	DataSet.set_time_index(0);
	DataSet.transform_time_series();

	//std::cout << "Correlations" << std::endl;
	// Missing Values
	DataSet.impute_missing_values_mean();

	// Instances
	DataSet.split_instances_sequential();

	const Vector<Descriptives> inputs_descriptives = DataSet.scale_inputs_minimum_maximum();
	const Vector<Descriptives> targets_descriptives = DataSet.scale_targets_minimum_maximum();

	std::cout << "Neural Network" << std::endl;
	const size_t inputs_number = DataSet.get_input_variables_number();
	const size_t hidden_perceptrons_number = 6;
	const size_t outputs_number = DataSet.get_target_variables_number();
	OpenNN::NeuralNetwork NeuralNetwork(OpenNN::NeuralNetwork::Forecasting, { inputs_number, hidden_perceptrons_number, outputs_number });
	// Setup layers
	OpenNN::ScalingLayer* ScalingLayerPtr = NeuralNetwork.get_scaling_layer_pointer();
	ScalingLayerPtr->set_descriptives(inputs_descriptives);
	ScalingLayerPtr->set_scaling_methods(ScalingLayer::NoScaling);
	OpenNN::UnscalingLayer* UnscalingLayerPtr = NeuralNetwork.get_unscaling_layer_pointer();
	UnscalingLayerPtr->set_descriptives(targets_descriptives);
	UnscalingLayerPtr->set_unscaling_method(UnscalingLayer::NoUnscaling);
	OpenNN::LongShortTermMemoryLayer* LTSMPtr = NeuralNetwork.get_long_short_term_memory_layer_pointer();
	// Set long term memory to 4 steps
	LTSMPtr->set_timesteps(4);

	//Training strategy
	OpenNN::TrainingStrategy TrainingStrategy(&NeuralNetwork, &DataSet);
	QuasiNewtonMethod* QuasiPtr = TrainingStrategy.get_quasi_Newton_method_pointer();
	QuasiPtr->set_maximum_epochs_number(10000);
	QuasiPtr->set_maximum_time(250);
	QuasiPtr->set_display_period(10);
	QuasiPtr->set_minimum_loss_decrease(0.0);
	QuasiPtr->set_reserve_training_error_history(true);
	QuasiPtr->set_reserve_selection_error_history(true);
	// Perform Training
	const OpenNN::OptimizationAlgorithm::Results TrainingResults = TrainingStrategy.perform_training();

	// Testing Analysis
	std::cout << "Testing Analysis" << std::endl;
	OpenNN::TestingAnalysis Analysis(&NeuralNetwork, &DataSet);

	OpenNN::Vector<OpenNN::TestingAnalysis::LinearRegressionAnalysis> LinearAnalysisVector = Analysis.perform_linear_regression_analysis();
	Vector<Vector<double>> ErrorAutocorrelation = Analysis.calculate_error_autocorrelation();
	Vector<Vector<double>> ErrorCorsscorrelation = Analysis.calculate_inputs_errors_cross_correlation();
	Vector<Matrix<double>> ErrorData = Analysis.calculate_error_data();
	Vector<Vector<Descriptives>> ErrorDataStats = Analysis.calculate_error_data_statistics();

	// Save Results
	std::string OutputDirectory = DataPath + "/output_" + NameOfSeries + "_";
	DataSet.save(OutputDirectory + "Data.xml");
	NeuralNetwork.save(OutputDirectory + "ANN.xml");
	NeuralNetwork.save_expression(OutputDirectory + "ANN_Expression.txt");
	TrainingStrategy.save(OutputDirectory + "Training_Strategy.xml");
	TrainingResults.save(OutputDirectory + "Training_Results.dat");
	//
	for (size_t LinearAnalysisIndex = 0; LinearAnalysisIndex < LinearAnalysisVector.size(); ++LinearAnalysisIndex)
	{
		OpenNN::TestingAnalysis::LinearRegressionAnalysis& LinearAnalysis = LinearAnalysisVector[LinearAnalysisIndex];
		Vector<double> Out;
		Out.push_back(LinearAnalysis.correlation);
		Out.push_back(LinearAnalysis.intercept);
		Out.push_back(LinearAnalysis.slope);
		std::string DataPath(OutputDirectory + "LinearAnalysis_" + std::to_string(LinearAnalysisIndex) + "Data.dat");
		Out.save(DataPath);
		std::string TargetsPath(OutputDirectory + "LinearAnalysis_" + std::to_string(LinearAnalysisIndex) + "Targets.dat");
		LinearAnalysis.targets.save(TargetsPath);
		std::string OutputPath(OutputDirectory + "LinearAnalysis_" + std::to_string(LinearAnalysisIndex) + "Outputs.dat");
		LinearAnalysis.outputs.save(OutputPath);
	}


	ErrorAutocorrelation.save(OutputDirectory + "ErrorAutocorrelation.dat");
	ErrorCorsscorrelation.save(OutputDirectory + "ErrorCrossCorrelation.dat");
	ErrorData.save(OutputDirectory + "ErrorData.dat");
}

int main(void)
{
    try
    {
        cout << "OpenNN. Time Series Examples." << endl;

		RunSeries("parabola");

		RunSeries("sine");

		RunSeries("increasing_sine");

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}  


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques SL
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
