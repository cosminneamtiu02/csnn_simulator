#include "Experiment.h"
#include "dataset/Video.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/SparseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/SimplePreprocessing.h"
#include "process/CompositeChannels.h"
#include "process/OnOffFilter.h"
#include "process/OnOffTempFilter.h"
#include "process/EarlyFusion.h"
#include "process/LateFusion.h"
#include "process/SeparateSign.h"
#include "tool/VideoFrameSelector.h"
#include "tool/AutoFrameNumberSelector.h"
#include "process/MaxScaling.h"

/**
 *  use this loop to find the ideal t_obj, for (float tobj = 0.10f; tobj <= 1.01f; tobj += 0.05f) float rounded_down = floorf(tobj * 100) / 100;
 */

int main(int argc, char **argv)
{
	size_t _filter_size = atoi(argv[1]);
	int _repeats = 10;
	int _epochs = 800;
	float _th = (argc > 2) ? atoi(argv[2]) : 8.;
	
	// size_t _filter_size = 10;
	// int _repeats = 10;
	// int _epochs = 800;
	// float _th = 8.0;

	time_t start_time;
	time(&start_time);

	for (int _repeat = 0; _repeat < _repeats; _repeat++)
	{

		std::string _dataset = "KTH_" + std::to_string(start_time) + "_3D_" + std::to_string(_filter_size) + "_" + std::to_string(_repeat) + "_" + std::to_string(_epochs) + "_" + std::to_string(floor(_th));

		Experiment<SparseIntermediateExecution> experiment(argc, argv, _dataset);

		// The new dimentions of a video frame, set to zero if default dimentions are needed.
		size_t _frame_size_width = 80, _frame_size_height = 60;

		// number of frames to skip, this speeds up the action.
		size_t _video_frames = 10, _train_sample_per_video = 0, _test_sample_per_video = 0, _train_sample_per_video_2 = 0, _test_sample_per_video_2 = 0;
		size_t _temporal_sum_pooling = 2, _sum_pooling = 20; // the dimensions of the output features going into the SVM

		// number of frames to skip, this speeds up the action.
		size_t _th_mv = 0, _frame_gap_train = 3, _frame_gap_test = 3;
		// if the vieo is in greyscale, and if I want tha dataset drawn or not
		size_t _grey = 1, _draw = 0;

		size_t tmp_filter_size = _filter_size;
		size_t filter_number = 64;
		size_t spacial_stride = 1, tmp_stride = 1;

		size_t sampling_size = _epochs; //(_frame_size_height * _frame_size_width * _frame_per_video) / (filter_size * filter_size * tmp_filter_size); // size_t tmp_pooling_size = tmp_filter_size == 2 ? 2 : 1;
		const char *input_path_ptr = std::getenv("INPUT_PATH");
		if (input_path_ptr == nullptr)
		{
			throw std::runtime_error("Require to define INPUT_PATH variable");
		}
		std::string input_path(input_path_ptr);

		experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0); //identificare contoururi

		experiment.push<process::MaxScaling>(); //normalieaza datele intre 0 si 1
		experiment.push<LatencyCoding>(); 

		// The location of the dataset Videos, seperated into train and test folders that contain labeled folders of videos.
		experiment.add_train<dataset::Video>(input_path + "/train", _video_frames, _frame_gap_train, _th_mv, _train_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);
		experiment.add_test<dataset::Video>(input_path + "/test", _video_frames, _frame_gap_test, _th_mv, _test_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);

		float t_obj = 0.65;
		float th_lr = 0.09f;
		float w_lr = 0.009f;

		// filter_width, filter_height, filter_depth, filter_number, model_path, stride_x, stride_y, stride_k, padding_x, padding_y, padding_k
		auto &conv1 = experiment.push<layer::Convolution3D>(_filter_size, _filter_size, tmp_filter_size, filter_number, "", 1, 1, tmp_stride);
		conv1.set_name("conv1");
		conv1.parameter<bool>("draw").set(false);
		conv1.parameter<bool>("save_weights").set(true);
		conv1.parameter<bool>("save_random_start").set(false);
		conv1.parameter<bool>("log_spiking_neuron").set(false);
		conv1.parameter<bool>("inhibition").set(true);
		conv1.parameter<uint32_t>("epoch").set(sampling_size);
		conv1.parameter<float>("annealing").set(0.95f);
		conv1.parameter<float>("min_th").set(1.0f);
		conv1.parameter<float>("t_obj").set(t_obj);
		conv1.parameter<float>("lr_th").set(th_lr);
		conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(_th, 0.1);
		conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
		conv1_out.add_postprocessing<process::SumPooling>(_sum_pooling, _sum_pooling);
		conv1_out.add_postprocessing<process::TemporalPooling>(_temporal_sum_pooling);
		conv1_out.add_postprocessing<process::FeatureScaling>();
		conv1_out.add_analysis<analysis::Activity>();
		conv1_out.add_analysis<analysis::Coherence>();
		conv1_out.add_analysis<analysis::Svm>();

		experiment.run(10000);
	}
}