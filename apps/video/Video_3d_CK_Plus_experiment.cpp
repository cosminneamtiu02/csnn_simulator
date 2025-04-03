#include "Experiment.h"
#include "dataset/CK_Plus.h"
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
#include "process/MaxScaling.h"

int main(int argc, char **argv)
{
    // Parse command line arguments 
    size_t _filter_size = (argc > 1) ? atoi(argv[1]) : 10;
    int _epochs = (argc > 2) ? atoi(argv[2]) : 800;
    float _th = (argc > 3) ? atof(argv[3]) : 8.0;
    
    // Get dataset paths from environment variables
    const char* csv_path_ptr = std::getenv("CK_PLUS_CSV_PATH");
    const char* images_dir_ptr = std::getenv("CK_PLUS_IMAGES_DIR");
    
    if (csv_path_ptr == nullptr || images_dir_ptr == nullptr) {
        std::cerr << "Error: Environment variables not set!" << std::endl;
        std::cerr << "Please set CK_PLUS_CSV_PATH and CK_PLUS_IMAGES_DIR environment variables." << std::endl;
        std::cerr << "Example: export CK_PLUS_CSV_PATH=/path/to/CK+_emotion.csv" << std::endl;
        std::cerr << "         export CK_PLUS_IMAGES_DIR=/path/to/CK+_TIM10" << std::endl;
        return 1;
    }
    
    // Convert to std::string
    std::string csv_path(csv_path_ptr);
    std::string images_dir(images_dir_ptr);
    
    int num_folds = 5;
    unsigned int random_seed = 42;
    
    // Video frame dimensions
    size_t _frame_size_width = 48;
    size_t _frame_size_height = 48;

    time_t start_time;
    time(&start_time);

    for (int fold = 1; fold <= num_folds; fold++) {
        std::string _dataset = "CK_Plus_" + std::to_string(start_time) + "_3D_" + 
                               std::to_string(_filter_size) + "_fold" + std::to_string(fold) + 
                               "_epochs" + std::to_string(_epochs);

        Experiment<SparseIntermediateExecution> experiment(argc, argv, _dataset);
        
        // Load CK+ dataset with paths from environment variables
        dataset::CK_Plus ck_plus(csv_path, images_dir, num_folds, random_seed, 
                                 _frame_size_width, _frame_size_height);
        if (!ck_plus.load()) {
            experiment.log() << "Failed to load CK+ dataset" << std::endl;
            return 1;
        }
        else {
            experiment.log() << "CK+ dataset loaded successfully" << std::endl;
        }
        
        // Get training and testing sequences for this fold
        auto training_sequences = ck_plus.getTrainingSequences(fold);
        auto testing_sequences = ck_plus.getTestSequences(fold);
        
        experiment.log() << "Training sequences: " << training_sequences.size() << std::endl;
        experiment.log() << "Testing sequences: " << testing_sequences.size() << std::endl;
        
        // Network parameters
        size_t _temporal_sum_pooling = 2;
        size_t _sum_pooling = 8;
        size_t tmp_filter_size = 1;
        size_t filter_number = 64;
        size_t tmp_stride = 1;

        size_t sampling_size = _epochs;

        // Preprocessing
        experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);
        experiment.push<process::MaxScaling>();
        experiment.push<LatencyCoding>();

        // Add training and testing data with additional error handling
        int training_count = 0;
        for (auto& seq : training_sequences) {
            if (seq.frames.empty()) {
                experiment.log() << "Skipping empty training sequence" << std::endl;
                continue;
            }
            
            try {
                // Use the CK_Plus_Input class instead of the nested class
                experiment.add_train<dataset::CK_Plus_Input>(seq, _frame_size_width, _frame_size_height);
                training_count++;
            } catch (const std::exception& e) {
                experiment.log() << "Error adding training sequence: " << e.what() << std::endl;
            }
        }
        experiment.log() << "Successfully added " << training_count << " training sequences" << std::endl;
        
        int testing_count = 0;
        for (auto& seq : testing_sequences) {
            if (seq.frames.empty()) {
                experiment.log() << "Skipping empty testing sequence" << std::endl;
                continue;
            }
            
            try {
                // Use the CK_Plus_Input class instead of the nested class
                experiment.add_test<dataset::CK_Plus_Input>(seq, _frame_size_width, _frame_size_height);
                testing_count++;
            } catch (const std::exception& e) {
                experiment.log() << "Error adding testing sequence: " << e.what() << std::endl;
            }
        }
        experiment.log() << "Successfully added " << testing_count << " testing sequences" << std::endl;
        
        // Ensure we actually have data before running the experiment
        if (training_count == 0 || testing_count == 0) {
            experiment.log() << "Insufficient data for experiment, skipping fold " << fold << std::endl;
            continue;
        }

        // Network parameters
        float t_obj = 0.65;
        float th_lr = 0.09f;
        float w_lr = 0.009f;

        // Setup 3D convolutional layer
        auto &conv1 = experiment.push<layer::Convolution3D>(
            _filter_size, _filter_size, tmp_filter_size, filter_number, "", 1, 1, tmp_stride);
            
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

        // Setup output
        auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
        conv1_out.add_postprocessing<process::SumPooling>(_sum_pooling, _sum_pooling);
        conv1_out.add_postprocessing<process::TemporalPooling>(_temporal_sum_pooling);
        conv1_out.add_postprocessing<process::FeatureScaling>();
        conv1_out.add_analysis<analysis::Activity>();
        conv1_out.add_analysis<analysis::Coherence>();
        conv1_out.add_analysis<analysis::Svm>();

        experiment.log() << "Running experiment for fold " << fold << std::endl;
        experiment.run(10000);
    }
    
    return 0;
}
