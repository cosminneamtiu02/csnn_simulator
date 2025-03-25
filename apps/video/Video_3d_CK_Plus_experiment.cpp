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


class CkPlusInput : public Input {
public:
    CkPlusInput(const dataset::CK_Plus::ImageSequence& sequence, int image_width, int image_height) 
        : Input(), _sequence(sequence), _width(image_width), _height(image_height), 
          _label(std::to_string(_sequence.emotion)), _class_name(std::to_string(_sequence.emotion)),
          _current_index(0) {
        
        // Verify that we have valid dimensions and non-empty frames
        if (_sequence.frames.empty()) {
            throw std::runtime_error("Sequence has no frames");
        }
        
        // Make sure all dimensions are valid (non-zero)
        if (_width <= 0 || _height <= 0) {
            throw std::runtime_error("Invalid dimensions: width=" + std::to_string(_width) + 
                                    ", height=" + std::to_string(_height));
        }
        
        // Initialize shape with explicit size_t values to avoid unexpected conversions
        size_t h = static_cast<size_t>(_height);
        size_t w = static_cast<size_t>(_width);
        size_t d = static_cast<size_t>(_sequence.frames.size());
        size_t c = 1; // channels
        
        std::cout << "Creating shape with dimensions: " << h << "×" << w << "×" << d << "×" << c << std::endl;
        _shape = Shape({h, w, d, c});
    }

    virtual ~CkPlusInput() {}

    virtual const Shape& shape() const override {
        return _shape;
    }

    // Implementation of required abstract methods from Input base class
    virtual bool has_next() const override {
        return _current_index < 1; // Only one item per sequence
    }
    
    virtual std::pair<std::string, Tensor<float>> next() override {
        if (!has_next()) {
            throw std::runtime_error("No more data");
        }
        
        // Debug output to check dimensions
        std::cout << "Creating tensor with shape: " << _shape.to_string() << std::endl;
        std::cout << "Frames count: " << _sequence.frames.size() << std::endl;
        
        try {
            Tensor<float> result(_shape);
            
            // Convert sequence frames to tensor
            for (size_t z = 0; z < _sequence.frames.size(); z++) {
                auto& frame = _sequence.frames[z];
                if (!frame) {
                    std::cerr << "Warning: Null frame at position " << z << std::endl;
                    continue; // Skip null frames
                }
                
                for (int y = 0; y < _height; y++) {
                    for (int x = 0; x < _width; x++) {
                        float value = frame->at(y, x, 0, 0);
                        result.at(y, x, z, 0) = value;
                    }
                }
            }
            
            _current_index++;
            return std::make_pair(_label, result);
        }
        catch (const std::exception& e) {
            std::cerr << "Error creating tensor: " << e.what() << std::endl;
            throw;
        }
    }
    
    virtual void reset() override {
        _current_index = 0;
    }
    
    virtual void close() override {
        // Nothing to close
    }

    virtual std::string to_string() const override {
        return "CkPlusInput: " + _label + ", frames: " + std::to_string(_sequence.frames.size());
    }

private:
    dataset::CK_Plus::ImageSequence _sequence;
    int _width;
    int _height;
    std::string _label;
    std::string _class_name;
    size_t _current_index;
    Shape _shape;
};

int main(int argc, char **argv)
{
    // Parse command line arguments
    size_t _filter_size = (argc > 1) ? atoi(argv[1]) : 10;
    int _repeats = (argc > 2) ? atoi(argv[2]) : 5;
    int _epochs = (argc > 3) ? atoi(argv[3]) : 800;
    float _th = (argc > 4) ? atof(argv[4]) : 8.0;
    
    // Dataset parameters
    std::string csv_path = "/home/cosmin/proiecte/datasets/CK+_TIM10/CK+_emotion.csv";
    std::string images_dir = "/home/cosmin/proiecte/datasets/CK+_TIM10";
    int num_folds = 5;
    unsigned int random_seed = 42;
    
    // Video frame dimensions
    size_t _frame_size_width = 48;
    size_t _frame_size_height = 48;

    time_t start_time;
    time(&start_time);

    for (int _repeat = 0; _repeat < _repeats; _repeat++)
    {
        for (int fold = 1; fold <= num_folds; fold++) {
            std::string _dataset = "CK_Plus_" + std::to_string(start_time) + "_3D_" + 
                                  std::to_string(_filter_size) + "_repeat" + std::to_string(_repeat) + 
                                  "_fold" + std::to_string(fold) + "_epochs" + std::to_string(_epochs);

            Experiment<SparseIntermediateExecution> experiment(argc, argv, _dataset);
            
            // Load CK+ dataset
            dataset::CK_Plus ck_plus(csv_path, images_dir, num_folds, random_seed, 
                                     _frame_size_width, _frame_size_height);
            if (!ck_plus.load()) {
                experiment.log() << "Failed to load CK+ dataset" << std::endl;
                return 1;
            }
            
            // Get training and testing sequences for this fold
            auto training_sequences = ck_plus.getTrainingSequences(fold);
            auto testing_sequences = ck_plus.getTestSequences(fold);
            
            experiment.log() << "Training sequences: " << training_sequences.size() << std::endl;
            experiment.log() << "Testing sequences: " << testing_sequences.size() << std::endl;
            
            // Network parameters
            size_t _temporal_sum_pooling = 2;
            size_t _sum_pooling = 8;
            size_t tmp_filter_size = _filter_size;
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
                    experiment.add_train<CkPlusInput>(seq, _frame_size_width, _frame_size_height);
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
                    experiment.add_test<CkPlusInput>(seq, _frame_size_width, _frame_size_height);
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

            experiment.log() << "Running experiment for fold " << fold << " (repeat " << _repeat << ")" << std::endl;
            experiment.run(10000);
        }
    }
    
    return 0;
}
