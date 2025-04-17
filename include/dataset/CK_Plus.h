#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "Input.h"
#include "Tensor.h"
#include "Spike.h"

namespace dataset {

// Forward declaration
class CK_Plus_Input;

class CK_Plus {
public:
    struct ImageSequence {
        std::string subject = "";
        int ipostase = 0;
        int emotion = 0;
        std::vector<std::shared_ptr<Tensor<float>>> frames;
    };
    
    // Emotion mapping constants
    enum Emotion {
        HAPPY = 1,
        FEAR = 2,
        SURPRISE = 3,
        ANGER = 4,
        DISGUST = 5,
        SADNESS = 6
    };
    
    CK_Plus(const std::string& csv_path, const std::string& images_dir, 
            int num_folds = 10, unsigned int random_seed = 42, 
            int image_width = 48, int image_height = 48);
    ~CK_Plus();

    // Core dataset functions
    bool load();
    std::vector<ImageSequence> getSequences(int fold, int emotion);
    std::vector<ImageSequence> getTrainingSequences(int test_fold);
    std::vector<ImageSequence> getTestSequences(int test_fold);
    
    // Helper functions
    int getNumEmotions() const { return 6; }
    int getNumFolds() const { return m_num_folds; }
    unsigned int getRandomSeed() const { return m_random_seed; }
    std::map<int, std::map<int, int>> getEmotionCounts() const;
    std::string getEmotionName(int emotion) const;
    void printEmotionDistribution() const;
    
    // Data conversion functions
    std::shared_ptr<Tensor<float>> sequenceToTensor(const ImageSequence& seq);
    std::shared_ptr<Input> createInput(const ImageSequence& seq);
    std::shared_ptr<Spike> sequenceToSpike(const ImageSequence& seq); // Placeholder

private:
    // Configuration
    std::string m_csv_path;
    std::string m_images_dir;
    int m_image_width;
    int m_image_height;
    int m_num_folds;
    unsigned int m_random_seed;
    
    // Data structure: fold -> emotion -> sequences
    std::map<int, std::map<int, std::vector<ImageSequence>>> m_data;

    // Internal helper methods
    std::shared_ptr<Tensor<float>> loadImage(const std::string& path, bool verbose = false);
    std::vector<std::shared_ptr<Tensor<float>>> loadImageSequence(
        const std::string& subject, int ipostase, bool verbose = false);
    void distributeSequences(std::vector<ImageSequence>& sequences);
};

// Standalone input class for CK+ dataset
class CK_Plus_Input : public Input {
public:
    CK_Plus_Input(const CK_Plus::ImageSequence& sequence, int image_width, int image_height);
    virtual ~CK_Plus_Input();

    // Implementation of Input interface
    virtual const Shape& shape() const override;
    virtual bool has_next() const override;
    virtual std::pair<std::string, Tensor<float>> next() override;
    virtual void reset() override;
    virtual void close() override;
    virtual std::string to_string() const override;

private:
    static Shape createShape(const CK_Plus::ImageSequence& sequence, int width, int height);
    void validateSequence();
    
    CK_Plus::ImageSequence _sequence;
    int _width;
    int _height;
    std::string _label;
    std::string _class_name;
    size_t _current_index;
    Shape _shape;
    bool _has_valid_data;
};

}
