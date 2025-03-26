#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <tuple>
#include "Input.h"
#include "Tensor.h"
#include "Spike.h"

namespace dataset {

class CK_Plus {
public:
    struct ImageSequence {
        ImageSequence() : subject(""), ipostase(0), emotion(0), frames() {}
        std::string subject;
        int ipostase;
        int emotion;
        std::vector<std::shared_ptr<Tensor<float>>> frames;
    };
    
    CK_Plus(const std::string& csv_path, const std::string& images_dir, 
            int num_folds = 10, unsigned int random_seed = 42, 
            int image_width = 48, int image_height = 48);
    ~CK_Plus();

    // Load the dataset from CSV and images
    bool load();

    // Get all sequences for a specific fold and emotion
    std::vector<ImageSequence> getSequences(int fold, int emotion);

    // Get all sequences for training (all folds except test_fold)
    std::vector<ImageSequence> getTrainingSequences(int test_fold);

    // Get all sequences for testing (only test_fold)
    std::vector<ImageSequence> getTestSequences(int test_fold);

    // Get number of emotions
    int getNumEmotions() const { return 6; }

    // Get number of folds
    int getNumFolds() const { return m_num_folds; }
    
    // Get counts of emotions per fold
    std::map<int, std::map<int, int>> getEmotionCounts() const;
    
    // Get emotion name from emotion ID
    std::string getEmotionName(int emotion) const;

    // Convert ImageSequence to Tensor or Spike input
    std::shared_ptr<Tensor<float>> sequenceToTensor(const ImageSequence& seq);
    std::shared_ptr<Input> sequenceToInput(const ImageSequence& seq);
    std::shared_ptr<Spike> sequenceToSpike(const ImageSequence& seq);

    // Print emotion distribution across folds
    void printEmotionDistribution() const;

    // Create an Input object from a sequence
    std::shared_ptr<Input> createInput(const ImageSequence& seq);

private:
    std::string m_csv_path;
    std::string m_images_dir;
    int m_image_width;
    int m_image_height;
    int m_num_folds;
    unsigned int m_random_seed;
    
    // Data structure: fold -> emotion -> sequences
    std::map<int, std::map<int, std::vector<ImageSequence>>> m_data;

    // Load a single image and convert to Tensor
    std::shared_ptr<Tensor<float>> loadImage(const std::string& path);
    
    // Load a sequence of images for a subject/ipostase
    std::vector<std::shared_ptr<Tensor<float>>> loadImageSequence(const std::string& subject, int ipostase);
    
    // Distribute sequences evenly across folds
    void distributeSequences(std::vector<ImageSequence>& sequences);
};

// Define CkPlusInput as a standalone class in the dataset namespace
class CK_Plus_Input : public Input {
public:
    CK_Plus_Input(const CK_Plus::ImageSequence& sequence, int image_width, int image_height);
    virtual ~CK_Plus_Input();

    // Implement required Input interface methods
    virtual const Shape& shape() const override;
    virtual bool has_next() const override;
    virtual std::pair<std::string, Tensor<float>> next() override;
    virtual void reset() override;
    virtual void close() override;
    virtual std::string to_string() const override;

private:
    // Helper methods for constructor
    static Shape createShape(const CK_Plus::ImageSequence& sequence, int width, int height);
    void validateSequence();
    
    CK_Plus::ImageSequence _sequence;
    int _width;
    int _height;
    std::string _label;
    std::string _class_name;
    size_t _current_index;
    Shape _shape;
};

} // namespace dataset
