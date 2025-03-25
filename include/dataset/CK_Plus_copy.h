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

    CK_Plus(const std::string& csv_path, const std::string& images_dir, int image_width = 48, int image_height = 48);
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
    int getNumFolds() const { return 10; }

    // Convert ImageSequence to Tensor or Spike input
    std::shared_ptr<Tensor<float>> sequenceToTensor(const ImageSequence& seq);
    std::shared_ptr<Input> sequenceToInput(const ImageSequence& seq);
    std::shared_ptr<Spike> sequenceToSpike(const ImageSequence& seq);

    // Helper functions for tensor-emotion verification
    std::string getEmotionName(int emotion) const;
    void printTensorInfo(const std::shared_ptr<Tensor<float>>& tensor);

private:
    std::string m_csv_path;
    std::string m_images_dir;
    int m_image_width;
    int m_image_height;
    
    // Data structure: fold -> emotion -> sequences
    std::map<int, std::map<int, std::vector<ImageSequence>>> m_data;

    // Load a single image and convert to Tensor
    std::shared_ptr<Tensor<float>> loadImage(const std::string& path);
    
    // Load a sequence of images for a subject/ipostase
    std::vector<std::shared_ptr<Tensor<float>>> loadImageSequence(const std::string& subject, int ipostase);
};

} // namespace dataset
