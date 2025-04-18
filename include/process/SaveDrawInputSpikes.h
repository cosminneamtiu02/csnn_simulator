#ifndef _SAVE_DRAW_INPUT_SPIKES_H
#define _SAVE_DRAW_INPUT_SPIKES_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{

	/**
	 * @brief This method joins video frames together and changes the temporal depth. 
	 * Be careful to increase the sample count when using this in order not to end up with less samples.
	 *
	 * @param frame_number The number of frames to join
	 */
	class SaveDrawInputSpikes : public UniquePassProcess
	{

	public:
		SaveDrawInputSpikes();
		SaveDrawInputSpikes(std::string expName, size_t save = 0, size_t draw = 0);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		std::string _expName;
		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
		size_t _save;
		size_t _draw;
	};

}
#endif
