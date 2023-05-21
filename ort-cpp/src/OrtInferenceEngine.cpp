#include "inference-engine/OrtInferenceEngine.hpp"

#include <onnxruntime_cxx_api.h>
#include <vector>

namespace inference_engine
{
class OrtInferenceEngine::Impl
{
public:
    Impl(const std::byte *model_data, size_t model_data_size)
        : session(env, model_data, model_data_size, Ort::SessionOptions().SetIntraOpNumThreads(1))
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
        , run_options(Ort::RunOptions{nullptr})
        , input_count(session.GetInputCount())
        , output_count(session.GetOutputCount())
    {
        for (auto i = 0; i < input_count; i++)
        {
            input_names.push_back(session.GetInputName(i, allocator));

            auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            input_shapes.push_back(shape);
            input_element_counts.push_back(get_element_count(shape));
        }

        for (auto i = 0; i < output_count; i++)
        {
            output_names.push_back(session.GetOutputName(i, allocator));

            auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            output_shapes.push_back(shape);
            output_element_counts.push_back(get_element_count(shape));
        }
    }

    const std::vector<std::vector<int64_t>> &get_input_shapes() const
    {
        return input_shapes;
    }

    const std::vector<std::vector<int64_t>> &get_output_shapes() const
    {
        return output_shapes;
    }

    void run(
        const float *const *input_data,
        float **output_data,
        const int64_t *const *input_shapes,
        const int64_t *const *output_shapes
    )
    {
        std::vector<Ort::Value> inputs;
        inputs.reserve(input_count);
        for (auto i = 0; i < input_count; i++)
        {
            auto &shape = this->input_shapes[i];
            auto element_count = input_element_counts[i];
            auto shape_data = const_cast<const int64_t *>(shape.data());
            auto shape_size = shape.size();

            if (input_shapes && shape_size > 0)
            {
                shape_data = input_shapes[i];
                element_count = get_element_count(shape_data, shape_size);
            }

            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                const_cast<float *>(input_data[i]),
                element_count,
                shape_data,
                shape_size
            );

            inputs.push_back(std::move(tensor));
        }

        std::vector<Ort::Value> outputs;
        outputs.reserve(output_count);
        for (auto i = 0; i < output_count; i++)
        {
            auto &shape = this->output_shapes[i];
            auto element_count = output_element_counts[i];
            auto shape_data = const_cast<const int64_t *>(shape.data());
            auto shape_size = shape.size();

            if (output_shapes && shape_size > 0)
            {
                shape_data = output_shapes[i];
                element_count = get_element_count(shape_data, shape_size);
            }

            auto tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                output_data[i],
                element_count,
                shape_data,
                shape_size
            );

            outputs.push_back(std::move(tensor));
        }

        session.Run(
            run_options,
            input_names.data(),
            inputs.data(),
            inputs.size(),
            output_names.data(),
            outputs.data(),
            outputs.size()
        );
    }

private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    const Ort::MemoryInfo memory_info;
    const Ort::RunOptions run_options;

    const size_t input_count;
    std::vector<const char *> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_element_counts;

    const size_t output_count;
    std::vector<const char *> output_names;
    std::vector<std::vector<int64_t>> output_shapes;
    std::vector<int64_t> output_element_counts;
};

OrtInferenceEngine::OrtInferenceEngine(const std::byte *model_data, size_t model_data_size)
    : impl(new Impl(model_data, model_data_size))
{
}

const std::vector<std::vector<int64_t>> &OrtInferenceEngine::get_input_shapes() const
{
    return impl->get_input_shapes();
}

const std::vector<std::vector<int64_t>> &OrtInferenceEngine::get_output_shapes() const
{
    return impl->get_output_shapes();
}

void OrtInferenceEngine::run(
    const float *const *input_data,
    float **output_data,
    const int64_t *const *input_shapes,
    const int64_t *const *output_shapes
)
{
    impl->run(input_data, output_data, input_shapes, output_shapes);
}
} // namespace inference_engine
