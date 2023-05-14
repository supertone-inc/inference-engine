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
            auto shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            auto info = TensorInfo(std::move(shape));
            input_info.push_back(std::move(info));
            input_names.push_back(session.GetInputName(i, allocator));
        }

        for (auto i = 0; i < output_count; i++)
        {
            auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            auto info = TensorInfo(std::move(shape));
            output_info.push_back(std::move(info));
            output_names.push_back(session.GetOutputName(i, allocator));
        }
    }

    const std::vector<TensorInfo> &get_input_info() const
    {
        return input_info;
    }

    const std::vector<TensorInfo> &get_output_info() const
    {
        return output_info;
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
            auto &info = input_info[i];
            auto element_count = info.element_count;
            auto shape_data = info.shape.data();
            auto shape_size = info.shape.size();

            if (input_shapes && shape_size > 0)
            {
                shape_data = input_shapes[i];

                element_count = 1;
                for (auto j = 0; j < shape_size; j++)
                {
                    element_count *= shape_data[j];
                }
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
            auto &info = output_info[i];
            auto element_count = info.element_count;
            auto shape_data = info.shape.data();
            auto shape_size = info.shape.size();

            if (output_shapes && shape_size > 0)
            {
                shape_data = output_shapes[i];

                element_count = 1;
                for (auto j = 0; j < shape_size; j++)
                {
                    element_count *= shape_data[j];
                }
            }

            auto tensor =
                Ort::Value::CreateTensor<float>(memory_info, output_data[i], element_count, shape_data, shape_size);

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
    std::vector<TensorInfo> input_info;
    std::vector<const char *> input_names;

    const size_t output_count;
    std::vector<TensorInfo> output_info;
    std::vector<const char *> output_names;
};

OrtInferenceEngine::OrtInferenceEngine(const std::byte *model_data, size_t model_data_size)
    : impl(new Impl(model_data, model_data_size))
{
}

const std::vector<TensorInfo> &OrtInferenceEngine::get_input_info() const
{
    return impl->get_input_info();
}

const std::vector<TensorInfo> &OrtInferenceEngine::get_output_info() const
{
    return impl->get_output_info();
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
