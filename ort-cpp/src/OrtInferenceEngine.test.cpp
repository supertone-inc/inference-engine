#include "inference-engine/OrtInferenceEngine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

const std::filesystem::path PROJECT_DIR = std::filesystem::path(__FILE__).parent_path().parent_path();

std::vector<std::byte> read_file(const std::filesystem::path &file_path)
{
    auto file_size = std::filesystem::file_size(file_path);
    std::vector<std::byte> file(file_size);
    std::ifstream ifs(file_path, std::ios::binary);
    ifs.read(reinterpret_cast<char *>(file.data()), file.size());

    return std::move(file);
}

using namespace inference_engine;

namespace Catch
{
template <>
struct StringMaker<TensorInfo>
{
    static std::string convert(TensorInfo const &value)
    {
        return Catch::rangeToString(value.shape);
    }
};
} // namespace Catch

TEST_CASE("OrtInferenceEngine Fixed Shape")
{
    auto model = read_file(PROJECT_DIR / "test-models/fixed-shape.onnx");

    auto engine = OrtInferenceEngine(model.data(), model.size());

    auto input_info = engine.get_input_info();
    REQUIRE(
        input_info ==
        std::vector<TensorInfo>{
            TensorInfo(std::vector<int64_t>{1, 100}),
        }
    );

    auto output_info = engine.get_output_info();
    REQUIRE(
        output_info ==
        std::vector<TensorInfo>{
            TensorInfo(std::vector<int64_t>{1, 100}),
        }
    );

    std::vector<std::vector<float>> input_values;
    std::vector<float *> inputs;
    for (auto i = 0; i < input_info.size(); i++)
    {
        auto count = input_info[i].element_count;
        std::vector<float> values(count, 1.0f);
        input_values.push_back(values);
        inputs.push_back(input_values[i].data());
    }

    std::vector<std::vector<float>> output_values;
    std::vector<float *> outputs;
    for (auto i = 0; i < output_info.size(); i++)
    {
        auto count = output_info[i].element_count;
        std::vector<float> values(count, 0.0f);
        output_values.push_back(values);
        outputs.push_back(output_values[i].data());
    }

    engine.run(inputs.data(), outputs.data());

    for (auto i = 0; i < output_info[0].element_count; i++)
    {
        REQUIRE(outputs[0][i] != 0.0f);
    }
}

TEST_CASE("OrtInferenceEngine Dynamic Shape")
{
    auto model = read_file(PROJECT_DIR / "test-models/dynamic-shape.onnx");

    auto engine = OrtInferenceEngine(model.data(), model.size());

    auto input_info = engine.get_input_info();
    REQUIRE(
        input_info ==
        std::vector<TensorInfo>{
            TensorInfo(std::vector<int64_t>{1, -1}),
        }
    );

    auto output_info = engine.get_output_info();
    REQUIRE(
        output_info ==
        std::vector<TensorInfo>{
            TensorInfo(std::vector<int64_t>{1, -1}),
        }
    );

    auto element_count = 200;

    std::vector<std::vector<int64_t>> input_shape_values{{1, element_count}};
    std::vector<int64_t *> input_shapes;
    for (auto i = 0; i < input_shape_values.size(); i++)
    {
        input_shapes.push_back(input_shape_values[i].data());
    }

    std::vector<std::vector<float>> input_values;
    std::vector<float *> inputs;
    for (auto i = 0; i < input_shape_values.size(); i++)
    {
        std::vector<float> values(element_count, 1.0f);
        input_values.push_back(values);
        inputs.push_back(input_values[i].data());
    }

    std::vector<std::vector<int64_t>> output_shape_values{{1, element_count}};
    std::vector<int64_t *> output_shapes;
    for (auto i = 0; i < output_shape_values.size(); i++)
    {
        output_shapes.push_back(output_shape_values[i].data());
    }

    std::vector<std::vector<float>> output_values;
    std::vector<float *> outputs;
    for (auto i = 0; i < output_info.size(); i++)
    {
        std::vector<float> values(element_count, 0.0f);
        output_values.push_back(values);
        outputs.push_back(output_values[i].data());
    }

    engine.run(inputs.data(), outputs.data(), input_shapes.data(), output_shapes.data());

    for (auto i = 0; i < element_count; i++)
    {
        REQUIRE(outputs[0][i] != 0.0f);
    }
}
