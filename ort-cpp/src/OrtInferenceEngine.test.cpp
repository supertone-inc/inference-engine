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

TEST_CASE("OrtInferenceEngine Fixed Shape")
{
    auto model = read_file(PROJECT_DIR / "test-models/fixed-shape.onnx");
    auto engine = OrtInferenceEngine(model.data(), model.size());

    auto &input_shapes = engine.get_input_shapes();
    REQUIRE(input_shapes == std::vector<std::vector<int64_t>>{{1, 100}});

    auto &output_shapes = engine.get_output_shapes();
    REQUIRE(output_shapes == std::vector<std::vector<int64_t>>{{1, 100}});

    std::vector<std::vector<float>> inputs;
    std::vector<float *> input_ptrs;
    for (auto i = 0; i < input_shapes.size(); i++)
    {
        inputs.push_back(std::vector<float>(
            get_element_count(input_shapes[i]),
            1.0f
        ));
        input_ptrs.push_back(inputs[i].data());
    }

    std::vector<std::vector<float>> outputs;
    std::vector<float *> output_ptrs;
    for (auto i = 0; i < output_shapes.size(); i++)
    {
        outputs.push_back(std::vector<float>(
            get_element_count(output_shapes[i]),
            0.0f
        ));
        output_ptrs.push_back(outputs[i].data());
    }

    engine.run(
        input_ptrs.data(),
        output_ptrs.data()
    );

    for (auto output : outputs)
    {
        for (auto value : output)
        {
            REQUIRE(value != 0.0f);
        }
    }
}

TEST_CASE("OrtInferenceEngine Dynamic Shape")
{
    auto model = read_file(PROJECT_DIR / "test-models/dynamic-shape.onnx");
    auto engine = OrtInferenceEngine(model.data(), model.size());

    auto input_shapes = engine.get_input_shapes();
    REQUIRE(input_shapes == std::vector<std::vector<int64_t>>{{1, -1}});
    input_shapes[0][1] = 200;

    auto output_shapes = engine.get_output_shapes();
    REQUIRE(output_shapes == std::vector<std::vector<int64_t>>{{1, -1}});
    output_shapes[0][1] = 200;

    std::vector<std::vector<float>> inputs;
    std::vector<float *> input_ptrs;
    std::vector<int64_t *> input_shape_ptrs;
    for (auto i = 0; i < input_shapes.size(); i++)
    {
        inputs.push_back(std::vector<float>(
            get_element_count(input_shapes[i]),
            1.0f
        ));
        input_ptrs.push_back(inputs[i].data());
        input_shape_ptrs.push_back(input_shapes[i].data());
    }

    std::vector<std::vector<float>> outputs;
    std::vector<float *> output_ptrs;
    std::vector<int64_t *> output_shape_ptrs;
    for (auto i = 0; i < output_shapes.size(); i++)
    {
        outputs.push_back(std::vector<float>(
            get_element_count(output_shapes[i]),
            0.0f
        ));
        output_ptrs.push_back(outputs[i].data());
        output_shape_ptrs.push_back(output_shapes[i].data());
    }

    engine.run(
        input_ptrs.data(),
        output_ptrs.data(),
        input_shape_ptrs.data(),
        output_shape_ptrs.data()
    );

    for (auto output : outputs)
    {
        for (auto value : output)
        {
            REQUIRE(value != 0.0f);
        }
    }
}
