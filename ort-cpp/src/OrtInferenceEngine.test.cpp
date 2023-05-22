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

TEST_CASE("OrtInferenceEngine with fixed-shape model")
{
    auto model = read_file(PROJECT_DIR / "test-models/mat_mul.onnx");
    auto engine = OrtInferenceEngine(model.data(), model.size());

    auto &input_shapes = engine.get_input_shapes();
    REQUIRE(input_shapes == std::vector<std::vector<int64_t>>{{2, 2}, {2, 2}});

    auto &output_shapes = engine.get_output_shapes();
    REQUIRE(output_shapes == std::vector<std::vector<int64_t>>{{2, 2}});

    std::vector<std::vector<float>> inputs{{1, 2, 3, 4}, {5, 6, 7, 8}};
    std::vector<float *> input_ptrs;
    for (auto i = 0; i < inputs.size(); i++)
    {
        input_ptrs.push_back(inputs[i].data());
    }
    engine.set_input_data(input_ptrs.data());

    std::vector<std::vector<float>> outputs{{0, 0, 0, 0}};
    std::vector<float *> output_ptrs;
    for (auto i = 0; i < outputs.size(); i++)
    {
        output_ptrs.push_back(outputs[i].data());
    }
    engine.set_output_data(output_ptrs.data());

    engine.run();

    REQUIRE(outputs == std::vector<std::vector<float>>{{{19, 22, 43, 50}}});
}

TEST_CASE("OrtInferenceEngine with dynamic-shape model")
{
    auto model = read_file(PROJECT_DIR / "test-models/mat_mul_dynamic_shape.onnx");
    auto engine = OrtInferenceEngine(model.data(), model.size());

    REQUIRE(engine.get_input_shapes() == std::vector<std::vector<int64_t>>{{-1, -1}, {-1, -1}});
    REQUIRE(engine.get_output_shapes() == std::vector<std::vector<int64_t>>{{-1, -1}});

    engine.set_input_shapes({{2, 1}, {1, 2}});
    auto &input_shapes = engine.get_input_shapes();
    REQUIRE(input_shapes == std::vector<std::vector<int64_t>>{{2, 1}, {1, 2}});

    engine.set_output_shapes({{2, 2}});
    auto &output_shapes = engine.get_output_shapes();
    REQUIRE(output_shapes == std::vector<std::vector<int64_t>>{{2, 2}});

    std::vector<std::vector<float>> inputs{{1, 2}, {3, 4}};
    std::vector<float *> input_ptrs;
    for (auto i = 0; i < inputs.size(); i++)
    {
        input_ptrs.push_back(inputs[i].data());
    }
    engine.set_input_data(input_ptrs.data());

    std::vector<std::vector<float>> outputs{{0, 0, 0, 0}};
    std::vector<float *> output_ptrs;
    for (auto i = 0; i < outputs.size(); i++)
    {
        output_ptrs.push_back(outputs[i].data());
    }
    engine.set_output_data(output_ptrs.data());

    engine.run();

    REQUIRE(outputs == std::vector<std::vector<float>>{{{3, 4, 6, 8}}});
}
