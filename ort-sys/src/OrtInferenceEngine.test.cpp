#include "OrtInferenceEngine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>

const std::filesystem::path PROJECT_DIR = std::filesystem::path(__FILE__).parent_path().parent_path();

std::vector<std::byte> read_file(const std::filesystem::path &file_path)
{
    auto file_size = std::filesystem::file_size(file_path);
    std::vector<std::byte> file(file_size);
    std::ifstream ifs(file_path, std::ios::binary);
    ifs.read(reinterpret_cast<char *>(file.data()), file.size());

    return std::move(file);
}

TEST_CASE("OrtInferenceEngine with invalid model data")
{
    auto result = OrtInferenceEngine::create(nullptr, 0);
    REQUIRE(result.code == ResultCode::Error);
    REQUIRE(std::string(result.error.get_message()) == "No graph was found in the protobuf.");
}

// TEST_CASE("OrtInferenceEngine")
// {
//     auto model = read_file(PROJECT_DIR / "../ort-cpp/test-models/mat_mul.onnx");
//     auto result = OrtInferenceEngine::create(model.data(), model.size());
//     REQUIRE(result.code == ResultCode::Ok);
// }
