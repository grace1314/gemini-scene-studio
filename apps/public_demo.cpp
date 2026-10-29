#include "studio/core.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using studio::json;

namespace {

cv::Rect normalized_rect(const json &value, const cv::Size &size) {
    if (!value.is_array() || value.size() != 4)
        throw std::runtime_error("privacy mask rect must be [x, y, width, height]");
    const int x = std::clamp(static_cast<int>(std::floor(value[0].get<double>() * size.width)), 0, size.width - 1);
    const int y = std::clamp(static_cast<int>(std::floor(value[1].get<double>() * size.height)), 0, size.height - 1);
    const int right = std::clamp(static_cast<int>(std::ceil((value[0].get<double>() + value[2].get<double>()) * size.width)), x + 1, size.width);
    const int bottom = std::clamp(static_cast<int>(std::ceil((value[1].get<double>() + value[3].get<double>()) * size.height)), y + 1, size.height);
    return {x, y, right - x, bottom - y};
}

void pixelate(cv::Mat &image, const cv::Rect &rect) {
    cv::Mat region = image(rect);
    cv::Mat tiny;
    const int block = 22;
    cv::resize(region, tiny, cv::Size(std::max(1, rect.width / block), std::max(1, rect.height / block)), 0, 0, cv::INTER_AREA);
    cv::resize(tiny, region, rect.size(), 0, 0, cv::INTER_NEAREST);
    cv::GaussianBlur(region, region, cv::Size(0, 0), 4.0, 4.0, cv::BORDER_REPLICATE);
}

void redact_image(const fs::path &source, const fs::path &destination, const json &masks) {
    cv::Mat image = cv::imread(source.string(), cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("cannot read image: " + source.string());
    for (const auto &mask : masks)
        pixelate(image, normalized_rect(mask.at("rect"), image.size()));
    if (!cv::imwrite(destination.string(), image, {cv::IMWRITE_PNG_COMPRESSION, 5}))
        throw std::runtime_error("cannot write image: " + destination.string());
}

json public_capture(const json &source) {
    json result = {
        {"device", {{"name", source.value("device", json::object()).value("name", "Orbbec Gemini 305")}}},
        {"preset", source.value("preset", "Dual Color Streams")},
        {"profile", source.value("profile", json::object())},
        {"settings_sdk_units", source.value("settings_readback_sdk_units", json::object())},
        {"privacy", {
            {"redacted", true},
            {"method", "deterministic local pixelation"},
            {"removed_metadata", "serial numbers, timestamps, source paths and raw frame identifiers"}
        }},
        {"notes", json::array({
            "Rectified stereo PNGs; privacy masks do not resize or crop images",
            "Public research demo; not a metrology reference"
        })}
    };
    return result;
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: studio_public_demo SOURCE_SCENE PRIVACY_MASKS_JSON OUTPUT_SCENE\n";
            return 2;
        }
        const fs::path source = fs::weakly_canonical(argv[1]);
        const fs::path config_path = fs::weakly_canonical(argv[2]);
        const fs::path output = fs::absolute(argv[3]);
        if (!fs::is_regular_file(source / "manifest.json"))
            throw std::runtime_error("source scene has no manifest.json");
        if (fs::exists(output))
            throw std::runtime_error("output already exists; refusing to overwrite: " + output.string());

        json manifest = studio::read_json(source / "manifest.json");
        const json config = studio::read_json(config_path);
        fs::create_directories(output / "frames");

        for (auto &frame : manifest.at("frames")) {
            const std::string id = frame.at("id");
            const fs::path source_frame = source / "frames" / id;
            const fs::path output_frame = output / "frames" / id;
            fs::create_directories(output_frame);
            const json masks = config.at("frames").value(id, json::array());
            redact_image(source_frame / "left.png", output_frame / "left.png", masks);
            redact_image(source_frame / "right.png", output_frame / "right.png", masks);
            fs::copy_file(source_frame / "K.txt", output_frame / "K.txt");
            const json capture = public_capture(studio::read_json(source_frame / "capture.json"));
            studio::write_json(output_frame / "capture.json", capture);
            frame.erase("source");
            frame["source"] = "public-demo/" + id;
            frame["hashes"] = {
                {"left.png", studio::sha256(output_frame / "left.png")},
                {"right.png", studio::sha256(output_frame / "right.png")},
                {"K.txt", studio::sha256(output_frame / "K.txt")},
                {"capture.json", studio::sha256(output_frame / "capture.json")}
            };
        }

        manifest["id"] = "a11ce5afe001";
        manifest["name"] = "Gemini 305 public desk demo";
        manifest["description"] = "20-frame rectified stereo example with deterministic privacy redaction";
        manifest["revision"] = 1;
        manifest["created_at"] = 0;
        manifest["public_demo"] = {
            {"privacy_redacted", true},
            {"mask_config", "privacy_masks.json"},
            {"source_identity_removed", true}
        };
        studio::write_json(output / "manifest.json", manifest);
        fs::copy_file(config_path, output / "privacy_masks.json");
        std::cout << "wrote public demo scene to " << output << "\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR " << error.what() << "\n";
        return 1;
    }
}
