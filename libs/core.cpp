#include "studio/core.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <openssl/evp.h>
#include <random>
#include <sstream>

namespace studio {
const std::map<std::string, std::vector<std::string>> dependencies = {
    {"depth", {}},
    {"poses", {"depth"}},
    {"cloud", {"depth", "poses"}},
    {"tsdf", {"depth", "poses"}},
    {"train", {"depth", "poses", "cloud"}}};
std::string read_text(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot read: " + p.string());
    return {std::istreambuf_iterator<char>(f), {}};
}
json read_json(const fs::path &p) {
    return json::parse(read_text(p));
}
std::string id() {
    std::random_device r;
    std::ostringstream s;
    for (int i = 0; i < 6; ++i)
        s << std::hex << std::setw(2) << std::setfill('0') << (r() & 255);
    return s.str();
}
double now() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}
void write_json(const fs::path &p, const json &j) {
    fs::create_directories(p.parent_path());
    auto t = p.string() + "." + id() + ".tmp";
    {
        std::ofstream f(t);
        f << j.dump(2);
        if (!f)
            throw std::runtime_error("Write failed");
    }
    fs::rename(t, p);
}
std::string sha256(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("Hash input missing");
    auto *c = EVP_MD_CTX_new();
    EVP_DigestInit_ex(c, EVP_sha256(), nullptr);
    std::array<char, 65536> b;
    while (f) {
        f.read(b.data(), b.size());
        EVP_DigestUpdate(c, b.data(), f.gcount());
    }
    unsigned char d[32];
    unsigned n;
    EVP_DigestFinal_ex(c, d, &n);
    EVP_MD_CTX_free(c);
    std::ostringstream s;
    for (unsigned i = 0; i < n; ++i)
        s << std::hex << std::setw(2) << std::setfill('0') << int(d[i]);
    return s.str();
}
bool contained(const fs::path &c, const fs::path &p) {
    auto a = fs::weakly_canonical(c), b = fs::weakly_canonical(p);
    auto ai = a.begin();
    for (auto bi = b.begin(); bi != b.end(); ++bi, ++ai)
        if (ai == a.end() || *ai != *bi)
            return false;
    return true;
}
fs::path scene_path(const fs::path &root, const std::string &s) {
    if (s.size() != 12 || s.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error("Invalid scene ID");
    auto p = root / "data/scenes" / s;
    if (!fs::exists(p / "manifest.json"))
        throw std::runtime_error("Scene not found");
    return p;
}
std::vector<json> runs(const fs::path &scene) {
    std::vector<json> r;
    auto dir = scene / "runs";
    if (fs::exists(dir))
        for (auto &d : fs::directory_iterator(dir))
            if (fs::exists(d.path() / "status.json")) {
                auto status = read_json(d.path() / "status.json");
                if (fs::exists(d.path() / "progress.json")) {
                    try {
                        auto progress = read_json(d.path() / "progress.json");
                        status["progress"] = progress.value("value", 0.0);
                        status["progress_message"] = progress.value("message", std::string());
                        status["progress_phase"] = progress.value("phase", std::string());
                    } catch (...) {
                        // A worker may be atomically replacing this file while the API is read.
                    }
                }
                r.push_back(std::move(status));
            }
    std::sort(r.begin(), r.end(), [](auto &a, auto &b) {
        return a["created_at"].template get<double>() < b["created_at"].template get<double>();
    });
    return r;
}
json active_runs(const fs::path &scene) {
    json current = json::object();
    auto revision = read_json(scene / "manifest.json")["revision"];
    for (auto &r : runs(scene)) {
        if (r["state"] != "succeeded" || r["revision"] != revision)
            continue;
        bool valid = true;
        for (auto &[k, v] : r["dependencies"].items())
            if (!current.contains(k) || current[k]["id"] != v)
                valid = false;
        if (!valid)
            continue;
        current[r["stage"].get<std::string>()] = r;
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = current.begin(); it != current.end();) {
                bool ok = true;
                for (auto &[k, v] : it.value()["dependencies"].items())
                    if (!current.contains(k) || current[k]["id"] != v)
                        ok = false;
                if (!ok) {
                    it = current.erase(it);
                    changed = true;
                } else
                    ++it;
            }
        }
    }
    return current;
}
json validate_config(const std::string &stage, json c) {
    if (!dependencies.count(stage))
        throw std::runtime_error("Unknown stage");
    if (!c.is_object())
        throw std::runtime_error("Config must be an object");
    json defaults;
    if (stage == "depth")
        defaults = {{"backend", "bridgedepth"},
                    {"width", 1280},
                    {"min_depth", 0.04},
                    {"max_depth", 1.5},
                    {"lr_threshold", 1.5}};
    if (stage == "poses")
        defaults = {{"backend", "colmap_rgb"},
                    {"target_width", 224},
                    {"max_depth", 1.5},
                    {"min_quality", 0.25}};
    if (stage == "cloud")
        defaults = {{"voxel_size", 0.002},
                    {"crop_min_depth", 0.04},
                    {"crop_max_depth", 1.5},
                    {"roi_x", 0.0},
                    {"roi_y", 0.0},
                    {"roi_w", 1.0},
                    {"roi_h", 1.0}};
    if (stage == "tsdf")
        defaults = {{"voxel_size", 0.003},
                    {"sdf_trunc", 0.012},
                    {"min_depth", 0.04},
                    {"max_depth", 1.5}};
    if (stage == "train")
        defaults = {{"quality_mode", false},
                    {"adaptive_density", false},
                    {"iterations", 500},
                    {"initial_gaussians", 20000},
                    {"max_gaussians", 30000},
                    {"sh_degree", 0},
                    {"ssim_weight", 0.0},
                    {"depth_weight", 0.2},
                    {"surface_weight", 0.1},
                    {"dense_match_weight", 0.1},
                    {"crop_min_depth", 0.04},
                    {"crop_max_depth", 1.5},
                    {"roi_x", 0.0},
                    {"roi_y", 0.0},
                    {"roi_w", 1.0},
                    {"roi_h", 1.0},
                    {"width", 640}};
    for (auto &[k, v] : c.items())
        if (!defaults.contains(k))
            throw std::runtime_error("Unknown config key: " + k);
    defaults.update(c);
    auto range = [&](const char *k, double lo, double hi) {
        if (defaults.contains(k)) {
            if (!defaults[k].is_number())
                throw std::runtime_error(std::string(k) + " must be numeric");
            double v = defaults[k];
            if (!std::isfinite(v) || v < lo || v > hi)
                throw std::runtime_error(std::string(k) + " out of range");
        }
    };
    range("width", 160, 1280);
    range("min_depth", 0.005, 5);
    range("max_depth", 0.01, 10);
    range("lr_threshold", 0.1, 5);
    range("voxel_size", 0.0005, 0.03);
    range("sdf_trunc", 0.001, 0.1);
    range("iterations", 1, 10000);
    range("initial_gaussians", 100, 200000);
    range("max_gaussians", 100, 200000);
    range("sh_degree", 0, 3);
    range("ssim_weight", 0, 1);
    range("depth_weight", 0, 10);
    range("surface_weight", 0, 10);
    range("dense_match_weight", 0, 10);
    range("target_width", 196, 518);
    range("min_quality", 0, 1);
    range("crop_min_depth", 0.005, 10);
    range("crop_max_depth", 0.01, 10);
    range("roi_x", 0, 1);
    range("roi_y", 0, 1);
    range("roi_w", 0.01, 1);
    range("roi_h", 0.01, 1);
    for (auto k : {"width", "target_width", "iterations", "initial_gaussians", "max_gaussians", "sh_degree"})
        if (defaults.contains(k) && !defaults[k].is_number_integer())
            throw std::runtime_error(std::string(k) + " must be integer");
    if (stage == "train" && !defaults["quality_mode"].is_boolean())
        throw std::runtime_error("quality_mode must be boolean");
    if (stage == "train" && !defaults["adaptive_density"].is_boolean())
        throw std::runtime_error("adaptive_density must be boolean");
    if (stage == "train" && defaults["initial_gaussians"].get<int>() > defaults["max_gaussians"].get<int>())
        throw std::runtime_error("initial_gaussians must not exceed max_gaussians");
    if ((stage == "cloud" || stage == "train") &&
        (defaults["crop_min_depth"].get<double>() >= defaults["crop_max_depth"].get<double>() ||
         defaults["roi_x"].get<double>() + defaults["roi_w"].get<double>() > 1.000001 ||
         defaults["roi_y"].get<double>() + defaults["roi_h"].get<double>() > 1.000001))
        throw std::runtime_error("Invalid training depth range or ROI");
    if (stage == "depth") {
        auto b = defaults["backend"].get<std::string>();
        if (b != "sgbm" && b != "bridgedepth")
            throw std::runtime_error("Backend not integrated");
        if (defaults["min_depth"].get<double>() >= defaults["max_depth"].get<double>())
            throw std::runtime_error("min_depth must be below max_depth");
    }
    if (stage == "poses" && defaults["backend"] != "colmap_rgb")
        throw std::runtime_error("Pose backend not integrated");
    if (stage == "tsdf" &&
        (defaults["sdf_trunc"].get<double>() < defaults["voxel_size"].get<double>() * 2 ||
         defaults["min_depth"].get<double>() >= defaults["max_depth"].get<double>()))
        throw std::runtime_error("TSDF truncation must be >= 2 voxels");
    return defaults;
}
void verify_inputs(const fs::path &scene) {
    auto m = read_json(scene / "manifest.json");
    for (auto &f : m["frames"])
        for (auto &[name, digest] : f["hashes"].items())
            if (sha256(scene / "frames" / f["id"].get<std::string>() / name) != digest)
                throw std::runtime_error("Input changed; import into a new scene: " + name);
}
} // namespace studio
