#include "studio/core.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <csignal>
#include <deque>
#include <fcntl.h>
#include <httplib.h>
#include <iostream>
#include <mutex>
#include <regex>
#include <set>
#include <spawn.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
extern char **environ;
using namespace studio;

struct GeminiCameraProxy {
    std::string token() const {
        httplib::Client c("127.0.0.1", 8767);
        c.set_connection_timeout(1, 0);
        c.set_read_timeout(2, 0);
        auto r = c.Get("/");
        if (!r || r->status != 200)
            throw std::runtime_error("Gemini 305 采集服务未启动（127.0.0.1:8767）");
        std::smatch m;
        if (!std::regex_search(r->body, m, std::regex("let token='([^']+)'")))
            throw std::runtime_error("无法读取 Gemini 305 采集会话");
        return m[1].str();
    }
    json status() const {
        httplib::Client c("127.0.0.1", 8767);
        c.set_connection_timeout(1, 0);
        c.set_read_timeout(2, 0);
        auto r = c.Get("/api/status");
        if (!r || r->status != 200)
            throw std::runtime_error("Gemini 305 状态不可用");
        return json::parse(r->body);
    }
    std::string preview() const {
        httplib::Client c("127.0.0.1", 8767);
        c.set_connection_timeout(1, 0);
        c.set_read_timeout(2, 0);
        auto r = c.Get("/preview.jpg");
        if (!r || r->status != 200)
            throw std::runtime_error("Gemini 305 尚未输出实时画面");
        return r->body;
    }
    void capture() const {
        auto t = token();
        httplib::Client c("127.0.0.1", 8767);
        c.set_connection_timeout(2, 0);
        auto p = c.Get("/api/projects");
        if (!p || p->status != 200)
            throw std::runtime_error("Gemini 305 当前项目不可用");
        auto project = json::parse(p->body).at("active").get<std::string>();
        httplib::Headers h{{"X-Capture-Token", t}, {"X-Project-ID", project}};
        auto r = c.Post("/api/capture", h, "{}", "application/json");
        if (!r || r->status >= 300)
            throw std::runtime_error(r ? r->body : "Gemini 305 采集请求失败");
    }
    void action(const std::string &name, const json &body) const {
        auto t = token();
        httplib::Client c("127.0.0.1", 8767);
        c.set_connection_timeout(2, 0);
        c.set_read_timeout(3, 0);
        auto p = c.Get("/api/projects");
        if (!p || p->status != 200)
            throw std::runtime_error("Gemini 305 当前项目不可用");
        auto project = json::parse(p->body).at("active").get<std::string>();
        httplib::Headers h{{"X-Capture-Token", t}, {"X-Project-ID", project}};
        auto r = c.Post("/api/" + name, h, body.dump(), "application/json");
        if (!r || r->status >= 300)
            throw std::runtime_error(r ? json::parse(r->body).value("error", r->body)
                                       : "Gemini 305 参数请求失败");
    }
};

class Jobs {
    fs::path root;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<fs::path> queue;
    std::thread thread;
    bool stop = false;
    pid_t pid = -1;
    std::string active;
    bool cancelled = false;

  public:
    explicit Jobs(fs::path r) : root(std::move(r)) {
        fs::create_directories(root / "data/scenes");
        for (auto &s : fs::directory_iterator(root / "data/scenes"))
            if (s.is_directory())
                for (auto run : runs(s.path()))
                    if (run["state"] == "running" || run["state"] == "queued") {
                        run["state"] = "interrupted";
                        run["message"] = "Server restarted; run must be started again";
                        write_json(s.path() / "runs" / run["id"].get<std::string>() / "status.json",
                                   run);
                    }
        thread = std::thread([this] { loop(); });
    }
    ~Jobs() {
        {
            std::lock_guard<std::mutex> l(mu);
            stop = true;
            if (pid > 0)
                kill(-pid, SIGTERM);
        }
        cv.notify_all();
        thread.join();
    }
    bool busy(const fs::path &s) {
        for (auto &r : runs(s))
            if (r["state"] == "queued" || r["state"] == "running")
                return true;
        return false;
    }
    json submit(const fs::path &scene, const std::string &stage, json config) {
        std::lock_guard<std::mutex> l(mu);
        if (busy(scene))
            throw std::runtime_error("This scene already has an active job");
        auto manifest = read_json(scene / "manifest.json");
        json deps = json::object();
        if (dependencies.count(stage)) {
            config = validate_config(stage, config);
            if (manifest["frames"].empty())
                throw std::runtime_error("Import data first");
            auto current = active_runs(scene);
            for (auto &d : dependencies.at(stage)) {
                if (!current.contains(d))
                    throw std::runtime_error("Missing current successful stage: " + d);
                deps[d] = current[d]["id"];
            }
        } else if (stage != "import" && stage != "demo" && stage != "archive")
            throw std::runtime_error("Unknown stage");
        if (stage == "demo" && !manifest["frames"].empty())
            throw std::runtime_error("Synthetic data requires an empty scene");
        auto rid = id();
        auto dir = scene / "runs" / rid;
        json request = {
            {"id", rid},          {"scene_id", manifest["id"]},       {"stage", stage},
            {"config", config},   {"revision", manifest["revision"]}, {"dependencies", deps},
            {"created_at", now()}};
        auto status = request;
        status["state"] = "queued";
        status["message"] = "等待本地串行执行";
        write_json(dir / "request.json", request);
        write_json(dir / "status.json", status);
        queue.push_back(dir);
        cv.notify_one();
        return status;
    }
    void cancel(const fs::path &scene, const std::string &rid) {
        std::lock_guard<std::mutex> l(mu);
        auto p = scene / "runs" / rid;
        if (!fs::exists(p / "status.json"))
            throw std::runtime_error("Job not found");
        auto s = read_json(p / "status.json");
        if (s["state"] == "queued") {
            s["state"] = "cancelled";
            write_json(p / "status.json", s);
        } else if (s["state"] == "running" && active == rid) {
            cancelled = true;
            if (pid > 0)
                kill(-pid, SIGTERM);
        }
    }

  private:
    void loop() {
        while (true) {
            fs::path dir;
            {
                std::unique_lock<std::mutex> l(mu);
                cv.wait(l, [&] { return stop || !queue.empty(); });
                if (stop)
                    return;
                dir = queue.front();
                queue.pop_front();
            }
            auto status = read_json(dir / "status.json");
            if (status["state"] == "cancelled")
                continue;
            auto req = read_json(dir / "request.json");
            auto scene = dir.parent_path().parent_path();
            try {
                auto stage = req["stage"].get<std::string>();
                if (dependencies.count(stage)) {
                    verify_inputs(scene);
                    auto current = active_runs(scene);
                    for (auto &[k, v] : req["dependencies"].items())
                        if (!current.contains(k) || current[k]["id"] != v)
                            throw std::runtime_error("Upstream changed; submit a new run");
                }
                std::vector<std::string> args;
                bool native = stage == "import" ||
                              (stage == "poses" && req["config"]["backend"] == "opencv_rgbd") ||
                              stage == "cloud" ||
                              stage == "tsdf" ||
                              (stage == "depth" && req["config"]["backend"] == "sgbm");
                if (native)
                    args = {(root / "build/studio_geometry").string(), dir.string()};
                else
                    args = {(root / ".venv/bin/python").string(), "-u", "-m", "studio.worker",
                            dir.string()};
                std::vector<char *> argv;
                for (auto &a : args)
                    argv.push_back(a.data());
                argv.push_back(nullptr);
                posix_spawn_file_actions_t actions;
                posix_spawn_file_actions_init(&actions);
                posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, (dir / "log.txt").c_str(),
                                                 O_WRONLY | O_CREAT | O_TRUNC, 0644);
                posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
                posix_spawnattr_t attr;
                posix_spawnattr_init(&attr);
                posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
                posix_spawnattr_setpgroup(&attr, 0);
                {
                    std::lock_guard<std::mutex> l(mu);
                    if (read_json(dir / "status.json")["state"] == "cancelled") {
                        posix_spawn_file_actions_destroy(&actions);
                        posix_spawnattr_destroy(&attr);
                        continue;
                    }
                    active = req["id"];
                    cancelled = false;
                    status["state"] = "running";
                    status["started_at"] = now();
                    status["message"] = "正在执行";
                    write_json(dir / "status.json", status);
                    int err = posix_spawn(&pid, argv[0], &actions, &attr, argv.data(), environ);
                    posix_spawn_file_actions_destroy(&actions);
                    posix_spawnattr_destroy(&attr);
                    if (err) {
                        pid = -1;
                        throw std::runtime_error("Cannot launch worker: " + std::to_string(err));
                    }
                }
                int code = 0;
                double cancel_start = 0;
                while (true) {
                    pid_t p;
                    bool c;
                    {
                        std::lock_guard<std::mutex> l(mu);
                        p = pid;
                        c = cancelled || stop;
                    }
                    auto waited = waitpid(p, &code, WNOHANG);
                    if (waited == p)
                        break;
                    if (waited < 0 && errno != EINTR)
                        throw std::runtime_error("waitpid failed");
                    if (c) {
                        if (cancel_start == 0)
                            cancel_start = now();
                        if (now() - cancel_start > 3)
                            kill(-p, SIGKILL);
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                {
                    std::lock_guard<std::mutex> l(mu);
                    pid = -1;
                    status["state"] = (cancelled || stop)
                                          ? "cancelled"
                                          : (WIFEXITED(code) && WEXITSTATUS(code) == 0 &&
                                                     fs::exists(dir / "result.json")
                                                 ? "succeeded"
                                                 : "failed");
                    active.clear();
                }
                status["exit_code"] = WIFEXITED(code) ? WEXITSTATUS(code) : -1;
                status["message"] = status["state"] == "succeeded" ? "完成" : "查看日志了解原因";
                if (status["state"] == "failed") {
                    try {
                        std::istringstream lines(read_text(dir / "log.txt"));
                        std::string line, error;
                        while (std::getline(lines, line))
                            if (line.rfind("ERROR ", 0) == 0)
                                error = line.substr(6);
                        if (!error.empty())
                            status["message"] = error;
                    } catch (...) {
                    }
                }
            } catch (const std::exception &e) {
                status["state"] = "failed";
                status["message"] = e.what();
            }
            status["finished_at"] = now();
            write_json(dir / "status.json", status);
        }
    }
};

int main(int argc, char **argv) {
    try {
        fs::path root = argc > 1 ? fs::canonical(argv[1]) : fs::current_path();
        int port = argc > 2 ? std::stoi(argv[2]) : 8765;
        fs::current_path(root);
        setenv("GEMINI_STUDIO_ROOT", root.c_str(), 1);
        fs::create_directories(root / "data");
        int lock = open((root / "data/server.lock").c_str(), O_CREAT | O_RDWR, 0644);
        if (lock < 0 || flock(lock, LOCK_EX | LOCK_NB))
            throw std::runtime_error("Another server owns this project");
        Jobs jobs(root);
        GeminiCameraProxy camera;
        std::mutex camera_capture_mutex;
        httplib::Server server;
        server.set_payload_max_length(1024 * 1024);
        server.set_pre_routing_handler([&](const httplib::Request &r, httplib::Response &s) {
            const auto host = r.get_header_value("Host");
            if (host != "127.0.0.1:" + std::to_string(port) &&
                host != "localhost:" + std::to_string(port)) {
                s.status = 403;
                return httplib::Server::HandlerResponse::Handled;
            }
            if (r.method != "GET" && r.method != "HEAD") {
                auto origin = r.get_header_value("Origin");
                if (r.get_header_value("X-Studio-Request") != "1" ||
                    (!origin.empty() && origin != "http://" + host)) {
                    s.status = 403;
                    s.set_content("Same-origin request required", "text/plain");
                    return httplib::Server::HandlerResponse::Handled;
                }
            }
            s.set_header("X-Content-Type-Options", "nosniff");
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server.set_exception_handler([](const auto &, auto &s, std::exception_ptr e) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception &ex) {
                s.status = 400;
                s.set_content(json({{"error", ex.what()}}).dump(), "application/json");
            }
        });
        auto reply = [](httplib::Response &s, const json &j) {
            s.set_content(j.dump(), "application/json");
            s.set_header("Cache-Control", "no-store");
        };
        server.Get("/api/health", [&](const auto &, auto &s) {
            reply(s, {{"ok", true}, {"server", "C++17"}, {"version", "0.1.0"}});
        });
        server.Get("/api/system", [&](const auto &, auto &s) {
            json j = {
                {"root", root.string()},
                {"server", "C++17 / cpp-httplib"},
                {"disk_free_gb", fs::space(root).available / 1e9},
                {"camera_status", "通过已运行的本机 Gemini 305 采集服务代理"},
                {"max_frames", 20},
                {"modules",
                 json::array(
                      {{{"name", "OpenCV C++"},
                       {"status", "ready"},
                       {"detail", "SGBM 深度基线、ROI 与点云处理"}},
                      {{"name", "Open3D C++"}, {"status", "ready"}, {"detail", "点云与TSDF"}},
                      {{"name", "BridgeDepth"},
                       {"status",
                        fs::exists(root / "models/bridge_rvc_pretrain.pth") ? "ready" : "missing"},
                       {"detail", "独立PyTorch进程"}},
                      {{"name", "gsplat"},
                       {"status", "installed"},
                       {"detail", "RGB+depth+表面锚定，需CUDA"}},
                      {{"name", "COLMAP"},
                       {"status", "ready"},
                       {"detail", "左目RGB增量SfM、稀疏三角化与固定内参全局BA"}},
                      {{"name", "ALIKED + LightGlue"},
                       {"status",
                        fs::exists(root / "models/lightglue/hub/checkpoints/aliked-n16.pth") &&
                                fs::exists(root / "models/lightglue/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth")
                            ? "ready"
                            : "missing"},
                       {"detail", "纯左目RGB特征与全图像对匹配；不使用RGB-D位姿初值"}},
                      {{"name", "PGSR"}, {"status", "planned"}, {"detail", "后续3DGS表面约束"}}})}};
            if (fs::exists(root / "data/environment.json"))
                j["environment"] = read_json(root / "data/environment.json");
            if (fs::exists(root / "models/manifest.json"))
                j["weights"] = read_json(root / "models/manifest.json");
            reply(s, j);
        });
        server.Get("/api/camera/status", [&](const auto &, auto &s) {
            try {
                reply(s, camera.status());
            } catch (const std::exception &e) {
                s.status = 503;
                reply(s, {{"ready", false}, {"error", e.what()}});
            }
        });
        server.Get("/api/camera/preview.jpg", [&](const auto &, auto &s) {
            try {
                s.set_content(camera.preview(), "image/jpeg");
                s.set_header("Cache-Control", "no-store");
            } catch (const std::exception &e) {
                s.status = 503;
                s.set_content(e.what(), "text/plain");
            }
        });
        server.Post("/api/camera/settings", [&](const auto &r, auto &s) {
            auto body = json::parse(r.body);
            for (auto key : {"exposure", "gain", "white_balance"})
                if (!body.contains(key) || !body[key].is_number_integer())
                    throw std::runtime_error(std::string("相机参数缺失或类型错误: ") + key);
            camera.action("settings", body);
            reply(s, {{"ok", true}, {"message", "参数已发送，等待设备回读"}});
        });
        server.Post("/api/camera/auto", [&](const auto &, auto &s) {
            camera.action("auto", json::object());
            reply(s, {{"ok", true}, {"message", "自动曝光与白平衡调整已启动"}});
        });
        server.Get("/api/scenes", [&](const auto &, auto &s) {
            json list = json::array();
            for (auto &d : fs::directory_iterator(root / "data/scenes"))
                if (fs::exists(d.path() / "manifest.json"))
                    list.push_back(read_json(d.path() / "manifest.json"));
            reply(s, list);
        });
        server.Post("/api/scenes", [&](const auto &r, auto &s) {
            auto b = json::parse(r.body);
            auto name = b.value("name", std::string("未命名场景"));
            auto description = b.value("description", std::string());
            if (name.empty() || name.size() > 200)
                throw std::runtime_error("Name must contain 1–200 bytes");
            if (description.size() > 500)
                throw std::runtime_error("Description must not exceed 500 bytes");
            auto sid = id();
            json m = {{"id", sid},
                      {"name", name},
                      {"description", description},
                      {"created_at", now()},
                      {"revision", 0},
                      {"synthetic", false},
                      {"frames", json::array()},
                      {"processing", {{"crop_min_depth", 0.04},
                                      {"crop_max_depth", 1.5},
                                      {"roi_x", 0.0}, {"roi_y", 0.0},
                                      {"roi_w", 1.0}, {"roi_h", 1.0},
                                      {"frame_rois", json::object()},
                                      {"voxel_size", 0.002}}},
                      {"coordinate_system", "OpenCV; T_world_camera; metres; Z-depth"}};
            write_json(root / "data/scenes" / sid / "manifest.json", m);
            reply(s, m);
        });
        server.Post(R"(/api/scenes/([0-9a-f]{12})/settings)", [&](const auto &r, auto &s) {
            auto p = scene_path(root, r.matches[1]);
            if (jobs.busy(p))
                throw std::runtime_error("请等待当前场景任务完成后再修改重建范围");
            auto b = json::parse(r.body);
            auto finite = [&](const char *key, double lo, double hi) {
                if (!b.contains(key) || !b[key].is_number())
                    throw std::runtime_error(std::string("参数缺失: ") + key);
                const double value = b[key];
                if (!std::isfinite(value) || value < lo || value > hi)
                    throw std::runtime_error(std::string("参数超出范围: ") + key);
                return value;
            };
            const double min_depth = finite("crop_min_depth", 0.005, 10);
            const double max_depth = finite("crop_max_depth", 0.01, 10);
            const double x = finite("roi_x", 0, 1), y = finite("roi_y", 0, 1);
            const double w = finite("roi_w", 0.01, 1), h = finite("roi_h", 0.01, 1);
            const double voxel = finite("voxel_size", 0.0005, 0.03);
            if (min_depth >= max_depth || x + w > 1.000001 || y + h > 1.000001)
                throw std::runtime_error("深度范围或 ROI 无效");
            auto manifest = read_json(p / "manifest.json");
            json frame_rois = b.value("frame_rois", manifest.value("processing", json::object())
                                                        .value("frame_rois", json::object()));
            if (!frame_rois.is_object())
                throw std::runtime_error("逐帧 ROI 必须是对象");
            std::set<std::string> frame_ids;
            for (const auto &frame : manifest["frames"])
                frame_ids.insert(frame.at("id").template get<std::string>());
            for (auto &[frame_id, polygon] : frame_rois.items()) {
                if (!frame_ids.count(frame_id) || !polygon.is_array() ||
                    (!polygon.empty() && (polygon.size() < 3 || polygon.size() > 128)))
                    throw std::runtime_error("逐帧 ROI 无效: " + frame_id);
                double twice_area = 0;
                for (size_t point_index = 0; point_index < polygon.size(); ++point_index) {
                    const auto &point = polygon[point_index];
                    if (!point.is_array() || point.size() != 2 || !point[0].is_number() ||
                        !point[1].is_number())
                        throw std::runtime_error("ROI 顶点格式无效: " + frame_id);
                    const double px = point[0], py = point[1];
                    if (!std::isfinite(px) || !std::isfinite(py) || px < 0 || px > 1 || py < 0 ||
                        py > 1)
                        throw std::runtime_error("ROI 顶点超出图像范围: " + frame_id);
                    if (!polygon.empty()) {
                        const auto &next = polygon[(point_index + 1) % polygon.size()];
                        if (next.is_array() && next.size() == 2 && next[0].is_number() &&
                            next[1].is_number())
                            twice_area += px * next[1].get<double>() - next[0].get<double>() * py;
                    }
                }
                if (!polygon.empty() && std::abs(twice_area) < 0.0002)
                    throw std::runtime_error("ROI 多边形面积过小: " + frame_id);
            }
            manifest["processing"] = {{"crop_min_depth", min_depth},
                                      {"crop_max_depth", max_depth},
                                      {"roi_x", x}, {"roi_y", y},
                                      {"roi_w", w}, {"roi_h", h},
                                      {"frame_rois", frame_rois},
                                      {"voxel_size", voxel}};
            write_json(p / "manifest.json", manifest);
            reply(s, manifest["processing"]);
        });
        server.Get(R"(/api/scenes/([0-9a-f]{12}))", [&](const auto &r, auto &s) {
            auto p = scene_path(root, r.matches[1]);
            auto m = read_json(p / "manifest.json");
            m["runs"] = runs(p);
            m["active"] = active_runs(p);
            json artifacts = json::array();
            for (auto &d : fs::recursive_directory_iterator(p)) {
                if (d.is_regular_file() && !d.is_symlink() && d.path().extension() != ".tmp") {
                    auto rel = fs::relative(d.path(), p).generic_string();
                    artifacts.push_back({{"path", rel}, {"bytes", d.file_size()}});
                }
            }
            m["artifacts"] = artifacts;
            reply(s, m);
        });
        server.Delete(R"(/api/scenes/([0-9a-f]{12})/frames/([0-9]{4,8}))",
                      [&](const auto &r, auto &s) {
            std::lock_guard<std::mutex> mutation_guard(camera_capture_mutex);
            auto scene = scene_path(root, r.matches[1]);
            if (jobs.busy(scene))
                throw std::runtime_error("请等待当前场景任务完成后再删除帧");
            auto manifest = read_json(scene / "manifest.json");
            const auto frame_id = r.matches[2].str();
            auto found = std::find_if(manifest["frames"].begin(), manifest["frames"].end(),
                                      [&](const auto &frame) {
                                          return frame.value("id", std::string()) == frame_id;
                                      });
            if (found == manifest["frames"].end())
                throw std::runtime_error("帧不存在或已经移入回收站");
            auto frame = *found;
            auto source = scene / "frames" / frame_id;
            if (!fs::is_directory(source))
                throw std::runtime_error("帧目录不存在，未修改场景清单");
            auto destination = scene / "trash" / "frames" / (frame_id + "_" + id());
            fs::create_directories(destination.parent_path());
            auto marker = source / ".deleted.json";
            write_json(marker,
                       {{"deleted_at", now()}, {"scene_id", manifest["id"]}, {"frame", frame}});
            try {
                fs::rename(source, destination);
            } catch (...) {
                fs::remove(marker);
                throw;
            }
            try {
                manifest["frames"].erase(found);
                manifest["revision"] = manifest["revision"].template get<int>() + 1;
                write_json(scene / "manifest.json", manifest);
            } catch (...) {
                fs::rename(destination, source);
                fs::remove(marker);
                throw;
            }
            reply(s, {{"ok", true},
                      {"frame_id", frame_id},
                      {"remaining_frames", manifest["frames"].size()},
                      {"revision", manifest["revision"]},
                      {"trash", fs::relative(destination, scene).generic_string()}});
        });
        server.Post(R"(/api/scenes/([0-9a-f]{12})/jobs)", [&](const auto &r, auto &s) {
            auto b = json::parse(r.body);
            reply(s, jobs.submit(scene_path(root, r.matches[1]), b.at("stage"),
                                 b.value("config", json::object())));
        });
        server.Post(R"(/api/scenes/([0-9a-f]{12})/capture)", [&](const auto &r, auto &s) {
            std::lock_guard<std::mutex> capture_guard(camera_capture_mutex);
            auto scene = scene_path(root, r.matches[1]);
            auto manifest = read_json(scene / "manifest.json");
            if (manifest.value("synthetic", false))
                throw std::runtime_error("合成场景不接受相机帧，请创建真实采集场景");
            if (manifest["frames"].size() >= 20)
                throw std::runtime_error("场景已达到 20 组双目帧上限");
            if (jobs.busy(scene))
                throw std::runtime_error("请等待当前场景任务完成后再采集");
            auto before_state = camera.status();
            auto before = before_state.contains("last_capture") &&
                                  before_state["last_capture"].is_string()
                              ? before_state["last_capture"].get<std::string>()
                              : std::string();
            camera.capture();
            json state;
            for (int i = 0; i < 120; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                state = camera.status();
                auto captured = state.contains("last_capture") &&
                                        state["last_capture"].is_string()
                                    ? state["last_capture"].get<std::string>()
                                    : std::string();
                if (!captured.empty() && captured != before && !state.value("busy", true)) {
                    auto job = jobs.submit(scene, "import",
                                           {{"source", captured}, {"rectified_confirmed", true}});
                    reply(s, {{"camera", state}, {"import_job", job}});
                    return;
                }
                auto error = state.contains("error") && state["error"].is_string()
                                 ? state["error"].get<std::string>()
                                 : std::string();
                if (!error.empty())
                    throw std::runtime_error(error);
            }
            throw std::runtime_error("等待新双目帧超时；未写入场景");
        });
        server.Post(R"(/api/scenes/([0-9a-f]{12})/jobs/([0-9a-f]{12})/cancel)",
                    [&](const auto &r, auto &s) {
                        jobs.cancel(scene_path(root, r.matches[1]), r.matches[2]);
                        reply(s, {{"ok", true}});
                    });
        server.Get(R"(/api/scenes/([0-9a-f]{12})/file)", [&](const auto &r, auto &s) {
            auto scene = scene_path(root, r.matches[1]);
            auto p = fs::weakly_canonical(scene / r.get_param_value("path"));
            if (!contained(p, scene) || !fs::is_regular_file(p))
                throw std::runtime_error("File missing or outside scene");
            auto ext = p.extension().string();
            std::string mime = ext == ".png"                    ? "image/png"
                               : ext == ".json"                 ? "application/json"
                               : ext == ".txt" || ext == ".log" ? "text/plain"
                                                                : "application/octet-stream";
            s.set_file_content(p.string(), mime);
            if (r.has_param("download"))
                s.set_header("Content-Disposition",
                             "attachment; filename=\"" + p.filename().string() + "\"");
        });
        server.set_mount_point("/vendor", (root / "3rdparty/three").string());
        server.set_mount_point("/splat-viewer", (root / "3rdparty/splat-viewer").string());
        if (!server.set_mount_point("/", (root / "web").string()))
            throw std::runtime_error("Missing web directory");
        std::cout << "Gemini Scene Studio http://127.0.0.1:" << port << "\n" << std::flush;
        if (!server.listen("127.0.0.1", port))
            throw std::runtime_error("Cannot listen on local port");
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
