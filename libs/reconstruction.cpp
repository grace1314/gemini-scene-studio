#include "studio/reconstruction.hpp"
#include <algorithm>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iostream>
#include <open3d/Open3D.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <set>
using namespace studio;
using Mat4 = Eigen::Matrix4d;
using Vec3 = Eigen::Vector3d;
namespace {
fs::path progress_path;
std::string progress_phase;
}

json matrix_json(const Mat4 &T) {
    json j = json::array();
    for (int r = 0; r < 4; ++r) {
        json row = json::array();
        for (int c = 0; c < 4; ++c)
            row.push_back(T(r, c));
        j.push_back(row);
    }
    return j;
}
Mat4 matrix(const json &j) {
    Mat4 T;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            T(r, c) = j.at(r).at(c);
    return T;
}
cv::Mat intrinsic(const json &j) {
    cv::Mat K(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            K.at<double>(r, c) = j.at(r).at(c);
    return K;
}
json intrinsic_json(const cv::Mat &K) {
    json j = json::array();
    for (int r = 0; r < 3; ++r) {
        json row = json::array();
        for (int c = 0; c < 3; ++c)
            row.push_back(K.at<double>(r, c));
        j.push_back(row);
    }
    return j;
}
void progress(double v, const std::string &m) {
    std::cout << "PROGRESS " << v << " " << m << std::endl;
    if (!progress_path.empty())
        write_json(progress_path,
                   {{"value", std::clamp(v, 0.0, 1.0)},
                    {"message", m},
                    {"phase", progress_phase},
                    {"updated_at", now()}});
}
fs::path upstream(const fs::path &s, const json &r, const std::string &key) {
    return s / "runs" / r["dependencies"][key].get<std::string>();
}
struct Frame {
    std::string id;
    cv::Mat depth, quality, rgb, K;
};
Frame load_frame(const fs::path &dir, const std::string &id) {
    auto m = read_json(dir / "result.json")["frames"][id];
    Frame f{id, cv::imread((dir / id / "depth.tiff").string(), cv::IMREAD_UNCHANGED),
            cv::imread((dir / id / "quality.tiff").string(), cv::IMREAD_UNCHANGED),
            cv::imread((dir / id / "rgb.png").string()), intrinsic(m["K"])};
    if (f.depth.empty() || f.quality.empty() || f.rgb.empty() || f.depth.type() != CV_32FC1)
        throw std::runtime_error("Missing float32 depth/quality TIFF: " + id);
    return f;
}
Vec3 unproject(const Frame &f, int x, int y) {
    double z = f.depth.at<float>(y, x);
    return {(x - f.K.at<double>(0, 2)) * z / f.K.at<double>(0, 0),
            (y - f.K.at<double>(1, 2)) * z / f.K.at<double>(1, 1), z};
}

cv::Mat frame_roi_mask(const fs::path &scene, const std::string &ident, int width, int height) {
    cv::Mat mask(height, width, CV_8U, cv::Scalar(255));
    auto manifest = read_json(scene / "manifest.json");
    auto processing = manifest.value("processing", json::object());
    auto rois = processing.value("frame_rois", json::object());
    if (!rois.contains(ident) || !rois[ident].is_array() || rois[ident].size() < 3)
        return mask;
    std::vector<cv::Point> polygon;
    for (const auto &point : rois[ident])
        polygon.emplace_back(std::clamp(int(std::lround(point[0].get<double>() * (width - 1))), 0,
                                        width - 1),
                             std::clamp(int(std::lround(point[1].get<double>() * (height - 1))), 0,
                                        height - 1));
    mask.setTo(0);
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{polygon}, cv::Scalar(255));
    return mask;
}

json import_data(const fs::path &scene, const json &req) {
    auto cfg = req["config"];
    if (!cfg.value("rectified_confirmed", false))
        throw std::runtime_error(
            "Confirm rectified images, equal principal points, baseline in metres");
    auto source = fs::canonical(cfg.at("source").get<std::string>());
    auto root = fs::canonical(std::getenv("GEMINI_STUDIO_ROOT"));
    auto allowed = fs::path(std::getenv("HOME")) / "Tars_Project";
    if (!contained(source, allowed) || contained(source, root))
        throw std::runtime_error(
            "Import source must be inside ~/Tars_Project and outside this project");
    std::vector<fs::path> candidates;
    if (fs::exists(source / "left.png"))
        candidates.push_back(source);
    else
        for (auto &d : fs::directory_iterator(source))
            if (d.is_directory() && fs::exists(d.path() / "left.png"))
                candidates.push_back(d.path());
    std::sort(candidates.begin(), candidates.end());
    auto m = read_json(scene / "manifest.json");
    const size_t old_count = m["frames"].size();
    if (candidates.empty() || candidates.size() > 20 - std::min<size_t>(20, old_count))
        throw std::runtime_error("Scene accepts at most 20 stereo pairs total");
    json frames = m["frames"];
    size_t next_index = 0;
    for (const auto &frame : frames) {
        const auto existing = frame.at("id").get<std::string>();
        if (existing.empty() || existing.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("Invalid existing frame ID");
        next_index = std::max(next_index, size_t(std::stoul(existing)) + 1);
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
        auto p = candidates[i];
        auto l = cv::imread((p / "left.png").string()), r = cv::imread((p / "right.png").string());
        if (l.empty() || r.empty() || l.size() != r.size() || l.cols > 8192 || l.rows > 8192 ||
            l.cols < 32 || l.rows < 32)
            throw std::runtime_error("Invalid stereo PNG pair");
        cv::Mat K(3, 3, CV_64F);
        std::ifstream kfile(p / "K.txt");
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                kfile >> K.at<double>(a, b);
        double baseline = 0;
        kfile >> baseline;
        if (!kfile || !cv::checkRange(K) || baseline < 0.001 || baseline > 0.5 ||
            !std::isfinite(baseline) || K.at<double>(0, 0) <= 0 || K.at<double>(1, 1) <= 0 ||
            K.at<double>(0, 2) < 0 || K.at<double>(0, 2) >= l.cols || K.at<double>(1, 2) < 0 ||
            K.at<double>(1, 2) >= l.rows || std::abs(K.at<double>(2, 2) - 1) > 1e-8 ||
            std::abs(K.at<double>(0, 1)) + std::abs(K.at<double>(1, 0)) +
                    std::abs(K.at<double>(2, 0)) + std::abs(K.at<double>(2, 1)) >
                1e-8)
            throw std::runtime_error("Invalid calibration / metre baseline");
        char fid[8];
        std::snprintf(fid, sizeof(fid), "%04d", int(next_index + i));
        frames.push_back({{"id", fid},
                          {"K", intrinsic_json(K)},
                          {"baseline_m", baseline},
                          {"size", {l.cols, l.rows}},
                          {"source", p.string()},
                          {"hashes", json::object()}});
    }
    for (size_t i = old_count; i < frames.size(); ++i) {
        auto &f = frames[i];
        auto p = fs::path(f["source"].get<std::string>()),
             dest = scene / "frames" / f["id"].get<std::string>();
        fs::create_directories(dest);
        for (auto name : {"left.png", "right.png", "K.txt", "capture.json"})
            if (fs::exists(p / name)) {
                fs::copy_file(p / name, dest / name, fs::copy_options::overwrite_existing);
                f["hashes"][name] = sha256(dest / name);
            }
    }
    m["frames"] = frames;
    m["revision"] = m["revision"].get<int>() + 1;
    m["rectified_confirmed"] = true;
    write_json(scene / "manifest.json", m);
    return {{"imported_frames", frames.size()}, {"backend", "C++ OpenCV"}};
}

json stereo_depth(const fs::path &scene, const fs::path &out, const json &req) {
    auto frames = read_json(scene / "manifest.json")["frames"];
    auto cfg = req["config"];
    json result = {{"backend", "OpenCV C++ SGBM"}, {"frames", json::object()}};
    for (size_t i = 0; i < frames.size(); ++i) {
        auto f = frames[i];
        auto ident = f["id"].get<std::string>();
        auto dir = out / ident;
        fs::create_directories(dir);
        auto src = scene / "frames" / ident;
        cv::Mat left = cv::imread((src / "left.png").string()),
                right = cv::imread((src / "right.png").string());
        int w = std::min(cfg["width"].get<int>(), left.cols),
            h = std::lround(left.rows * double(w) / left.cols);
        auto K = intrinsic(f["K"]);
        K.row(0) *= double(w) / left.cols;
        K.row(1) *= double(h) / left.rows;
        cv::resize(left, left, {w, h}, 0, 0, cv::INTER_AREA);
        cv::resize(right, right, {w, h}, 0, 0, cv::INTER_AREA);
        auto infer = [&](cv::Mat a, cv::Mat b) {
            cv::Mat ag, bg, disp;
            cv::cvtColor(a, ag, cv::COLOR_BGR2GRAY);
            cv::cvtColor(b, bg, cv::COLOR_BGR2GRAY);
            int n = std::min(128, std::max(16, w / 3 / 16 * 16));
            auto s = cv::StereoSGBM::create(0, n, 5, 8 * 25, 32 * 25, 1, 0, 8, 80, 2,
                                            cv::StereoSGBM::MODE_SGBM_3WAY);
            s->compute(ag, bg, disp);
            disp.convertTo(disp, CV_32F, 1.0 / 16);
            return disp;
        };
        auto disp = infer(left, right);
        cv::Mat rf, lf;
        cv::flip(right, rf, 1);
        cv::flip(left, lf, 1);
        auto dr = infer(rf, lf);
        cv::flip(dr, dr, 1);
        cv::Mat depth = cv::Mat::zeros(h, w, CV_32F), quality = depth.clone(),
                valid = cv::Mat::zeros(h, w, CV_8U), gray = valid.clone();
        int count = 0;
        double min = cfg["min_depth"], max = cfg["max_depth"], tol = cfg["lr_threshold"],
               B = f["baseline_m"];
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float d = disp.at<float>(y, x);
                int xr = std::lround(x - d);
                if (!std::isfinite(d) || d <= 0 || xr < 0 || xr >= w)
                    continue;
                float other = dr.at<float>(y, xr), err = std::abs(d - other);
                double z = K.at<double>(0, 0) * B / d;
                if (other <= 0 || err > tol || z < min || z > max)
                    continue;
                depth.at<float>(y, x) = z;
                quality.at<float>(y, x) = std::exp(-err);
                valid.at<uchar>(y, x) = 255;
                gray.at<uchar>(y, x) = std::clamp(int((z - min) / (max - min) * 255), 0, 255);
                ++count;
            }
        cv::Mat colored;
        cv::applyColorMap(gray, colored, cv::COLORMAP_TURBO);
        colored.setTo(cv::Scalar(), valid == 0);
        cv::imwrite((dir / "depth.tiff").string(), depth);
        cv::imwrite((dir / "quality.tiff").string(), quality);
        cv::imwrite((dir / "disparity.tiff").string(), disp);
        cv::imwrite((dir / "rgb.png").string(), left);
        cv::imwrite((dir / "valid.png").string(), valid);
        cv::imwrite((dir / "preview.png").string(), colored);
        result["frames"][ident] = {{"K", intrinsic_json(K)},
                                   {"width", w},
                                   {"height", h},
                                   {"valid_fraction", double(count) / (w * h)}};
        progress(double(i + 1) / frames.size(), "Stereo depth " + ident);
    }
    return result;
}

json estimate_poses(const fs::path &scene, const fs::path &out, const json &req) {
    auto depth = upstream(scene, req, "depth");
    auto manifest = read_json(scene / "manifest.json");
    std::vector<Frame> frames;
    std::vector<std::vector<cv::KeyPoint>> keys;
    std::vector<cv::Mat> descs;
    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> icp_clouds;
    std::vector<std::shared_ptr<open3d::geometry::PointCloud>> geometry_clouds;
    auto sift = cv::SIFT::create(5000);
    for (auto &f : manifest["frames"]) {
        frames.push_back(load_frame(depth, f["id"]));
        cv::Mat gray, d;
        std::vector<cv::KeyPoint> k;
        cv::cvtColor(frames.back().rgb, gray, cv::COLOR_BGR2GRAY);
        sift->detectAndCompute(gray, cv::noArray(), k, d);
        keys.push_back(k);
        descs.push_back(d);
        auto dense = std::make_shared<open3d::geometry::PointCloud>();
        for (int y = 0; y < frames.back().depth.rows; y += 2)
            for (int x = 0; x < frames.back().depth.cols; x += 2) {
                float z = frames.back().depth.at<float>(y, x);
                float q = frames.back().quality.at<float>(y, x);
                if (!std::isfinite(z) || z < 0.04f || z > 1.5f || !std::isfinite(q) || q < 0.3f)
                    continue;
                dense->points_.push_back(unproject(frames.back(), x, y));
            }
        auto down = dense->VoxelDownSample(0.006);
        down->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.02, 30));
        icp_clouds.push_back(down);
        auto fine_voxels = dense->VoxelDownSample(0.0035);
        auto [clean, clean_indices] = fine_voxels->RemoveStatisticalOutliers(16, 1.8);
        clean->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(0.012, 24));
        clean->EstimateCovariances(open3d::geometry::KDTreeSearchParamHybrid(0.012, 24));
        auto geometry = std::make_shared<open3d::geometry::PointCloud>();
        geometry->points_.reserve(clean->points_.size());
        geometry->normals_.reserve(clean->points_.size());
        geometry->covariances_.reserve(clean->points_.size());
        for (size_t p = 0; p < clean->points_.size(); ++p) {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(clean->covariances_[p]);
            auto values = eig.eigenvalues().cwiseMax(0.0);
            double curvature = values.sum() > 1e-12 ? values[0] / values.sum() : 0.0;
            // Preserve all locally curved/edge-like geometry, but only one fifth of
            // near-planar samples so the tabletop cannot dominate tangential motion.
            if (curvature < 0.0025 && p % 5 != 0)
                continue;
            geometry->points_.push_back(clean->points_[p]);
            geometry->normals_.push_back(clean->normals_[p]);
            geometry->covariances_.push_back(clean->covariances_[p]);
        }
        geometry_clouds.push_back(geometry);
        progress(double(frames.size()) / manifest["frames"].size() * 0.1,
                 "SIFT 特征 " + frames.back().id);
    }
    int n = frames.size();
    if (n < 2)
        throw std::runtime_error("At least two stereo views required");
    struct Edge {
        int i, j, count, coverage, mutual_matches;
        Mat4 T;
        double metric_error, pixel_error, icp_fitness, icp_rmse, icp_translation_delta_m,
            icp_rotation_delta_deg;
        bool weak, planar_degenerate, icp_used;
    };
    std::vector<Edge> edges;
    constexpr int min_inliers = 6;
    constexpr double ratio_test = 0.80;
    constexpr double max_metric_error_m = 0.025;
    constexpr double max_pair_translation_m = 2.0;
    int rejected_translation_edges = 0;
    int rejected_geometry_edges = 0;
    int planar_degenerate_edges = 0;
    int icp_accepted_edges = 0;
    int icp_rejected_edges = 0;
    auto point = [](const Frame &f, const cv::Point2f &p, Vec3 &out) {
        int cx = std::clamp(int(std::lround(p.x)), 0, f.depth.cols - 1);
        int cy = std::clamp(int(std::lround(p.y)), 0, f.depth.rows - 1);
        std::vector<float> values;
        for (int y = std::max(0, cy - 1); y <= std::min(f.depth.rows - 1, cy + 1); ++y)
            for (int x = std::max(0, cx - 1); x <= std::min(f.depth.cols - 1, cx + 1); ++x) {
                float z = f.depth.at<float>(y, x), q = f.quality.at<float>(y, x);
                if (std::isfinite(z) && z > 0 && std::isfinite(q) && q >= 0.2f)
                    values.push_back(z);
            }
        if (values.empty())
            return false;
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        double z = values[values.size() / 2];
        out = {(p.x - f.K.at<double>(0, 2)) * z / f.K.at<double>(0, 0),
               (p.y - f.K.at<double>(1, 2)) * z / f.K.at<double>(1, 1), z};
        return out.allFinite();
    };
    auto fit_rigid = [](const std::vector<Vec3> &a, const std::vector<Vec3> &b,
                        const std::vector<int> &indices, Mat4 &T) {
        if (indices.size() < 3)
            return false;
        Vec3 ca = Vec3::Zero(), cb = Vec3::Zero();
        for (int q : indices) {
            ca += a[q];
            cb += b[q];
        }
        ca /= indices.size();
        cb /= indices.size();
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        for (int q : indices)
            H += (a[q] - ca) * (b[q] - cb).transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d V = svd.matrixV(), R = V * svd.matrixU().transpose();
        if (R.determinant() < 0) {
            V.col(2) *= -1;
            R = V * svd.matrixU().transpose();
        }
        T = Mat4::Identity();
        T.block<3, 3>(0, 0) = R;
        T.block<3, 1>(0, 3) = cb - R * ca;
        return T.allFinite() && std::abs(R.determinant() - 1.0) < 1e-4;
    };
    cv::BFMatcher matcher(cv::NORM_L2);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (descs[i].rows < 2 || descs[j].rows < 2)
                continue;
            std::vector<std::vector<cv::DMatch>> forward, reverse;
            matcher.knnMatch(descs[i], descs[j], forward, 2);
            matcher.knnMatch(descs[j], descs[i], reverse, 2);
            std::vector<int> reverse_best(descs[j].rows, -1);
            for (const auto &m : reverse)
                if (m.size() == 2 && m[0].distance < ratio_test * m[1].distance)
                    reverse_best[m[0].queryIdx] = m[0].trainIdx;
            std::vector<cv::DMatch> mutual;
            std::vector<cv::Point2f> pi, pj;
            for (const auto &m : forward) {
                if (m.size() != 2 || m[0].distance >= ratio_test * m[1].distance ||
                    reverse_best[m[0].trainIdx] != m[0].queryIdx)
                    continue;
                mutual.push_back(m[0]);
                pi.push_back(keys[i][m[0].queryIdx].pt);
                pj.push_back(keys[j][m[0].trainIdx].pt);
            }
            if (mutual.size() < 8)
                continue;
            cv::Mat essential_mask;
            auto essential = cv::findEssentialMat(pi, pj, frames[i].K, cv::RANSAC, 0.999, 1.5,
                                                   essential_mask);
            if (essential.empty())
                continue;
            cv::Mat relative_rotation, relative_translation;
            if (cv::recoverPose(essential, pi, pj, frames[i].K, relative_rotation,
                                relative_translation, essential_mask) < 8)
                continue;
            cv::Mat homography_mask;
            cv::findHomography(pi, pj, cv::RANSAC, 2.0, homography_mask, 2000, 0.999);
            int essential_inliers = cv::countNonZero(essential_mask);
            int homography_inliers = homography_mask.empty() ? 0 : cv::countNonZero(homography_mask);
            bool planar_degenerate = homography_inliers >= std::max(8, int(0.9 * essential_inliers));
            if (planar_degenerate)
                ++planar_degenerate_edges;
            std::vector<Vec3> source, target;
            std::vector<cv::Point2f> source_pixels, target_pixels;
            for (int q = 0; q < int(mutual.size()); ++q) {
                if (!essential_mask.at<uchar>(q))
                    continue;
                Vec3 a, b;
                if (!point(frames[i], pi[q], a) || !point(frames[j], pj[q], b))
                    continue;
                source.push_back(a);
                target.push_back(b);
                source_pixels.push_back(pi[q]);
                target_pixels.push_back(pj[q]);
            }
            if (source.size() < min_inliers)
                continue;
            std::mt19937 rng(0x9e3779b9u + i * 131u + j * 977u);
            std::vector<int> best;
            Mat4 candidate;
            for (int iteration = 0; iteration < 600; ++iteration) {
                std::set<int> chosen;
                while (chosen.size() < 3)
                    chosen.insert(int(rng() % source.size()));
                std::vector<int> sample(chosen.begin(), chosen.end());
                if (!fit_rigid(source, target, sample, candidate))
                    continue;
                std::vector<int> inliers;
                for (int q = 0; q < int(source.size()); ++q)
                    if ((candidate.block<3, 3>(0, 0) * source[q] +
                         candidate.block<3, 1>(0, 3) - target[q])
                            .norm() < max_metric_error_m)
                        inliers.push_back(q);
                if (inliers.size() > best.size())
                    best = std::move(inliers);
            }
            if (best.size() < min_inliers || !fit_rigid(source, target, best, candidate)) {
                ++rejected_geometry_edges;
                continue;
            }
            std::vector<int> refined;
            double metric_error = 0, pixel_error = 0;
            std::set<int> cells_i, cells_j;
            for (int q = 0; q < int(source.size()); ++q) {
                auto transformed = candidate.block<3, 3>(0, 0) * source[q] +
                                   candidate.block<3, 1>(0, 3);
                double residual = (transformed - target[q]).norm();
                if (residual >= max_metric_error_m)
                    continue;
                refined.push_back(q);
                metric_error += residual;
                cv::Point2f projected(float(frames[j].K.at<double>(0, 0) * transformed.x() /
                                                transformed.z() + frames[j].K.at<double>(0, 2)),
                                      float(frames[j].K.at<double>(1, 1) * transformed.y() /
                                                transformed.z() + frames[j].K.at<double>(1, 2)));
                pixel_error += cv::norm(projected - target_pixels[q]);
                cells_i.insert(std::clamp(int(source_pixels[q].x / frames[i].rgb.cols * 4), 0, 3) +
                               4 * std::clamp(int(source_pixels[q].y / frames[i].rgb.rows * 4), 0, 3));
                cells_j.insert(std::clamp(int(target_pixels[q].x / frames[j].rgb.cols * 4), 0, 3) +
                               4 * std::clamp(int(target_pixels[q].y / frames[j].rgb.rows * 4), 0, 3));
            }
            if (refined.size() < min_inliers || !fit_rigid(source, target, refined, candidate))
                continue;
            metric_error /= refined.size();
            pixel_error /= refined.size();
            if (!candidate.allFinite() ||
                candidate.block<3, 1>(0, 3).norm() > max_pair_translation_m) {
                ++rejected_translation_edges;
                continue;
            }
            int coverage = std::min(cells_i.size(), cells_j.size());
            double icp_fitness = 0, icp_rmse = 0, icp_translation_delta = 0,
                   icp_rotation_delta = 0;
            bool icp_used = false;
            if (coverage >= 3 && icp_clouds[i]->points_.size() >= 100 &&
                icp_clouds[j]->points_.size() >= 100 &&
                geometry_clouds[i]->points_.size() >= 100 &&
                geometry_clouds[j]->points_.size() >= 100) {
                auto kernel = std::make_shared<open3d::pipelines::registration::HuberLoss>(0.005);
                auto coarse_icp = open3d::pipelines::registration::RegistrationICP(
                    *icp_clouds[i], *icp_clouds[j], 0.015, candidate,
                    open3d::pipelines::registration::TransformationEstimationPointToPlane(kernel),
                    open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, 25));
                auto geometry_kernel =
                    std::make_shared<open3d::pipelines::registration::CauchyLoss>(0.0035);
                auto fine_icp = open3d::pipelines::registration::RegistrationGeneralizedICP(
                    *geometry_clouds[i], *geometry_clouds[j], 0.008,
                    coarse_icp.transformation_,
                    open3d::pipelines::registration::TransformationEstimationForGeneralizedICP(
                        1e-3, geometry_kernel),
                    open3d::pipelines::registration::ICPConvergenceCriteria(1e-7, 1e-7, 45));
                auto forward = open3d::pipelines::registration::EvaluateRegistration(
                    *geometry_clouds[i], *geometry_clouds[j], 0.008, fine_icp.transformation_);
                auto reverse = open3d::pipelines::registration::EvaluateRegistration(
                    *geometry_clouds[j], *geometry_clouds[i], 0.008,
                    fine_icp.transformation_.inverse());
                auto delta = fine_icp.transformation_ * candidate.inverse();
                icp_translation_delta = delta.block<3, 1>(0, 3).norm();
                double cosine = std::clamp((delta.block<3, 3>(0, 0).trace() - 1.0) * 0.5, -1.0, 1.0);
                icp_rotation_delta = std::acos(cosine) * 180.0 / 3.14159265358979323846;
                icp_fitness = std::min(forward.fitness_, reverse.fitness_);
                icp_rmse = std::max(forward.inlier_rmse_, reverse.inlier_rmse_);
                const double minimum_fitness = planar_degenerate ? 0.30 : 0.18;
                if (fine_icp.transformation_.allFinite() && icp_fitness >= minimum_fitness &&
                    icp_rmse <= 0.0065 && icp_rmse <= coarse_icp.inlier_rmse_ * 1.05 &&
                    icp_translation_delta <= 0.020 && icp_rotation_delta <= 5.0) {
                    candidate = fine_icp.transformation_;
                    icp_used = true;
                    ++icp_accepted_edges;
                } else {
                    ++icp_rejected_edges;
                }
            }
            bool weak = !((j - i <= 3 && refined.size() >= 8 && coverage >= 3) ||
                          (refined.size() >= 12 && coverage >= 4));
            if (planar_degenerate && !icp_used)
                weak = true;
            edges.push_back({i, j, int(refined.size()), coverage, int(mutual.size()), candidate,
                             metric_error, pixel_error, icp_fitness, icp_rmse,
                             icp_translation_delta, icp_rotation_delta, weak,
                             planar_degenerate, icp_used});
        }
        progress(0.1 + double(i + 1) / n * 0.62,
                 "SIFT 匹配 " + std::to_string(i + 1) + "/" + std::to_string(n));
    }
    std::vector<Mat4> poses(n, Mat4::Identity());
    std::set<int> known{0};
    std::set<std::pair<int, int>> tree;
    while (int(known.size()) < n) {
        Edge *best = nullptr;
        for (auto &e : edges)
            if ((known.count(e.i) > 0) != (known.count(e.j) > 0))
                if (!best || std::tie(e.count, e.coverage) > std::tie(best->count, best->coverage))
                    best = &e;
        if (!best)
            throw std::runtime_error("SIFT/PnP matching graph disconnected: localized " +
                                     std::to_string(known.size()) + "/" + std::to_string(n) +
                                     " frames with " + std::to_string(edges.size()) +
                                     " accepted constraints. Add overlapping views or remove "
                                     "blurred frames; identity poses were not fabricated.");
        if (known.count(best->i)) {
            poses[best->j] = poses[best->i] * best->T.inverse();
            known.insert(best->j);
        } else {
            poses[best->i] = poses[best->j] * best->T;
            known.insert(best->i);
        }
        tree.insert({best->i, best->j});
    }
    open3d::pipelines::registration::PoseGraph graph;
    for (auto &T : poses)
        graph.nodes_.emplace_back(T);
    int optimized_edges = 0;
    for (auto &e : edges) {
        bool is_tree = tree.count({e.i, e.j});
        if (e.weak && !is_tree)
            continue;
        double weight = e.count / std::max(0.003, e.metric_error) * (e.weak ? 0.2 : 1.0);
        graph.edges_.emplace_back(e.i, e.j, e.T,
                                  Eigen::Matrix<double, 6, 6>::Identity() * weight, !is_tree);
        ++optimized_edges;
    }
    progress(0.78, "构建鲁棒位姿图，共 " + std::to_string(optimized_edges) + "/" +
                       std::to_string(edges.size()) + " 条约束");
    const auto tree_poses = poses;
    open3d::pipelines::registration::GlobalOptimization(
        graph, open3d::pipelines::registration::GlobalOptimizationLevenbergMarquardt(),
        open3d::pipelines::registration::GlobalOptimizationConvergenceCriteria(),
        open3d::pipelines::registration::GlobalOptimizationOption(0.02, 0.25, 1.0, 0));
    bool optimization_fallback = false;
    for (const auto &node : graph.nodes_) {
        const auto &T = node.pose_;
        const double determinant = T.block<3, 3>(0, 0).determinant();
        if (!T.allFinite() || std::abs(determinant - 1.0) > 0.05 ||
            T.block<3, 1>(0, 3).norm() > 10.0) {
            optimization_fallback = true;
            break;
        }
    }
    if (optimization_fallback) {
        for (int i = 0; i < n; ++i)
            graph.nodes_[i].pose_ = tree_poses[i];
        progress(0.9, "全局优化结果发散，已安全回退到连通生成树位姿");
    } else {
        progress(0.9, "Open3D Levenberg-Marquardt 全局优化完成");
    }
    json result = {{"backend", "C++ calibrated Essential + RGB-D 3D-3D RANSAC + geometry-aware point-to-plane/GICP + Open3D pose graph"},
                   {"frames", n},
                   {"edges", json::array()},
                   {"matching_thresholds",
                    {{"ratio_test", ratio_test},
                     {"min_inliers", min_inliers},
                     {"max_metric_error_m", max_metric_error_m},
                     {"max_pair_translation_m", max_pair_translation_m}}},
                   {"rejected_translation_edges", rejected_translation_edges},
                   {"rejected_geometry_edges", rejected_geometry_edges},
                   {"planar_degenerate_edges", planar_degenerate_edges},
                   {"icp_accepted_edges", icp_accepted_edges},
                   {"icp_rejected_edges", icp_rejected_edges},
                   {"icp_thresholds",
                    {{"coarse_voxel_size_m", 0.006},
                     {"geometry_voxel_size_m", 0.0035},
                     {"coarse_max_correspondence_m", 0.015},
                     {"fine_max_correspondence_m", 0.008},
                     {"max_symmetric_rmse_m", 0.0065},
                     {"max_translation_delta_m", 0.020},
                     {"max_rotation_delta_deg", 5.0},
                     {"planar_keep_fraction", 0.2},
                     {"curvature_threshold", 0.0025}}},
                   {"optimized_edges", optimized_edges},
                   {"optimization_fallback", optimization_fallback},
                   {"colmap_export", "pose-only camera format; not COLMAP SfM or rig BA"}};
    json output = json::object();
    auto sparse = out / "colmap/sparse/0";
    auto images = out / "colmap/images";
    fs::create_directories(sparse);
    fs::create_directories(images);
    std::ofstream cams(sparse / "cameras.txt"), imgs(sparse / "images.txt"),
        points(sparse / "points3D.txt");
    points << "# Pose-only export: initialize using fused_cloud.ply\n";
    cams.precision(17);
    imgs.precision(17);
    for (int i = 0; i < n; ++i) {
        auto &T = graph.nodes_[i].pose_;
        output[frames[i].id] = matrix_json(T);
        auto &f = frames[i];
        cams << i + 1 << " PINHOLE " << f.rgb.cols << " " << f.rgb.rows << " "
             << f.K.at<double>(0, 0) << " " << f.K.at<double>(1, 1) << " " << f.K.at<double>(0, 2)
             << " " << f.K.at<double>(1, 2) << "\n";
        Mat4 W = T.inverse();
        Eigen::Quaterniond q(W.block<3, 3>(0, 0));
        imgs << i + 1 << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
             << W(0, 3) << " " << W(1, 3) << " " << W(2, 3) << " " << i + 1 << " " << f.id
             << ".png\n\n";
        fs::copy_file(depth / f.id / "rgb.png", images / (f.id + ".png"));
    }
    for (auto &e : edges)
        result["edges"].push_back({{"i", e.i},
                                   {"j", e.j},
                                   {"inliers", e.count},
                                   {"coverage_cells", e.coverage},
                                   {"mutual_matches", e.mutual_matches},
                                   {"mean_3d_error_m", e.metric_error},
                                   {"mean_reprojection_px", e.pixel_error},
                                   {"planar_degenerate", e.planar_degenerate},
                                   {"icp_used", e.icp_used},
                                   {"icp_fitness", e.icp_fitness},
                                   {"icp_rmse_m", e.icp_rmse},
                                   {"icp_translation_delta_m", e.icp_translation_delta_m},
                                   {"icp_rotation_delta_deg", e.icp_rotation_delta_deg},
                                   {"weak", e.weak},
                                   {"used", !e.weak || tree.count({e.i, e.j})}});
    write_json(out / "poses.json", output);
    progress(1, "Pose graph and COLMAP export complete");
    return result;
}

void preview(const fs::path &out, const std::vector<Vec3> &xyz, const std::vector<Vec3> &rgb) {
    json p = {{"positions", json::array()},
              {"colors", json::array()},
              {"total_points", xyz.size()},
              {"unit", "m"}};
    size_t stride = std::max(size_t(1), (xyz.size() + 24999) / 25000);
    for (size_t i = 0; i < xyz.size(); i += stride)
        for (int a = 0; a < 3; ++a) {
            p["positions"].push_back(xyz[i][a]);
            p["colors"].push_back(rgb.size() == xyz.size() ? rgb[i][a] : 0.65);
        }
    p["sampled_points"] = p["positions"].size() / 3;
    write_json(out / "preview.json", p);
}
json fuse_cloud(const fs::path &scene, const fs::path &out, const json &req) {
    auto source = upstream(scene, req, "depth");
    auto poses = read_json(upstream(scene, req, "poses") / "poses.json");
    const auto cfg = req["config"];
    const double min_depth = cfg["crop_min_depth"], max_depth = cfg["crop_max_depth"];
    const double rx = cfg["roi_x"], ry = cfg["roi_y"], rw = cfg["roi_w"],
                 rh = cfg["roi_h"];
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    size_t frame_index = 0;
    for (auto &[ident, j] : poses.items()) {
        auto f = load_frame(source, ident);
        auto polygon_mask = frame_roi_mask(scene, ident, f.depth.cols, f.depth.rows);
        auto T = matrix(j);
        const int x0 = std::clamp(int(std::floor(rx * f.depth.cols)), 0, f.depth.cols - 1);
        const int y0 = std::clamp(int(std::floor(ry * f.depth.rows)), 0, f.depth.rows - 1);
        const int x1 = std::clamp(int(std::ceil((rx + rw) * f.depth.cols)), x0 + 1, f.depth.cols);
        const int y1 = std::clamp(int(std::ceil((ry + rh) * f.depth.rows)), y0 + 1, f.depth.rows);
        for (int y = y0; y < y1; y += 2)
            for (int x = x0; x < x1; x += 2) {
                auto p = unproject(f, x, y);
                if (p.z() < min_depth || p.z() > max_depth ||
                    f.quality.at<float>(y, x) < 0.2 || polygon_mask.at<uchar>(y, x) == 0)
                    continue;
                cloud->points_.push_back(T.block<3, 3>(0, 0) * p + T.block<3, 1>(0, 3));
                auto c = f.rgb.at<cv::Vec3b>(y, x);
                cloud->colors_.emplace_back(c[2] / 255., c[1] / 255., c[0] / 255.);
            }
        progress(double(++frame_index) / poses.size() * 0.8,
                 "融合掩膜点云 " + ident + " (" + std::to_string(frame_index) + "/" +
                     std::to_string(poses.size()) + ")");
    }
    if (cloud->points_.size() < 50)
        throw std::runtime_error("Too few valid points");
    double voxel = cfg["voxel_size"];
    cloud = cloud->VoxelDownSample(voxel);
    cloud->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxel * 5, 30));
    if (!open3d::io::WritePointCloud((out / "fused_cloud.ply").string(), *cloud))
        throw std::runtime_error("Point cloud write failed");
    preview(out, cloud->points_, cloud->colors_);
    progress(1, "体素降采样、法线估计与点云导出完成");
    return {{"points", cloud->points_.size()},
            {"voxel_size_m", voxel},
            {"crop", {{"min_depth_m", min_depth},
                      {"max_depth_m", max_depth},
                      {"roi", {rx, ry, rw, rh}}}},
            {"method", "Full-frame SIFT pose registration; per-frame polygon/depth-filtered voxel fusion"}};
}
json tsdf(const fs::path &scene, const fs::path &out, const json &req) {
    auto cfg = req["config"];
    double voxel = cfg["voxel_size"], trunc = cfg["sdf_trunc"], mindepth = cfg["min_depth"],
           maxdepth = cfg["max_depth"];
    open3d::pipelines::integration::ScalableTSDFVolume volume(
        voxel, trunc, open3d::pipelines::integration::TSDFVolumeColorType::RGB8);
    auto source = upstream(scene, req, "depth");
    auto poses = read_json(upstream(scene, req, "poses") / "poses.json");
    int index = 0;
    for (auto &[ident, j] : poses.items()) {
        auto f = load_frame(source, ident);
        auto polygon_mask = frame_roi_mask(scene, ident, f.depth.cols, f.depth.rows);
        open3d::geometry::Image color, depth;
        color.Prepare(f.rgb.cols, f.rgb.rows, 3, 1);
        depth.Prepare(f.rgb.cols, f.rgb.rows, 1, 4);
        for (int y = 0; y < f.rgb.rows; ++y)
            for (int x = 0; x < f.rgb.cols; ++x) {
                auto c = f.rgb.at<cv::Vec3b>(y, x);
                for (int a = 0; a < 3; ++a)
                    *color.PointerAt<uint8_t>(x, y, a) = c[2 - a];
                *depth.PointerAt<float>(x, y) =
                    f.quality.at<float>(y, x) >= 0.2 && polygon_mask.at<uchar>(y, x) != 0 &&
                            f.depth.at<float>(y, x) >= mindepth && f.depth.at<float>(y, x) <= maxdepth
                        ? f.depth.at<float>(y, x)
                        : 0;
            }
        auto rgbd = open3d::geometry::RGBDImage::CreateFromColorAndDepth(color, depth, 1.0,
                                                                         maxdepth, false);
        open3d::camera::PinholeCameraIntrinsic intr(f.rgb.cols, f.rgb.rows, f.K.at<double>(0, 0),
                                                    f.K.at<double>(1, 1), f.K.at<double>(0, 2),
                                                    f.K.at<double>(1, 2));
        volume.Integrate(*rgbd, intr, matrix(j).inverse());
        progress(double(++index) / poses.size() * 0.9,
                 "TSDF 融合 " + ident + " (" + std::to_string(index) + "/" +
                     std::to_string(poses.size()) + ")");
    }
    auto mesh = volume.ExtractTriangleMesh();
    if (mesh->vertices_.empty())
        throw std::runtime_error("TSDF produced no surface");
    mesh->ComputeVertexNormals();
    progress(0.95, "提取 TSDF 等值面与网格法线");
    open3d::io::WriteTriangleMesh((out / "surface_mesh.ply").string(), *mesh);
    open3d::io::WriteTriangleMesh((out / "surface_mesh.obj").string(), *mesh);
    preview(out, mesh->vertices_, mesh->vertex_colors_);
    auto m = read_json(out / "preview.json");
    if (mesh->vertices_.size() <= 150000) {
        m["positions"] = json::array();
        m["colors"] = json::array();
        m["indices"] = json::array();
        for (size_t i = 0; i < mesh->vertices_.size(); ++i)
            for (int a = 0; a < 3; ++a) {
                m["positions"].push_back(mesh->vertices_[i][a]);
                m["colors"].push_back(mesh->vertex_colors_.size() == mesh->vertices_.size()
                                          ? mesh->vertex_colors_[i][a]
                                          : 0.65);
            }
        for (auto &t : mesh->triangles_)
            for (int a = 0; a < 3; ++a)
                m["indices"].push_back(t[a]);
        write_json(out / "mesh_preview.json", m);
    }
    progress(1, "TSDF 网格与预览导出完成");
    return {{"vertices", mesh->vertices_.size()},
            {"triangles", mesh->triangles_.size()},
            {"voxel_size_m", voxel},
            {"sdf_trunc_m", trunc},
            {"backend", "Open3D C++ TSDF"},
            {"volume_persisted", false}};
}
json studio::execute_geometry(const fs::path &scene, const fs::path &out, const json &req) {
    auto stage = req["stage"].get<std::string>();
    progress_path = out / "progress.json";
    progress_phase = stage;
    progress(0, "准备 " + stage);
    if (stage == "import")
        return import_data(scene, req);
    verify_inputs(scene);
    if (stage == "depth")
        return stereo_depth(scene, out, req);
    if (stage == "poses")
        return estimate_poses(scene, out, req);
    if (stage == "cloud")
        return fuse_cloud(scene, out, req);
    if (stage == "tsdf")
        return tsdf(scene, out, req);
    throw std::runtime_error("Unknown native stage");
}
