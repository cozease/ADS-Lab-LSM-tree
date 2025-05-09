#include "hnsw.h"

#include <unordered_set>
#include <cstdlib>
#include <chrono>
#include <iostream>

#include "utils.h"
#include "embedding.h"

// int HNSW::rand_level() {
//     // int level = 0;
//     // while (static_cast<double>(rand()) / RAND_MAX < 0.5 && level < m_L) {
//     //     level++;
//     // }
//     // return level;

//     // 随机生成层数
//     return rand() % (m_L + 1);
// }

int HNSW::rand_level() {
    float r = static_cast<float>(rand()) / RAND_MAX;
    uint32_t level = static_cast<uint32_t>(-log(r) * m_L);
    return std::min(level, m_L);
}

void HNSW::insert(uint64_t key, const std::vector<float> &vec) {
    int level = rand_level();
    HNSWNode *node = new HNSWNode(level, key, vec);
    nodes.push_back(node);

    // 如果 HNSW 为空，将新节点设置为 entry_point
    if (!entry_point) {
        entry_point = node;
        return;
    }

    // 第一步：自顶层到 level+1 层，在每一层导航到与待插入节点最近的节点作为下一层的入口节点
    HNSWNode *cur = entry_point;
    for (int i = m_L; i > level; --i) {
        // 如果当前层在 entry_point 的层级之上，则跳过
        if (i > entry_point->level) continue;

        // 否则，找到当前层的最近邻作为下一层的入口
        float max_sim = common_embd_similarity_cos(vec.data(), cur->vec.data(), vec.size());
        while (true) {
            bool flag = false; // 用于标记是否找到更近的邻居

            // 逐个搜索当前节点的所有邻居
            for (auto neighbor : cur->neighbors[i]) {
                float sim = common_embd_similarity_cos(vec.data(), neighbor.second->vec.data(), vec.size());

                // 如果该邻居与待插入节点的距离比当前的最小距离还小（即余弦相似度更大），则更新最小距离，并把该邻居设为 cur
                if (sim > max_sim) {
                    max_sim = sim;
                    cur = neighbor.second;
                    flag = true;
                }
            }

            // 如果没有找到更近的邻居，则退出循环
            if (!flag) break;
        }
    }

    // 第二步：自第 level 层到底层，维护当前层与待插入节点最近的 efConstruction 个邻居，选取不超过 M 个连接
    for (int i = level; i >= 0; --i) {
        if (i > entry_point->level) continue;

        // std::vector<std::pair<HNSWNode *, float>> candidates(efConstruction);
        std::priority_queue<
            std::pair<float, HNSWNode *>,
            std::vector<std::pair<float, HNSWNode *>>,
            std::greater<std::pair<float, HNSWNode *>>> candidates;  // 用一个小顶堆来存储候选节点
        float max_sim = common_embd_similarity_cos(vec.data(), cur->vec.data(), vec.size());
        candidates.push(std::make_pair(max_sim, cur));
        while (true) {
            bool flag = false;

            for (auto neighbor : cur->neighbors[i]) {
                float sim = common_embd_similarity_cos(vec.data(), neighbor.second->vec.data(), vec.size());

                // 把搜索过的邻居都放入候选
                if (candidates.size() < efConstruction) {
                    candidates.push(std::make_pair(sim, neighbor.second));
                } else {
                    if (sim > candidates.top().first) {
                        candidates.pop();
                        candidates.push(std::make_pair(sim, neighbor.second));
                    }
                }

                if (sim > max_sim) {
                    max_sim = sim;
                    cur = neighbor.second;
                    flag = true;
                }
            }

            if (!flag) break;
        }

        // 根据距离从小到大选出不超过 M 个邻居
        std::vector<std::pair<float, HNSWNode *>> neighbors;
        while (!candidates.empty()) {
            neighbors.push_back(candidates.top());
            candidates.pop();
        }
        int cnt = 0;  // 记录已连接的邻居数，确保不超过 M 个
        for (auto it = neighbors.rbegin(); it != neighbors.rend() && cnt < M; ++it) {
            if (it->second->neighbors[i].size() < M_max) {  // 如果该邻居节点已连接的邻居数不超过 M_max 个，则直接连接
                it->second->neighbors[i].push_back(std::make_pair(it->first, node));
                node->neighbors[i].push_back(std::make_pair(it->first, it->second));
                ++cnt;
            } else {  // 否则，找到该节点的最远邻居
                auto max_it = std::min_element(it->second->neighbors[i].begin(), it->second->neighbors[i].end());
                if (it->first > max_it->first) {  // 如果待插入节点与该节点的距离小于该节点与最远邻居的距离，则替换最远邻居
                    node->neighbors[i].push_back(std::make_pair(it->first, it->second));

                    // 确保邻居之间互相删掉了
                    max_it->second->neighbors[i].erase(std::find_if(max_it->second->neighbors[i].begin(), 
                        max_it->second->neighbors[i].end(),
                        [&](const auto& p) { return p.second == it->second; }));
                    it->second->neighbors[i].erase(max_it);
                
                    it->second->neighbors[i].push_back(std::make_pair(it->first, node));
                    ++cnt;
                }
            }
        }
    }

    // 如果待插入节点的 level 大于 entry_point 的 level，则更新 entry_point
    if (level > entry_point->level) {
        entry_point = node;
    }
}

// std::vector<std::pair<uint64_t, std::string>> HNSW::search(std::string query, int k) {
//     std::vector<float> query_vec = embedding_single(query);
    
//     // 自顶层向底层逐层搜索，导航到离待搜索节点最近的节点
//     HNSWNode *cur = entry_point;
//     for (int i = entry_point->level; i >= 1; --i) {
//         float max_sim = common_embd_similarity_cos(query_vec.data(), cur->vec.data(), query_vec.size());
//         while (true) {
//             bool flag = false;

//             for (auto neighbor : cur->neighbors[i]) {
//                 float sim = common_embd_similarity_cos(query_vec.data(), neighbor.second->vec.data(), query_vec.size());

//                 if (sim > max_sim) {
//                     max_sim = sim;
//                     cur = neighbor.second;
//                     flag = true;
//                 }
//             }

//             if (!flag) break;
//         }
//     }

//     // 在最底层进行 k 近邻搜索
//     std::priority_queue<
//         std::pair<float, HNSWNode *>,
//         std::vector<std::pair<float, HNSWNode *>>,
//         std::greater<std::pair<float, HNSWNode *>>> candidates;  // 用一个小顶堆来存储候选节点
//     float max_sim = common_embd_similarity_cos(query_vec.data(), cur->vec.data(), query_vec.size());
//     candidates.push(std::make_pair(max_sim, cur));
//     while (true) {
//         bool flag = false;

//         for (auto node : cur->neighbors[0]) {
//             float sim = common_embd_similarity_cos(query_vec.data(), node.second->vec.data(), query_vec.size());

//             // 把搜索过的邻居都放入候选
//             if (candidates.size() < k) {
//                 candidates.push(std::make_pair(sim, node.second));
//             } else {
//                 if (sim > candidates.top().first) {
//                     candidates.pop();
//                     candidates.push(std::make_pair(sim, node.second));
//                 }
//             }

//             if (sim > max_sim) {
//                 max_sim = sim;
//                 cur = node.second;
//                 flag = true;
//             }
//         }

//         if (!flag) break;
//     }
//     std::vector<std::pair<uint64_t, std::string>> ans;
//     while (!candidates.empty()) {
//         auto cur = candidates.top();
//         candidates.pop();
//         ans.push_back(std::make_pair(cur.second->key, cur.second->val));
//     }
//     std::reverse(ans.begin(), ans.end());
//     return ans;
//     // std::unordered_set<uint64_t> visited;  // 用哈希表记录已访问的节点
//     // std::priority_queue<std::pair<float, HNSWNode *>> heap;  // 用一个大顶堆来存储候选节点
//     // std::vector<std::pair<float, HNSWNode *>> res;
//     // heap.push(std::make_pair(common_embd_similarity_cos(query_vec.data(), cur->vec.data(), query_vec.size()), cur));
//     // visited.insert(cur->key);
//     // while (!heap.empty() && heap.size() < efConstruction) {
//     //     auto cur_res = heap.top();
//     //     heap.pop();
//     //     res.push_back(cur_res); 

//     //     for (auto neighbor : cur_res.second->neighbors[0]) {
//     //         if (!visited.contains(neighbor.second->key)) {  // 如果该节点未被访问过，则加入候选
//     //             float sim = common_embd_similarity_cos(query_vec.data(), neighbor.second->vec.data(), query_vec.size());
//     //             heap.push(std::make_pair(sim, neighbor.second));
//     //             visited.insert(neighbor.second->key);
//     //         }
//     //     }
//     // }
//     // std::sort(res.begin(), res.end(), std::greater<std::pair<float, HNSWNode *>>());
//     // std::vector<std::pair<uint64_t, std::string>> ans;
//     // for (int i = 0; i < k; ++i) {
//     //     ans.push_back(std::make_pair(res[i].second->key, res[i].second->val));
//     // }
//     // return ans;
// }

std::vector<uint64_t> HNSW::search(std::string query, int k) {
    // auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> query_vec = embedding_single(query);
    // auto end = std::chrono::high_resolution_clock::now();
    // long long embedding_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // static int cnt = 0;
    // ++cnt;
    // static long long TIME = 0;
    // TIME += embedding_duration;
    // if (cnt == 120) std::cout << "average embedding time: " << (double)TIME / cnt << "ms" << std::endl;
    
    // 第一阶段：多路径下降
    std::priority_queue<std::pair<float, HNSWNode *>> candidates;
    candidates.push(std::make_pair(
        common_embd_similarity_cos(query_vec.data(), entry_point->vec.data(), query_vec.size()),
        entry_point)
    );
        
    for (int level = entry_point->level; level >= 1; --level) {
        // 保持多个候选路径
        std::priority_queue<std::pair<float, HNSWNode *>> next_candidates;
        while (!candidates.empty()) {
            auto current = candidates.top();
            candidates.pop();
            
            for (auto& neighbor : current.second->neighbors[level]) {
                float sim = common_embd_similarity_cos(query_vec.data(), 
                    neighbor.second->vec.data(), query_vec.size());
                next_candidates.push(std::make_pair(sim, neighbor.second));
            }
        }
        candidates = next_candidates;
    }
    
    // 第二阶段：底层精确搜索
    std::unordered_set<uint64_t> visited;
    std::priority_queue<std::pair<float, HNSWNode *>,
        std::vector<std::pair<float, HNSWNode *>>,
        std::greater<std::pair<float, HNSWNode *>>> result;
        
    while (!candidates.empty()  && visited.size() < efConstruction) {
        auto current = candidates.top();
        candidates.pop();
        
        if (!visited.contains(current.second->key)) {
            visited.insert(current.second->key);
            
            // 忽略已被删除的节点
            if (std::find_if(deleted_nodes.begin(), deleted_nodes.end(),
                [&current](const std::pair<uint64_t, std::vector<float>> &node) {
                    bool flag = true;
                    const float epsilon = 1e-6f;
                    for (size_t i = 0; i < current.second->vec.size(); ++i) {
                        if (fabs(node.second[i] - current.second->vec[i]) > epsilon) {
                            flag = false;
                            break;
                        }
                    }
                    return flag && node.first == current.second->key;
                }) == deleted_nodes.end()) {
                if (result.size() < k) {
                    result.push(current);
                } else if (current.first > result.top().first) {
                    result.pop();
                    result.push(current);
                }
            }
            
            // 扩展邻居
            for (auto& neighbor : current.second->neighbors[0]) {
                if (!visited.contains(neighbor.second->key)) {
                    float sim = common_embd_similarity_cos(query_vec.data(), 
                        neighbor.second->vec.data(), query_vec.size());
                    candidates.push(std::make_pair(sim, neighbor.second));
                }
            }
        }
    }
    
    // 收集结果
    std::vector<uint64_t> ans;
    while (!result.empty()) {
        auto current = result.top();
        result.pop();
        ans.push_back(current.second->key);
    }
    std::reverse(ans.begin(), ans.end());
    return ans;
}

void HNSW::save_to_disk(const std::string &dir) {
    if (utils::dirExists(dir)) {
        utils::rmdir(dir.data());
    }
    utils::mkdir(dir.data());
    FILE *file;

    // 写入全局参数文件
    std::string global_header_path = dir + "global_header.bin";
    file = fopen(global_header_path.c_str(), "wb");
    fseek(file, 0, SEEK_SET);
    fwrite(&M, 4, 1, file);
    fwrite(&M_max, 4, 1, file);
    fwrite(&efConstruction, 4, 1, file);
    fwrite(&m_L, 4, 1, file);
    uint32_t entry_id = entry_point->id;
    fwrite(&entry_id, 4, 1, file);
    uint32_t num_nodes = nodes.size();
    fwrite(&num_nodes, 4, 1, file);
    uint32_t dim = 768;
    fwrite(&dim, 4, 1, file);
    fflush(file);
    fclose(file);

    // 写入被删除的节点数据
    std::string deleted_nodes_path = dir + "deleted_nodes.bin";
    file = fopen(deleted_nodes_path.c_str(), "wb");
    fseek(file, 0, SEEK_SET);
    for (auto &node : deleted_nodes) {
        fwrite(&node.first, 8, 1, file);
        fwrite(node.second.data(), 4, dim, file);
    }
    fflush(file);
    fclose(file);

    // 写入每个节点的数据
    std::string nodes_dir = dir + "nodes/";
    utils::mkdir(nodes_dir.data());
    for (auto &node : nodes) {
        std::string node_dir = nodes_dir + std::to_string(node->id) + "/";
        utils::mkdir(node_dir.data());
        
        // 写入节点的基本信息
        std::string node_header_path = node_dir + "header.bin";
        file = fopen(node_header_path.c_str(), "wb");
        fseek(file, 0, SEEK_SET);
        fwrite(&node->level, 4, 1, file);
        fwrite(&node->key, 8, 1, file);
        fwrite(node->vec.data(), 4, dim, file);
        fflush(file);
        fclose(file);

        // 写入邻接表
        std::string neighbors_dir = node_dir + "edges/";
        utils::mkdir(neighbors_dir.data());
        for (int i = 0; i <= node->level; ++i) {
            std::string neighbors_path = neighbors_dir + std::to_string(i) + ".bin";
            file = fopen(neighbors_path.c_str(), "wb");
            fseek(file, 0, SEEK_SET);
            uint32_t num_neighbors = node->neighbors[i].size();
            fwrite(&num_neighbors, 4, 1, file);
            for (auto &neighbor : node->neighbors[i]) {
                fwrite(&neighbor.second->id, 4, 1, file);
                fwrite(&neighbor.first, 4, 1, file);
            }
            fflush(file);
            fclose(file);
        }
    }
}

void HNSW::load_from_disk(const std::string &dir) {
    // 读取全局参数文件
    std::string global_header_path = dir + "global_header.bin";
    FILE *file = fopen(global_header_path.c_str(), "rb");
    fseek(file, 0, SEEK_SET);
    fread(&M, 4, 1, file);
    fread(&M_max, 4, 1, file);
    fread(&efConstruction, 4, 1, file);
    fread(&m_L, 4, 1, file);
    uint32_t entry_id;
    fread(&entry_id, 4, 1, file);
    uint32_t num_nodes;
    fread(&num_nodes, 4, 1, file);
    uint32_t dim;
    fread(&dim, 4, 1, file);
    fclose(file);

    // 读取被删除的节点数据
    std::string deleted_nodes_path = dir + "deleted_nodes.bin";
    file = fopen(deleted_nodes_path.c_str(), "rb");
    fseek(file, 0, SEEK_SET);
    while (true) {
        // 如果到了文件尾，则退出
        if (feof(file)) break;

        uint64_t key;
        fread(&key, sizeof(uint64_t), 1, file);
        std::vector<float> vec(dim);
        fread(vec.data(), sizeof(float), dim, file);
        deleted_nodes.push_back(std::make_pair(key, vec));
    }
    fclose(file);

    // 读取每个节点的数据
    std::string nodes_dir = dir + "nodes/";
    for (uint32_t i = 0; i < num_nodes; ++i) {
        std::string node_dir = nodes_dir + std::to_string(i) + "/";
        
        // 读取节点的基本信息
        std::string node_header_path = node_dir + "header.bin";
        file = fopen(node_header_path.c_str(), "rb");
        fseek(file, 0, SEEK_SET);
        uint32_t level;
        uint64_t key;
        fread(&level, sizeof(uint32_t), 1, file);
        fread(&key, sizeof(uint64_t), 1, file);
        std::vector<float> vec(dim);
        fread(vec.data(), sizeof(float), dim, file);
        fclose(file);

        // 创建节点对象
        HNSWNode *node = new HNSWNode(level, key, vec);
        nodes.push_back(node);
    }

    // 读取邻接表
    for (auto &node : nodes) {
        std::string neighbors_dir = nodes_dir + std::to_string(node->id) + "/edges/";
        for (int i = 0; i <= node->level; ++i) {
            std::string neighbors_path = neighbors_dir + std::to_string(i) + ".bin";
            file = fopen(neighbors_path.c_str(), "rb");
            fseek(file, 0, SEEK_SET);
            uint32_t num_neighbors;
            fread(&num_neighbors, sizeof(uint32_t), 1, file);
            for (uint32_t j = 0; j < num_neighbors; ++j) {
                uint32_t neighbor_id;
                float distance;
                fread(&neighbor_id, sizeof(uint32_t), 1, file);
                fread(&distance, sizeof(float), 1, file);
                node->neighbors[i].push_back(std::make_pair(distance, nodes[neighbor_id]));
            }
            fclose(file);
        }
    }

    // 设置 entry_point
    entry_point = nodes[entry_id];
}
