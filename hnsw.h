#pragma once

#ifndef HNSW_H
#define HNSW_H

#include <vector>

static uint32_t ID = 0; // 用于生成唯一的key

class HNSWNode {
public:
    uint32_t id;

    uint32_t level;  // 当前节点的层级
    uint64_t key;
    std::vector<float> vec;
    
    // 每层中的邻居节点和与该邻居节点的距离（注意：这里的距离是余弦相似度，余弦相似度越大，距离越小）
    std::vector<std::vector<std::pair<float, HNSWNode *>>> neighbors;

    HNSWNode(uint32_t level, uint64_t key, const std::vector<float> &vec)
        : level(level), key(key), vec(vec) {
        neighbors.resize(level + 1);
        id = ID++;
    }

    bool operator<(const HNSWNode &other) const {
        return true;
    }
    bool operator>(const HNSWNode &other) const {
        return true;
    }
};

class HNSW {
private:
    uint32_t M;
    uint32_t M_max;
    uint32_t efConstruction;
    uint32_t m_L;

    HNSWNode *entry_point;

public:
    HNSW() {}
    HNSW(uint32_t M, uint32_t M_max, uint32_t efConstruction, uint32_t m_L)
        : M(M), M_max(M_max), efConstruction(efConstruction), m_L(m_L) {
        entry_point = nullptr;
    }

    std::vector<HNSWNode *> nodes;

    std::vector<std::pair<uint64_t, std::vector<float>>> deleted_nodes;

    int rand_level();

    void insert(uint64_t key, const std::vector<float> &vec);
    std::vector<uint64_t> search(std::string query, int k);

    void save_to_disk(const std::string &dir);
    void load_from_disk(const std::string &dir);
};

#endif // HNSW_H
