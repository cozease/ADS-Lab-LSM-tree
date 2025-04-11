#include "skiplist.h"

double skiplist::my_rand() {
    return (double)rand() / RAND_MAX;
}

int skiplist::randLevel() {
    int level = 1;
    while (my_rand() < p && level < MAX_LEVEL)
        level++;
    return level;
}

void skiplist::insert(uint64_t key, const std::string &str) {
    int level = randLevel();
    if (level > curMaxL)
        curMaxL = level;
    std::vector<slnode *> update(level);
    slnode *p = head;
    for (int i = level - 1; i >= 0; --i) {
        while (p->nxt[i]->key < key) p = p->nxt[i];
        update[i] = p;
    }
    if (p->nxt[0]->key == key) {
        bytes += str.length() - p->nxt[0]->val.length();
        p->nxt[0]->val = str;
        p->nxt[0]->vec = embedding_single(str);
        return;
    }
    slnode *newNode = new slnode(key, str, NORMAL);
    for (int i = 0; i < level; ++i) {
        newNode->nxt[i] = update[i]->nxt[i];
        update[i]->nxt[i] = newNode;
    }
    bytes += 12 + str.length();
}

std::string skiplist::search(uint64_t key) {
    slnode *p = head;
    for (int i = curMaxL - 1; i >= 0; --i) {
        while (p->nxt[i]->key < key) p = p->nxt[i];
        if (p->nxt[i]->key == key) return p->nxt[i]->val;
    }
    return "";
}

bool skiplist::del(uint64_t key, uint32_t len) {

}

void skiplist::scan(uint64_t key1, uint64_t key2, std::vector<std::pair<uint64_t, std::string>> &list) {
    if (key1 >= key2) return;

    slnode *start = head, *end = head;
    for (int i = curMaxL - 1; i >= 0; --i) {
        while (start->nxt[i]->key < key1) start = start->nxt[i];
        while (end->nxt[i]->key < key2) end = end->nxt[i];
    }
    if (end->nxt[0]->key == key2) end = end->nxt[0];
    for (slnode *p = start; p != end; p = p->nxt[0]) {
        list.push_back(std::make_pair(p->nxt[0]->key, p->nxt[0]->val));
    }
}

struct memPair {
    uint64_t key;
    std::string val;
    float sim;
    memPair(uint64_t key, const std::string &val, float sim) {
        this->key = key;
        this->val = val;
        this->sim = sim;
    }
    bool operator>(const memPair &other) const {
        return sim > other.sim;
    }
};

std::vector<std::pair<float, std::pair<uint64_t, std::string>>> skiplist::search_knn(std::vector<float> query_vec, int k) {
    std::priority_queue<memPair, std::vector<memPair>, std::greater<memPair>> heap;
    slnode *p = head->nxt[0];
    while (p != tail) {
        float sim = common_embd_similarity_cos(query_vec.data(), p->vec.data(), query_vec.size());
        if (heap.size() < k) {
            heap.push(memPair(p->key, p->val, sim));
        } else {
            if (sim > heap.top().sim) {
                heap.pop();
                heap.push(memPair(p->key, p->val, sim));
            }
        }
        p = p->nxt[0];
    }
    std::vector<std::pair<float, std::pair<uint64_t, std::string>>> res;
    while (!heap.empty()) {
        memPair cur = heap.top();
        heap.pop();
        res.push_back(std::make_pair(cur.sim, std::make_pair(cur.key, cur.val)));
    }
    return res;
}

slnode *skiplist::lowerBound(uint64_t key) {

}

void skiplist::reset() {
    slnode *p = head->nxt[0];
    while (p != tail) {
        slnode *q = p;
        p = p->nxt[0];
        delete q;
    }
    bytes = 0;
    curMaxL = 1;
    for (int i = 0; i < MAX_LEVEL; ++i) {
        head->nxt[i] = tail;
    }
}

uint32_t skiplist::getBytes() {
    return bytes;
}

