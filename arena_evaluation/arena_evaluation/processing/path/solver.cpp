#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>

// ABI of the C surface below. arena_evaluation/processing/path/theta_star.py verifies it
// before binding, so a stale library from an older search raises instead of returning a
// different route.
#define ARENA_THETA_STAR_SOLVER_ABI 1

extern "C" int arena_theta_star_solver_abi(void) {
    return ARENA_THETA_STAR_SOLVER_ABI;
}

using namespace std;

static bool line_of_sight(
    const uint8_t* grid, int width, int height,
    int x0, int y0, int x1, int y1
) {
    int ddx = abs(x1 - x0), ddy = abs(y1 - y0);
    int sx  = (x0 < x1) ? 1 : -1;
    int sy  = (y0 < y1) ? 1 : -1;
    int err = ddx - ddy;
    int x = x0, y = y0;
    while (true) {
        if (x < 0 || x >= width || y < 0 || y >= height) return false;
        int idx = y * width + x;
        if (grid[idx] > 0) return false; // solid wall blocks LoS
        if (x == x1 && y == y1) break;
        int e2 = 2 * err;
        if (e2 > -ddy) { err -= ddy; x += sx; }
        if (e2 <  ddx) { err += ddx; y += sy; }
    }
    return true;
}

struct Node {
    float g;         // cost from start
    float h;         // heuristic
    float f;         // g + h
    int   index;     // grid index
    int   parent_x;
    int   parent_y;

    bool operator>(const Node& other) const {
        return f > other.f;
    }
};

extern "C" {
    int solve_geometric_theta_star(
        const uint8_t* grid,
        int width,
        int height,
        float resolution,
        int start_x,
        int start_y,
        int goal_x,
        int goal_y,
        float* out_path_x,
        float* out_path_y,
        int max_path_len,
        int max_iterations,
        float* out_length
    ) {
        if (start_x < 0 || start_x >= width || start_y < 0 || start_y >= height) return 0;
        if (goal_x < 0 || goal_x >= width || goal_y < 0 || goal_y >= height) return 0;
        
        int start_idx = start_y * width + start_x;
        int goal_idx = goal_y * width + goal_x;

        if (grid[start_idx] > 0 || grid[goal_idx] > 0) return 0;

        vector<float> min_g(width * height, numeric_limits<float>::infinity());
        vector<int> parent(width * height, -1);
        
        priority_queue<Node, vector<Node>, greater<Node>> pq;

        min_g[start_idx] = 0.0f;
        parent[start_idx] = start_idx;
        
        float h_start = sqrtf(powf((float)(start_x - goal_x), 2) + powf((float)(start_y - goal_y), 2)) * resolution;
        pq.push({0.0f, h_start, h_start, start_idx, start_x, start_y});

        int ddx_arr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int ddy_arr[] = { 0, 0, -1, 1, -1,  1, -1, 1};
        float step_dist[] = {1.0f, 1.0f, 1.0f, 1.0f,
                              1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

        int iterations = 0;
        while (!pq.empty()) {
            if (iterations++ >= max_iterations) return -1;
            Node curr = pq.top();
            pq.pop();

            if (curr.index == goal_idx) {
                break;
            }

            if (curr.g > min_g[curr.index] + 1e-4f) {
                continue;
            }

            int cx = curr.index % width;
            int cy = curr.index / width;

            for (int dir = 0; dir < 8; ++dir) {
                int nx = cx + ddx_arr[dir];
                int ny = cy + ddy_arr[dir];
                
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                
                int nidx = ny * width + nx;
                if (grid[nidx] > 0) continue; // wall

                float ng;
                int n_px, n_py;
                
                if (line_of_sight(grid, width, height, curr.parent_x, curr.parent_y, nx, ny)) {
                    float dx = (nx - curr.parent_x) * resolution;
                    float dy = (ny - curr.parent_y) * resolution;
                    ng = min_g[curr.parent_y * width + curr.parent_x] + sqrtf(dx*dx + dy*dy);
                    n_px = curr.parent_x;
                    n_py = curr.parent_y;
                } else {
                    ng = curr.g + step_dist[dir] * resolution;
                    n_px = cx;
                    n_py = cy;
                }

                if (ng < min_g[nidx]) {
                    min_g[nidx] = ng;
                    parent[nidx] = n_py * width + n_px;
                    float nh = sqrtf(powf((float)(nx - goal_x), 2) + powf((float)(ny - goal_y), 2)) * resolution;
                    pq.push({ng, nh, ng + nh, nidx, n_px, n_py});
                }
            }
        }

        if (min_g[goal_idx] == numeric_limits<float>::infinity()) return 0;

        vector<pair<int, int>> path;
        int curr_idx = goal_idx;
        int max_steps = width * height;
        for (int steps = 0; curr_idx != start_idx; ++steps) {
            if (steps >= max_steps || parent[curr_idx] < 0 || parent[curr_idx] == curr_idx) return -2;
            path.push_back({curr_idx % width, curr_idx / width});
            curr_idx = parent[curr_idx];
        }
        path.push_back({start_x, start_y});
        
        reverse(path.begin(), path.end());
        
        int n_pts = min((int)path.size(), max_path_len);
        for (int i = 0; i < n_pts; ++i) {
            out_path_x[i] = (float)path[i].first;
            out_path_y[i] = (float)path[i].second;
        }
        
        if (out_length) {
            *out_length = min_g[goal_idx];
        }

        return n_pts;
    }
}
