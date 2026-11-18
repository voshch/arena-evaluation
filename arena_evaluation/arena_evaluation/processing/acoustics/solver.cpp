#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>

#ifndef PARETO_K
#define PARETO_K 2
#endif

using namespace std;

struct Label {
    float dist;        // accumulated Euclidean distance (metres)
    int   index;       // grid pixel index (y*width + x)
    float tl;          // accumulated transmission loss (dB)
    float cost;        // 20*log10(max(dist, mic_distance)) + tl (priority key)
    int   slot;        // frontier slot index (0 .. PARETO_K-1)
    int   parent_x;    // parent x for Theta*
    int   parent_y;    // parent y for Theta*
    float parent_dist; // parent distance for Theta*

    bool operator>(const Label& other) const {
        return cost > other.cost;
    }
};

static inline bool is_acoustic_barrier(const uint8_t* grid, const float* pixel_tl, int idx) {
    if (pixel_tl != nullptr) {
        return pixel_tl[idx] > 0.0f;
    }
    return grid[idx] > 0;
}

struct AcousticSolverWorkspace {
    static const int MAX_FRONTIER = PARETO_K;
    int capacity = 0;
    vector<float> min_cost;
    vector<uint8_t> target_status; // 0: none, 1: unsettled, 2: settled
    vector<float> frontier_dist;
    vector<float> frontier_tl;
    vector<uint8_t> frontier_used;
    vector<int> touched;
    vector<int> sat;
    int sat_w = 0;
    int sat_h = 0;

    void ensure_capacity(int N) {
        if (capacity < N) {
            capacity = N;
            min_cost.assign(N, numeric_limits<float>::infinity());
            target_status.assign(N, (uint8_t)0);
            frontier_dist.assign((size_t)N * MAX_FRONTIER, 0.0f);
            frontier_tl.assign((size_t)N * MAX_FRONTIER, 0.0f);
            frontier_used.assign((size_t)N * MAX_FRONTIER, (uint8_t)0);
            touched.clear();
        }
    }

    void ensure_sat(const uint8_t* grid, const float* pixel_tl, int width, int height) {
        sat_w = width + 1;
        sat_h = height + 1;
        sat.assign((size_t)sat_w * sat_h, 0);

        for (int y = 0; y < height; ++y) {
            int row_sum = 0;
            for (int x = 0; x < width; ++x) {
                int idx = y * width + x;
                int val = is_acoustic_barrier(grid, pixel_tl, idx) ? 1 : 0;
                row_sum += val;
                sat[(y + 1) * sat_w + (x + 1)] = sat[y * sat_w + (x + 1)] + row_sum;
            }
        }
    }

    inline bool has_obstacles_in_box(int x0, int y0, int x1, int y1) const {
        if (sat.empty()) return true;
        int min_x = min(x0, x1);
        int max_x = max(x0, x1);
        int min_y = min(y0, y1);
        int max_y = max(y0, y1);

        int count = sat[(max_y + 1) * sat_w + (max_x + 1)]
                  - sat[min_y * sat_w + (max_x + 1)]
                  - sat[(max_y + 1) * sat_w + min_x]
                  + sat[min_y * sat_w + min_x];
        return count > 0;
    }

    int insert_pareto(int idx, float dist, float tl, float mic_distance) {
        size_t base = (size_t)idx * MAX_FRONTIER;
        for (int s = 0; s < MAX_FRONTIER; ++s) {
            if (frontier_used[base + s] && frontier_dist[base + s] <= dist && frontier_tl[base + s] <= tl) {
                return -1;
            }
        }

        for (int s = 0; s < MAX_FRONTIER; ++s) {
            if (frontier_used[base + s] && dist <= frontier_dist[base + s] && tl <= frontier_tl[base + s]) {
                frontier_used[base + s] = 0;
            }
        }

        int slot = -1;
        for (int s = 0; s < MAX_FRONTIER; ++s) {
            if (!frontier_used[base + s]) {
                slot = s;
                break;
            }
        }
        if (slot < 0) {
            float new_cost = 20.0f * log10f(fmaxf(dist, mic_distance)) + tl;
            float worst_cost = -numeric_limits<float>::infinity();
            int worst = 0;
            for (int s = 0; s < MAX_FRONTIER; ++s) {
                float c = 20.0f * log10f(fmaxf(frontier_dist[base + s], mic_distance)) + frontier_tl[base + s];
                if (c > worst_cost) {
                    worst_cost = c;
                    worst = s;
                }
            }
            if (new_cost >= worst_cost) {
                return -1;
            }
            slot = worst;
            frontier_used[base + slot] = 0;
        }

        frontier_dist[base + slot] = dist;
        frontier_tl[base + slot] = tl;
        frontier_used[base + slot] = 1;
        return slot;
    }

    void reset(const vector<int>& target_indices = {}) {
        for (int idx : touched) {
            size_t base = (size_t)idx * MAX_FRONTIER;
            for (int s = 0; s < MAX_FRONTIER; ++s) {
                frontier_used[base + s] = 0;
            }
            min_cost[idx] = numeric_limits<float>::infinity();
            target_status[idx] = 0;
        }
        touched.clear();
        for (int tidx : target_indices) {
            if (tidx >= 0 && tidx < capacity) {
                target_status[tidx] = 0;
            }
        }
    }
};

static thread_local AcousticSolverWorkspace ws;

static inline bool line_of_sight(
    const uint8_t* grid, const float* pixel_tl, int width, int height,
    int x0, int y0, int x1, int y1
) {
    if (x0 < 0 || x0 >= width || y0 < 0 || y0 >= height ||
        x1 < 0 || x1 >= width || y1 < 0 || y1 >= height) {
        return false;
    }

    if (!ws.has_obstacles_in_box(x0, y0, x1, y1)) {
        return true;
    }

    int ddx = abs(x1 - x0), ddy = abs(y1 - y0);
    int sx  = (x0 < x1) ? 1 : -1;
    int sy  = (y0 < y1) ? 1 : -1;
    int err = ddx - ddy;
    int x = x0, y = y0;
    while (true) {
        int idx = y * width + x;
        if (is_acoustic_barrier(grid, pixel_tl, idx)) return false;
        if (x == x1 && y == y1) break;
        int e2 = 2 * err;
        int nx = x, ny = y;
        if (e2 > -ddy) { err -= ddy; nx += sx; }
        if (e2 <  ddx) { err += ddx; ny += sy; }
        // A diagonal step may not squeeze between two barrier cells that touch at a corner, as in the Dijkstra step.
        if (nx != x && ny != y && is_acoustic_barrier(grid, pixel_tl, y * width + nx) &&
            is_acoustic_barrier(grid, pixel_tl, ny * width + x)) {
            return false;
        }
        x = nx;
        y = ny;
    }
    return true;
}

template <bool EarlyExit>
static inline void run_acoustic_dijkstra(
    const uint8_t* grid,
    int width,
    int height,
    float resolution,
    int start_ix,
    int start_iy,
    float wall_tl,
    float mic_distance,
    const float* pixel_tl,
    int num_unsettled_targets
) {
    int start_idx = start_iy * width + start_ix;

    auto pixel_tl_of = [&](int idx) -> float {
        if (pixel_tl != nullptr) return pixel_tl[idx];
        return grid[idx] > 0 ? wall_tl : 0.0f;
    };

    bool start_is_barrier = is_acoustic_barrier(grid, pixel_tl, start_idx);
    float start_tl    = start_is_barrier ? pixel_tl_of(start_idx) : 0.0f;
    float init_cost   = 20.0f * log10f(mic_distance) + start_tl;
    int start_slot = ws.insert_pareto(start_idx, 0.0f, start_tl, mic_distance);
    ws.min_cost[start_idx] = init_cost;
    ws.touched.push_back(start_idx);

    priority_queue<Label, vector<Label>, greater<Label>> pq;
    pq.push({0.0f, start_idx, start_tl, init_cost, start_slot,
             start_ix, start_iy, 0.0f});

    const int ddx_arr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
    const int ddy_arr[] = { 0, 0, -1, 1, -1,  1, -1, 1};
    const float step_dist[] = {1.0f, 1.0f, 1.0f, 1.0f,
                               1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

    int remaining = num_unsettled_targets;

    while (!pq.empty()) {
        Label curr = pq.top();
        pq.pop();

        size_t c_base = (size_t)curr.index * ws.MAX_FRONTIER;
        if (curr.slot < 0 || !ws.frontier_used[c_base + curr.slot] ||
            ws.frontier_dist[c_base + curr.slot] != curr.dist ||
            ws.frontier_tl[c_base + curr.slot] != curr.tl) {
            continue;
        }

        if constexpr (EarlyExit) {
            if (ws.target_status[curr.index] == 1) {
                ws.target_status[curr.index] = 2;
                remaining--;
                if (remaining <= 0) break;
            }
        }

        int cx = curr.index % width;
        int cy = curr.index / width;
        int par_x    = curr.parent_x;
        int par_y    = curr.parent_y;
        float par_dist = curr.parent_dist;

        for (int dir = 0; dir < 8; ++dir) {
            int nx = cx + ddx_arr[dir];
            int ny = cy + ddy_arr[dir];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            int nidx = ny * width + nx;
            float ndist, new_par_dist, next_tl_contrib;
            int new_par_x, new_par_y;

            if (line_of_sight(grid, pixel_tl, width, height, par_x, par_y, nx, ny)) {
                float fdx = (float)(nx - par_x) * resolution;
                float fdy = (float)(ny - par_y) * resolution;
                ndist         = par_dist + sqrtf(fdx * fdx + fdy * fdy);
                new_par_x     = par_x;
                new_par_y     = par_y;
                new_par_dist  = par_dist;
                next_tl_contrib = 0.0f;
            } else {
                // No squeezing between barriers: a diagonal step may not pass a barrier corner where both cells
                // beside it are barrier, otherwise a path could cross fewer cells of a wall end or door edge.
                if (dir >= 4 && is_acoustic_barrier(grid, pixel_tl, cy * width + nx) &&
                    is_acoustic_barrier(grid, pixel_tl, ny * width + cx)) {
                    continue;
                }
                ndist         = curr.dist + step_dist[dir] * resolution;
                new_par_x     = cx;
                new_par_y     = cy;
                new_par_dist  = curr.dist;
                // Every barrier cell entered costs its own per-cell TL, so travelling inside a barrier and
                // crossing touching barriers are not free. Walls and doors are normalised per cell upstream.
                float next_tl_val = pixel_tl_of(nidx);
                next_tl_contrib   = next_tl_val > 0.0f ? next_tl_val : 0.0f;
            }

            float new_tl = curr.tl + next_tl_contrib;
            int slot = ws.insert_pareto(nidx, ndist, new_tl, mic_distance);
            if (slot < 0) continue;

            if (ws.min_cost[nidx] == numeric_limits<float>::infinity()) {
                ws.touched.push_back(nidx);
            }
            float ncost  = 20.0f * log10f(fmaxf(ndist, mic_distance)) + new_tl;
            if (ncost < ws.min_cost[nidx]) {
                ws.min_cost[nidx] = ncost;
            }
            pq.push({ndist, nidx, new_tl, ncost, slot, new_par_x, new_par_y, new_par_dist});
        }
    }
}

extern "C" {
    void solve_acoustic_field(
        const uint8_t* grid,
        int width,
        int height,
        float resolution,
        float start_x,
        float start_y,
        const float* target_xs,
        const float* target_ys,
        int num_targets,
        float wall_tl,
        float mic_distance,
        const float* pixel_tl,
        float* out_attenuations
    ) {
        for (int i = 0; i < num_targets; ++i) {
            out_attenuations[i] = numeric_limits<float>::infinity();
        }

        int start_ix = (int)round(start_x);
        int start_iy = (int)round(start_y);
        if (start_ix < 0 || start_ix >= width || start_iy < 0 || start_iy >= height) return;

        const int N = width * height;
        ws.ensure_capacity(N);
        ws.ensure_sat(grid, pixel_tl, width, height);

        int num_unsettled = 0;
        vector<int> target_indices(num_targets);
        for (int i = 0; i < num_targets; ++i) {
            int tx = (int)round(target_xs[i]);
            int ty = (int)round(target_ys[i]);
            if (tx >= 0 && tx < width && ty >= 0 && ty < height) {
                if (line_of_sight(grid, pixel_tl, width, height, start_ix, start_iy, tx, ty)) {
                    float fdx = (float)(tx - start_ix) * resolution;
                    float fdy = (float)(ty - start_iy) * resolution;
                    float dist = sqrtf(fdx * fdx + fdy * fdy);
                    out_attenuations[i] = 20.0f * log10f(fmaxf(dist, mic_distance));
                    target_indices[i] = -1;
                    continue;
                }

                int tidx = ty * width + tx;
                target_indices[i] = tidx;
                if (ws.target_status[tidx] == 0) {
                    ws.target_status[tidx] = 1;
                    num_unsettled++;
                }
            } else {
                target_indices[i] = -1;
            }
        }

        if (num_unsettled > 0) {
            run_acoustic_dijkstra<true>(
                grid, width, height, resolution,
                start_ix, start_iy, wall_tl, mic_distance, pixel_tl,
                num_unsettled
            );
        }

        for (int i = 0; i < num_targets; ++i) {
            int tidx = target_indices[i];
            if (tidx != -1 && ws.min_cost[tidx] != numeric_limits<float>::infinity()) {
                out_attenuations[i] = ws.min_cost[tidx];
            }
        }

        ws.reset(target_indices);
    }

    void solve_acoustic_grid(
        const uint8_t* grid,
        int width,
        int height,
        float resolution,
        float start_x,
        float start_y,
        float wall_tl,
        float mic_distance,
        const float* pixel_tl,
        float* out_field
    ) {
        if (width <= 0 || height <= 0) return;
        const int N = width * height;
        int start_ix = (int)round(start_x);
        int start_iy = (int)round(start_y);
        if (start_ix < 0 || start_ix >= width || start_iy < 0 || start_iy >= height) {
            std::fill_n(out_field, N, numeric_limits<float>::infinity());
            return;
        }

        ws.ensure_capacity(N);
        ws.ensure_sat(grid, pixel_tl, width, height);

        run_acoustic_dijkstra<false>(
            grid, width, height, resolution,
            start_ix, start_iy, wall_tl, mic_distance, pixel_tl,
            0
        );
        std::copy_n(ws.min_cost.data(), N, out_field);
        ws.reset();
    }
}
