#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace std;

static bool line_of_sight(
    const uint8_t* grid, const float* pixel_tl, int width, int height,
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
        if (grid[idx] > 0) return false;                               // solid wall blocks LoS
        if (pixel_tl != nullptr && pixel_tl[idx] > 0.0f) return false; // TL barrier (e.g. closed door) blocks LoS
        if (x == x1 && y == y1) break;
        int e2 = 2 * err;
        if (e2 > -ddy) { err -= ddy; x += sx; }
        if (e2 <  ddx) { err += ddx; y += sy; }
    }
    return true;
}

struct Label {
    float dist;        // accumulated Euclidean distance (metres) to THIS node
    int   walls;       // number of air->wall transitions so far
    int   index;       // grid pixel index (y*width + x)
    float tl;          // accumulated transmission loss (dB) along the path
    float cost;        // 20*log10(max(dist, mic_distance)) + tl  (priority key)
    int   parent_x;    // pixel x of THIS label's parent node (for Theta*)
    int   parent_y;    // pixel y of THIS label's parent node (for Theta*)
    float parent_dist; // accumulated distance at the parent node (for Theta*)

    bool operator>(const Label& other) const {
        return cost > other.cost;
    }
};

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
        if (start_ix < 0 || start_ix >= width || start_iy < 0 || start_iy >= height) {
            return;
        }

        const int MAX_WALLS = 20;
        const int N = width * height;

        vector<vector<float>> min_dist(N, vector<float>(MAX_WALLS, numeric_limits<float>::infinity()));
        vector<float> min_cost(N, numeric_limits<float>::infinity());

        priority_queue<Label, vector<Label>, greater<Label>> pq;

        int start_idx = start_iy * width + start_ix;
        bool start_is_wall = grid[start_idx] > 0;

        auto pixel_tl_of = [&](int idx) -> float {
            if (pixel_tl != nullptr) {
                return pixel_tl[idx];
            }
            return grid[idx] > 0 ? wall_tl : 0.0f;
        };

        float start_tl    = start_is_wall ? pixel_tl_of(start_idx) : 0.0f;
        float init_cost   = 20.0f * log10f(mic_distance) + start_tl;
        int   start_walls = start_is_wall ? 1 : 0;
        min_dist[start_idx][start_walls] = 0.0f;
        min_cost[start_idx] = init_cost;
        pq.push({0.0f, start_walls, start_idx, start_tl, init_cost,
                 start_ix, start_iy, 0.0f});

        vector<int> target_indices(num_targets);
        for (int i = 0; i < num_targets; ++i) {
            int tx = (int)round(target_xs[i]);
            int ty = (int)round(target_ys[i]);
            target_indices[i] = (tx >= 0 && tx < width && ty >= 0 && ty < height)
                                    ? (ty * width + tx)
                                    : -1;
        }

        int ddx_arr[] = {-1, 1, 0, 0, -1, -1, 1, 1};
        int ddy_arr[] = { 0, 0, -1, 1, -1,  1, -1, 1};
        float step_dist[] = {1.0f, 1.0f, 1.0f, 1.0f,
                              1.41421356f, 1.41421356f, 1.41421356f, 1.41421356f};

        while (!pq.empty()) {
            Label curr = pq.top();
            pq.pop();

            if (curr.cost > min_cost[curr.index] + 1e-4f) {
                continue;
            }

            int cx = curr.index % width;
            int cy = curr.index / width;

            int   par_x    = curr.parent_x;
            int   par_y    = curr.parent_y;
            float par_dist = curr.parent_dist;

            for (int dir = 0; dir < 8; ++dir) {
                int nx = cx + ddx_arr[dir];
                int ny = cy + ddy_arr[dir];
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                    continue;
                }

                int nidx = ny * width + nx;
                bool next_is_wall = grid[nidx] > 0;
                bool curr_is_wall = grid[curr.index] > 0;

                int nwalls = curr.walls;
                if (next_is_wall && !curr_is_wall) {
                    nwalls += 1;
                }
                if (nwalls >= MAX_WALLS) {
                    continue;
                }

                float ndist;
                int   new_par_x, new_par_y;
                float new_par_dist;
                float next_tl_contrib;

                if (line_of_sight(grid, pixel_tl, width, height, par_x, par_y, nx, ny)) {
                    float fdx = (float)(nx - par_x) * resolution;
                    float fdy = (float)(ny - par_y) * resolution;
                    ndist         = par_dist + sqrtf(fdx * fdx + fdy * fdy);
                    new_par_x     = par_x;
                    new_par_y     = par_y;
                    new_par_dist  = par_dist;
                    next_tl_contrib = 0.0f;
                } else {
                    ndist         = curr.dist + step_dist[dir] * resolution;
                    new_par_x     = cx;
                    new_par_y     = cy;
                    new_par_dist  = curr.dist;
                    float next_tl_val = pixel_tl_of(nidx);
                    float curr_tl_val = pixel_tl_of(curr.index);
                    next_tl_contrib   = (curr_tl_val <= 0.0f && next_tl_val > 0.0f) ? next_tl_val : 0.0f;
                }

                bool dominated = false;
                for (int w = 0; w <= nwalls; ++w) {
                    if (min_dist[nidx][w] <= ndist) {
                        dominated = true;
                        break;
                    }
                }
                if (dominated) {
                    continue;
                }
                min_dist[nidx][nwalls] = ndist;

                float new_tl = curr.tl + next_tl_contrib;
                float ncost  = 20.0f * log10f(fmaxf(ndist, mic_distance)) + new_tl;
                if (ncost < min_cost[nidx]) {
                    min_cost[nidx] = ncost;
                }
                pq.push({ndist, nwalls, nidx, new_tl, ncost,
                         new_par_x, new_par_y, new_par_dist});
            }
        }

        for (int i = 0; i < num_targets; ++i) {
            int tidx = target_indices[i];
            if (tidx != -1 && min_cost[tidx] != numeric_limits<float>::infinity()) {
                out_attenuations[i] = min_cost[tidx];
            }
        }
    }
}
