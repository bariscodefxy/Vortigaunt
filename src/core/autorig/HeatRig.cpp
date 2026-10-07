#include "HeatRig.h"

#include "MeshBvh.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <unordered_map>

namespace heatrig {
namespace {

// Vector maths uses assimp's aiVector3D, like the rest of the rigging code.
// Mind its operators: a * b is the DOT product and a ^ b the CROSS product,
// only a * scalar scales.

struct BoneTable {
    std::vector<int>              boneId;        // slot -> bone index as handed in
    std::vector<aiVector3D>             head;          // slot -> bone world position
    std::vector<std::vector<int>> segments;      // slot -> indices into the input segments
    std::vector<int>              parentSlot;    // slot -> slot of the bone it hangs off, or -1
};

struct SparseMatrix {
    int n = 0;
    std::vector<int>    rowStart;   // n + 1 entries
    std::vector<int>    colIndex;
    std::vector<double> values;
    std::vector<double> diagonal;
};

void Multiply(const SparseMatrix& m, const std::vector<double>& x, std::vector<double>& out) {
    out.assign(m.n, 0.0);
    for (int i = 0; i < m.n; i++) {
        double sum = 0.0;
        for (int k = m.rowStart[i]; k < m.rowStart[i + 1]; k++) {
            sum += m.values[k] * x[m.colIndex[k]];
        }
        out[i] = sum;
    }
}

bool SolveCG(const SparseMatrix& m, const std::vector<double>& rhs,
             std::vector<double>& x, int maxIterations, double tolerance) {
    int n = m.n;
    x.assign(n, 0.0);

    std::vector<double> residual = rhs;
    std::vector<double> preconditioned(n);
    std::vector<double> direction(n);
    std::vector<double> temp(n);

    double rhsNorm = 0.0;
    for (double v : rhs) rhsNorm += v * v;
    if (rhsNorm < 1e-30) {
        return true;  // zero right hand side, zero solution
    }

    for (int i = 0; i < n; i++) {
        double d = m.diagonal[i];
        preconditioned[i] = (std::abs(d) > 1e-20) ? residual[i] / d : residual[i];
    }
    direction = preconditioned;

    double rz = 0.0;
    for (int i = 0; i < n; i++) rz += residual[i] * preconditioned[i];

    double threshold = tolerance * tolerance * rhsNorm;

    for (int iteration = 0; iteration < maxIterations; iteration++) {
        Multiply(m, direction, temp);

        double dAd = 0.0;
        for (int i = 0; i < n; i++) dAd += direction[i] * temp[i];
        if (std::abs(dAd) < 1e-30) {
            break;
        }

        double alpha = rz / dAd;
        double residualNorm = 0.0;
        for (int i = 0; i < n; i++) {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * temp[i];
            residualNorm += residual[i] * residual[i];
        }
        if (residualNorm < threshold) {
            return true;
        }

        for (int i = 0; i < n; i++) {
            double d = m.diagonal[i];
            preconditioned[i] = (std::abs(d) > 1e-20) ? residual[i] / d : residual[i];
        }

        double rzNext = 0.0;
        for (int i = 0; i < n; i++) rzNext += residual[i] * preconditioned[i];
        if (std::abs(rz) < 1e-30) {
            break;
        }
        double beta = rzNext / rz;
        rz = rzNext;

        for (int i = 0; i < n; i++) {
            direction[i] = preconditioned[i] + beta * direction[i];
        }
    }

    return true;  // an unconverged solve is still a usable approximation here
}

// Cotangent of the angle at `corner` in the triangle (corner, a, b).
double CotangentAt(const aiVector3D& corner, const aiVector3D& a, const aiVector3D& b) {
    aiVector3D u = a - corner;
    aiVector3D v = b - corner;
    double cross = static_cast<double>((u ^ v).Length());
    if (cross < 1e-12) {
        return 0.0;
    }
    return static_cast<double>(u * v) / cross;
}

void BuildLaplacian(const autorig::WeldedMesh& mesh, std::vector<std::unordered_map<int, double>>& edgeWeights, std::vector<double>& vertexAreas) {
    int n = static_cast<int>(mesh.positions.size());
    edgeWeights.assign(n, {});
    vertexAreas.assign(n, 0.0);

    size_t triCount = mesh.triangles.size() / 3;
    for (size_t t = 0; t < triCount; t++) {
        int i0 = mesh.triangles[t * 3 + 0];
        int i1 = mesh.triangles[t * 3 + 1];
        int i2 = mesh.triangles[t * 3 + 2];
        const aiVector3D& p0 = mesh.positions[i0];
        const aiVector3D& p1 = mesh.positions[i1];
        const aiVector3D& p2 = mesh.positions[i2];

        double area = 0.5 * static_cast<double>(((p1 - p0) ^ (p2 - p0)).Length());
        if (area < 1e-12) {
            continue;
        }
        double share = area / 3.0;
        vertexAreas[i0] += share;
        vertexAreas[i1] += share;
        vertexAreas[i2] += share;

        // The cotangent at each corner weights the edge opposite to it.
        const int   edgeA[3]  = {i1, i2, i0};
        const int   edgeB[3]  = {i2, i0, i1};
        const aiVector3D* pos[3]    = {&p0, &p1, &p2};
        const aiVector3D* posA[3]   = {&p1, &p2, &p0};
        const aiVector3D* posB[3]   = {&p2, &p0, &p1};

        for (int k = 0; k < 3; k++) {
            double cot = CotangentAt(*pos[k], *posA[k], *posB[k]);
            double w = 0.5 * std::max(0.0, cot);
            if (w <= 0.0) {
                continue;
            }
            edgeWeights[edgeA[k]][edgeB[k]] += w;
            edgeWeights[edgeB[k]][edgeA[k]] += w;
        }
    }

    for (double& a : vertexAreas) {
        a = std::max(a, 1e-8);
    }
}


BoneTable BuildBoneTable(const std::vector<BoneSegment>& segments) {
    BoneTable table;
    std::unordered_map<int, int> idToSlot;

    for (size_t i = 0; i < segments.size(); i++) {
        const BoneSegment& s = segments[i];
        auto it = idToSlot.find(s.boneIndex);
        int slot;
        if (it == idToSlot.end()) {
            slot = static_cast<int>(table.boneId.size());
            idToSlot[s.boneIndex] = slot;
            table.boneId.push_back(s.boneIndex);
            table.head.push_back(s.start);
            table.segments.push_back({});
        } else {
            slot = it->second;
        }
        table.segments[slot].push_back(static_cast<int>(i));
    }

    // A bone is the child of whichever bone has a segment ending on its head.
    table.parentSlot.assign(table.boneId.size(), -1);
    const float jointTolerance = 0.01f;
    for (size_t parent = 0; parent < table.boneId.size(); parent++) {
        for (int segIndex : table.segments[parent]) {
            const aiVector3D& tip = segments[segIndex].end;
            for (size_t child = 0; child < table.boneId.size(); child++) {
                if (child == parent) continue;
                if ((table.head[child] - tip).Length() <= jointTolerance) {
                    table.parentSlot[child] = static_cast<int>(parent);
                }
            }
        }
    }
    return table;
}


float SourceDistance(const aiVector3D& position, const aiVector3D& normal, const BoneTable& table, int slot, const std::vector<BoneSegment>& segments, aiVector3D& outClosest) {

    float best = std::numeric_limits<float>::max();
    outClosest = position;

    for (int segIndex : table.segments[slot]) {
        const BoneSegment& s = segments[segIndex];
        aiVector3D closest = autorig::ClosestPointOnSegment(position, s.start, s.end);
        aiVector3D delta = position - closest;
        float dist = delta.Length();

        float scaled = dist;
        if (dist > 1e-6f) {
            float cosine = (delta * (1.0f / dist)) * normal;
            scaled = dist / (0.5f * (cosine + 1.001f));
        }
        if (scaled < best) {
            best = scaled;
            outClosest = closest;
        }
    }
    return best;
}

} // namespace


Result Solve(const std::vector<float>& positions, const std::vector<BoneSegment>& segments, const Options& options) {
    Result result;

    size_t inputCount = positions.size() / 3;
    if (inputCount == 0) {
        result.error = "No vertices to rig.";
        return result;
    }
    if (segments.empty()) {
        result.error = "No bone segments supplied.";
        return result;
    }

    autorig::WeldedMesh mesh = autorig::BuildWeldedMesh(positions, options.weldEpsilon);

    int n = static_cast<int>(mesh.positions.size());
    result.weldedVertexCount = n;
    result.triangleCount = static_cast<int>(mesh.triangles.size() / 3);

    BoneTable table = BuildBoneTable(segments);
    int slotCount = static_cast<int>(table.boneId.size());

    autorig::TriangleBvh bvh;
    if (options.useVisibility) {
        bvh.Build(mesh);
    }

    std::vector<std::vector<int>> closestSlots(n);
    std::vector<double> heat(n, 0.0);

    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(slotCount);
    std::vector<aiVector3D> closestPoints(slotCount);

    std::vector<int>   fallbackSlot(n, 0);
    std::vector<float> fallbackDistance(n, 1.0f);

    const float kTieBand = 1.05f;       // sources this close to the nearest share the vertex
    const float kHiddenPenalty = 0.05f; // strength for vertices no bone can see

    for (int v = 0; v < n; v++) {
        const aiVector3D& position = mesh.positions[v];
        const aiVector3D& normal = mesh.normals[v];

        ranked.clear();
        for (int slot = 0; slot < slotCount; slot++) {
            ranked.emplace_back(
                SourceDistance(position, normal, table, slot, segments, closestPoints[slot]), slot);
        }
        std::sort(ranked.begin(), ranked.end());

        float acceptedDistance = -1.0f;
        for (const auto& entry : ranked) {
            if (acceptedDistance > 0.0f && entry.first > acceptedDistance * kTieBand) {
                break;
            }
            if (options.useVisibility && bvh.IsOccluded(position, closestPoints[entry.second], v)) {
                continue;
            }
            if (acceptedDistance < 0.0f) {
                acceptedDistance = std::max(entry.first, 1e-2f);
            }
            closestSlots[v].push_back(entry.second);
        }

        fallbackSlot[v] = ranked.front().second;
        fallbackDistance[v] = std::max(ranked.front().first, 1e-2f);

        if (closestSlots[v].empty()) {
            result.unreachedVertexCount++;
            heat[v] = 0.0;
            continue;
        }

        int count = static_cast<int>(closestSlots[v].size());
        heat[v] = static_cast<double>(count) * static_cast<double>(options.heatStrength) /
                  (static_cast<double>(acceptedDistance) * static_cast<double>(acceptedDistance));
    }

    
    std::vector<int> component(n, -1);
    std::vector<int> stack;
    int componentCount = 0;
    for (int start = 0; start < n; start++) {
        if (component[start] >= 0) {
            continue;
        }
        stack.clear();
        stack.push_back(start);
        component[start] = componentCount;

        double total = 0.0;
        std::vector<int> members;
        while (!stack.empty()) {
            int v = stack.back();
            stack.pop_back();
            members.push_back(v);
            total += heat[v];
            for (int neighbor : mesh.neighbors[v]) {
                if (component[neighbor] < 0) {
                    component[neighbor] = componentCount;
                    stack.push_back(neighbor);
                }
            }
        }
        componentCount++;

        if (total > 0.0) {
            continue;
        }
        for (int v : members) {
            closestSlots[v].push_back(fallbackSlot[v]);
            double d = static_cast<double>(fallbackDistance[v]);
            heat[v] = static_cast<double>(options.heatStrength) * static_cast<double>(kHiddenPenalty) / (d * d);
        }
    }
    

    std::vector<std::unordered_map<int, double>> edgeWeights;
    std::vector<double> vertexAreas;
    BuildLaplacian(mesh, edgeWeights, vertexAreas);

    SparseMatrix matrix;
    matrix.n = n;
    matrix.rowStart.assign(n + 1, 0);
    matrix.diagonal.assign(n, 0.0);

    for (int i = 0; i < n; i++) {
        matrix.rowStart[i + 1] = matrix.rowStart[i] + static_cast<int>(edgeWeights[i].size()) + 1;
    }
    matrix.colIndex.resize(matrix.rowStart[n]);
    matrix.values.resize(matrix.rowStart[n]);

    for (int i = 0; i < n; i++) {
        double diagonal = vertexAreas[i] * heat[i];
        int cursor = matrix.rowStart[i];
        for (const auto& entry : edgeWeights[i]) {
            matrix.colIndex[cursor] = entry.first;
            matrix.values[cursor] = -entry.second;
            diagonal += entry.second;
            cursor++;
        }
        matrix.colIndex[cursor] = i;
        matrix.values[cursor] = diagonal;
        matrix.diagonal[i] = diagonal;
    }

    std::vector<float> bestWeight(n, -1.0f), secondWeight(n, -1.0f);
    std::vector<int>   bestSlot(n, -1), secondSlot(n, -1);

    std::vector<bool> slotActive(slotCount, false);
    for (int v = 0; v < n; v++) {
        for (int slot : closestSlots[v]) {
            slotActive[slot] = true;
        }
    }

    std::vector<double> rhs(n), weights;
    for (int slot = 0; slot < slotCount; slot++) {
        if (!slotActive[slot]) {
            continue;  // no source anywhere, the solution would be all zeroes
        }
        result.solvedBoneCount++;

        for (int v = 0; v < n; v++) {
            const auto& list = closestSlots[v];
            bool isSource = std::find(list.begin(), list.end(), slot) != list.end();
            rhs[v] = isSource ? vertexAreas[v] * heat[v] / static_cast<double>(list.size()) : 0.0;
        }

        SolveCG(matrix, rhs, weights, options.maxIterations, static_cast<double>(options.tolerance));

        for (int v = 0; v < n; v++) {
            float w = static_cast<float>(weights[v]);
            if (w > bestWeight[v]) {
                secondWeight[v] = bestWeight[v];
                secondSlot[v] = bestSlot[v];
                bestWeight[v] = w;
                bestSlot[v] = slot;
            } else if (w > secondWeight[v]) {
                secondWeight[v] = w;
                secondSlot[v] = slot;
            }
        }
    }

    std::vector<int> assigned(n);
    std::vector<float> margin(n);
    for (int v = 0; v < n; v++) {
        assigned[v] = (bestSlot[v] >= 0) ? bestSlot[v] : 0;
        margin[v] = std::max(0.0f, bestWeight[v] - std::max(0.0f, secondWeight[v]));
    }

    for (int pass = 0; pass < options.cleanupPasses; pass++) {
        bool changed = false;
        std::vector<int> next = assigned;
        for (int v = 0; v < n; v++) {
            if (margin[v] >= options.undecidedMargin || mesh.neighbors[v].empty()) {
                continue;
            }
            std::unordered_map<int, int> votes;
            votes[assigned[v]] += 1;
            for (int neighbor : mesh.neighbors[v]) {
                votes[assigned[neighbor]] += 1;
            }
            int winner = assigned[v];
            int winnerVotes = 0;
            for (const auto& entry : votes) {
                if (entry.second > winnerVotes) {
                    winnerVotes = entry.second;
                    winner = entry.first;
                }
            }
            if (winner != assigned[v]) {
                next[v] = winner;
                changed = true;
            }
        }
        assigned.swap(next);
        if (!changed) {
            break;
        }
    }


    if (options.minRegionSize > 0) {
        for (int sweep = 0; sweep < 2; sweep++) {
            std::vector<int> regionOf(n, -1);
            std::vector<std::vector<int>> regions;
            std::vector<int> pending;

            for (int start = 0; start < n; start++) {
                if (regionOf[start] >= 0) {
                    continue;
                }
                int id = static_cast<int>(regions.size());
                regions.push_back({});
                pending.clear();
                pending.push_back(start);
                regionOf[start] = id;
                while (!pending.empty()) {
                    int v = pending.back();
                    pending.pop_back();
                    regions[id].push_back(v);
                    for (int neighbor : mesh.neighbors[v]) {
                        if (regionOf[neighbor] < 0 && assigned[neighbor] == assigned[v]) {
                            regionOf[neighbor] = id;
                            pending.push_back(neighbor);
                        }
                    }
                }
            }

            bool changed = false;
            for (const std::vector<int>& region : regions) {
                if (static_cast<int>(region.size()) >= options.minRegionSize) {
                    continue;
                }
                float marginSum = 0.0f;
                for (int v : region) {
                    marginSum += margin[v];
                }
                if (marginSum / static_cast<float>(region.size()) >= options.undecidedMargin) {
					continue;  // the solver was sure, this is a real small bone like a finger bone, leave it alone
                }

                std::unordered_map<int, int> surrounding;
                for (int v : region) {
                    for (int neighbor : mesh.neighbors[v]) {
                        if (regionOf[neighbor] != regionOf[v]) {
                            surrounding[assigned[neighbor]]++;
                        }
                    }
                }
                if (surrounding.empty()) {
                    continue;  // a free floating shell, nothing to absorb it
                }

                int winner = -1;
                int winnerVotes = 0;
                for (const auto& entry : surrounding) {
                    if (entry.second > winnerVotes) {
                        winnerVotes = entry.second;
                        winner = entry.first;
                    }
                }
                for (int v : region) {
                    assigned[v] = winner;
                }
                result.absorbedRegionCount++;
                changed = true;
            }
            if (!changed) {
                break;
            }
        }
    }



    if (options.pivotSnap > 0.0f) {
        float band = options.undecidedMargin * options.pivotSnap;
        for (int v = 0; v < n; v++) {
            int a = assigned[v];
            int b = secondSlot[v];
            if (b < 0 || a == b || margin[v] >= band) {
                continue;
            }
            int parent = -1, child = -1;
            if (table.parentSlot[b] == a) {
                parent = a; child = b;
            } else if (table.parentSlot[a] == b) {
                parent = b; child = a;
            } else {
                continue;  // not a shared joint, leave the answer from the solver alone
            }

            aiVector3D axis = table.head[child] - table.head[parent];
            axis.NormalizeSafe();
            if (axis.SquareLength() < 0.5f) {
                continue;  // zero length bone, no meaningful cutting plane
            }
            assigned[v] = (((mesh.positions[v] - table.head[child]) * axis) >= 0.0f) ? child : parent;
        }
    }

    result.boneIndices.resize(inputCount);
    result.confidence.resize(inputCount);
    for (size_t i = 0; i < inputCount; i++) {
        int welded = mesh.originalToWelded[i];
        result.boneIndices[i] = table.boneId[assigned[welded]];
        result.confidence[i] = margin[welded];
    }

    result.ok = true;
    return result;
}

} // namespace heatrig
