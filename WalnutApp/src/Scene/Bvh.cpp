#include "Bvh.h"

#include <algorithm>
#include <array>

namespace RayScene
{
	namespace
	{
		// Testing every position between two primitives would make the build
		// quadratic. Sorting them into a fixed number of slices along the axis and
		// only cutting between slices keeps it linear per axis, and the candidate
		// it finds is close enough to the best one that the difference does not
		// show.
		constexpr uint32_t SahBinCount = 12;

		// The cost of descending into one interior node, measured in primitive
		// tests. It only decides how eagerly the build stops splitting, because it
		// is the one term that does not shrink when the boxes get tighter.
		constexpr float TraversalCost = 0.125f;

		struct SahBin
		{
			BvhBounds Bounds;
			uint32_t Count = 0;
		};

		// Which slice a centroid falls into. The clamp catches the centroid
		// sitting exactly on the upper bound.
		uint32_t SahBinIndex(float centroid, float axisMin, float binScale)
		{
			const float slice = (centroid - axisMin) * binScale;
			if (slice <= 0.0f)
				return 0;

			return std::min(SahBinCount - 1u, static_cast<uint32_t>(slice));
		}

		struct BuildEntry
		{
			uint32_t Start;
			uint32_t Count;
			uint32_t NodeIndex;
			uint32_t Depth;
		};

		// NoCandidate is not a failure: it means every centroid fell on the same
		// plane, so no cut can separate them and the median split has to take over.
		enum class SahResult
		{
			Split,
			Leaf,
			NoCandidate
		};

		class Builder
		{
		public:
			Builder(
				const std::vector<BvhPrimitive>& primitives,
				const BvhBuildSettings& settings,
				std::vector<uint32_t>& order)
				: m_Primitives(primitives), m_Settings(settings), m_Order(order)
			{
			}

			// Searches for the cheapest place to cut one range in two.
			//
			// A ray that has already entered this node's box enters a child box
			// with a probability equal to the ratio of their surface areas, and
			// once inside it pays one test per primitive the child holds. So the
			// expected cost of a cut is one descent plus each side's count
			// weighted by its share of the parent area.
			SahResult FindSahSplit(
				const BuildEntry& entry,
				const BvhBounds& centroidBounds,
				float parentSurfaceArea,
				uint32_t& splitAxis,
				uint32_t& leftCount)
			{
				if (parentSurfaceArea <= 0.0f)
					return SahResult::NoCandidate;

				const glm::vec3 centroidExtent =
					centroidBounds.Max - centroidBounds.Min;
				float bestCost = std::numeric_limits<float>::max();
				uint32_t bestAxis = 3;
				uint32_t bestCut = 0;

				for (uint32_t axis = 0; axis < 3; axis++)
				{
					// Every centroid shares this coordinate, so no cut along it
					// separates anything and the bin scale would divide by zero.
					if (centroidExtent[axis] <= 0.0f)
						continue;

					const float axisMin = centroidBounds.Min[axis];
					const float binScale =
						static_cast<float>(SahBinCount) / centroidExtent[axis];

					std::array<SahBin, SahBinCount> bins{};
					for (uint32_t offset = 0; offset < entry.Count; offset++)
					{
						const BvhPrimitive& primitive =
							m_Primitives[m_Order[entry.Start + offset]];
						SahBin& bin = bins[SahBinIndex(
							primitive.Centroid[axis], axisMin, binScale)];
						bin.Count++;
						bin.Bounds.Grow(primitive.Bounds);
					}

					// One sweep from each side records the box and count on either
					// side of every cut, keeping the search linear in bins.
					std::array<float, SahBinCount - 1> leftAreas{};
					std::array<uint32_t, SahBinCount - 1> leftCounts{};
					BvhBounds leftBounds;
					uint32_t leftSoFar = 0;
					for (uint32_t binIndex = 0; binIndex + 1 < SahBinCount; binIndex++)
					{
						leftBounds.Grow(bins[binIndex].Bounds);
						leftSoFar += bins[binIndex].Count;
						leftAreas[binIndex] = leftBounds.SurfaceArea();
						leftCounts[binIndex] = leftSoFar;
					}

					BvhBounds rightBounds;
					uint32_t rightSoFar = 0;
					for (uint32_t binIndex = SahBinCount - 1; binIndex > 0; binIndex--)
					{
						rightBounds.Grow(bins[binIndex].Bounds);
						rightSoFar += bins[binIndex].Count;

						const uint32_t cut = binIndex - 1;
						// An empty side is not a split at all, and scoring it would
						// let a cut that changes nothing look free.
						if (leftCounts[cut] == 0 || rightSoFar == 0)
							continue;

						const float cost =
							TraversalCost +
							(leftAreas[cut] * static_cast<float>(leftCounts[cut]) +
								rightBounds.SurfaceArea() * static_cast<float>(rightSoFar)) /
								parentSurfaceArea;

						if (cost < bestCost)
						{
							bestCost = cost;
							bestAxis = axis;
							bestCut = cut;
						}
					}
				}

				if (bestAxis > 2)
					return SahResult::NoCandidate;

				// Keeping the range whole costs one test per primitive. A split only
				// earns its extra node when it beats that, unless the range is past
				// the leaf limit.
				if (bestCost >= static_cast<float>(entry.Count) &&
					entry.Count <= m_Settings.MaxLeafSize)
				{
					return SahResult::Leaf;
				}

				const float axisMin = centroidBounds.Min[bestAxis];
				const float binScale =
					static_cast<float>(SahBinCount) / centroidExtent[bestAxis];
				const auto rangeBegin = m_Order.begin() + entry.Start;
				const auto splitPoint = std::partition(
					rangeBegin,
					rangeBegin + entry.Count,
					[this, bestAxis, bestCut, axisMin, binScale](uint32_t index)
					{
						return SahBinIndex(
							m_Primitives[index].Centroid[bestAxis],
							axisMin,
							binScale) <= bestCut;
					});

				leftCount = static_cast<uint32_t>(splitPoint - rangeBegin);
				// The bin counts already proved both sides hold something; this only
				// guards against the two disagreeing and building an empty child.
				if (leftCount == 0 || leftCount >= entry.Count)
					return SahResult::NoCandidate;

				splitAxis = bestAxis;
				return SahResult::Split;
			}

			// Orders the range by centroid along its widest axis and cuts the count
			// in half. Splitting by index rather than by a plane always succeeds and
			// stays balanced, even when every centroid coincides.
			void MedianSplit(
				const BuildEntry& entry,
				const BvhBounds& centroidBounds,
				uint32_t& splitAxis,
				uint32_t& leftCount)
			{
				const glm::vec3 extent = centroidBounds.Max - centroidBounds.Min;
				uint32_t widestAxis = 0;
				if (extent.y > extent[widestAxis])
					widestAxis = 1;
				if (extent.z > extent[widestAxis])
					widestAxis = 2;

				const auto rangeBegin = m_Order.begin() + entry.Start;
				std::nth_element(
					rangeBegin,
					rangeBegin + entry.Count / 2,
					rangeBegin + entry.Count,
					[this, widestAxis](uint32_t left, uint32_t right)
					{
						return m_Primitives[left].Centroid[widestAxis] <
							m_Primitives[right].Centroid[widestAxis];
					});

				splitAxis = widestAxis;
				leftCount = entry.Count / 2;
			}

		private:
			const std::vector<BvhPrimitive>& m_Primitives;
			const BvhBuildSettings& m_Settings;
			std::vector<uint32_t>& m_Order;
		};
	}

	float BvhBounds::SurfaceArea() const
	{
		if (IsEmpty())
			return 0.0f;

		const glm::vec3 extent = Max - Min;
		return 2.0f * (
			extent.x * extent.y +
			extent.y * extent.z +
			extent.z * extent.x);
	}

	BvhBuildResult BuildBvh(
		const std::vector<BvhPrimitive>& primitives,
		const BvhBuildSettings& settings)
	{
		BvhBuildResult result;
		const uint32_t primitiveCount = static_cast<uint32_t>(primitives.size());
		result.PrimitiveOrder.resize(primitiveCount);
		for (uint32_t index = 0; index < primitiveCount; index++)
			result.PrimitiveOrder[index] = index;

		if (primitiveCount == 0)
			return result;

		// Every split adds two nodes, and a binary tree over N primitives never
		// needs more than 2N-1 of them, so reserving that keeps references to
		// nodes stable while children are appended.
		result.Nodes.reserve(2 * static_cast<size_t>(primitiveCount));
		result.Nodes.push_back({});

		Builder builder(primitives, settings, result.PrimitiveOrder);
		const glm::vec3 padding(settings.BoundsPadding);
		float costSum = 0.0f;
		float rootSurfaceArea = 0.0f;

		std::vector<BuildEntry> pending;
		pending.push_back({ 0, primitiveCount, 0, 1 });

		while (!pending.empty())
		{
			const BuildEntry entry = pending.back();
			pending.pop_back();
			result.Depth = std::max(result.Depth, entry.Depth);

			BvhBounds nodeBounds;
			BvhBounds centroidBounds;
			for (uint32_t offset = 0; offset < entry.Count; offset++)
			{
				const BvhPrimitive& primitive =
					primitives[result.PrimitiveOrder[entry.Start + offset]];
				nodeBounds.Grow(primitive.Bounds);
				centroidBounds.Grow(primitive.Centroid);
			}

			const float nodeSurfaceArea = nodeBounds.SurfaceArea();
			// The root is the first entry popped, so its area is known before any
			// child needs to be weighed against it.
			if (entry.NodeIndex == 0)
				rootSurfaceArea = nodeSurfaceArea;

			const bool canSplit =
				entry.Count > settings.MinLeafSize &&
				entry.Depth < settings.MaxDepth;

			uint32_t splitAxis = 0;
			uint32_t leftCount = 0;
			bool splitRange = false;
			if (canSplit && settings.UseSah)
			{
				const SahResult sah = builder.FindSahSplit(
					entry, centroidBounds, nodeSurfaceArea, splitAxis, leftCount);
				splitRange = sah == SahResult::Split;

				// No plane separates these centroids, so a leaf is the honest
				// answer, but past the leaf limit the median split keeps the leaf
				// from growing without end.
				if (sah == SahResult::NoCandidate &&
					entry.Count > settings.MaxLeafSize)
				{
					builder.MedianSplit(entry, centroidBounds, splitAxis, leftCount);
					splitRange = true;
				}
			}
			else if (canSplit)
			{
				builder.MedianSplit(entry, centroidBounds, splitAxis, leftCount);
				splitRange = true;
			}

			BvhNode& node = result.Nodes[entry.NodeIndex];
			node.BoundsMin = glm::vec4(nodeBounds.Min - padding, 0.0f);
			node.BoundsMax = glm::vec4(nodeBounds.Max + padding, 0.0f);

			if (!splitRange)
			{
				node.Links = glm::uvec4(entry.Start, 0, entry.Count, 0);
				costSum += nodeSurfaceArea * static_cast<float>(entry.Count);
				continue;
			}

			const uint32_t leftChildIndex =
				static_cast<uint32_t>(result.Nodes.size());
			const uint32_t rightChildIndex = leftChildIndex + 1;
			// A count of zero marks an interior node; the split axis lets the
			// shader tell which child lies on the near side of the ray.
			node.Links = glm::uvec4(leftChildIndex, rightChildIndex, 0, splitAxis);
			costSum += nodeSurfaceArea * TraversalCost;
			result.Nodes.push_back({});
			result.Nodes.push_back({});

			pending.push_back({
				entry.Start, leftCount, leftChildIndex, entry.Depth + 1 });
			pending.push_back({
				entry.Start + leftCount,
				entry.Count - leftCount,
				rightChildIndex,
				entry.Depth + 1 });
		}

		result.Cost = rootSurfaceArea > 0.0f
			? costSum / rootSurfaceArea
			: static_cast<float>(primitiveCount);
		return result;
	}
}
