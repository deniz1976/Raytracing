#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <glm/glm.hpp>

namespace RayScene
{
	// An axis aligned box grown one point or one box at a time.
	struct BvhBounds
	{
		glm::vec3 Min{ std::numeric_limits<float>::max() };
		glm::vec3 Max{ std::numeric_limits<float>::lowest() };

		void Grow(const glm::vec3& point)
		{
			Min = glm::min(Min, point);
			Max = glm::max(Max, point);
		}

		void Grow(const BvhBounds& other)
		{
			Min = glm::min(Min, other.Min);
			Max = glm::max(Max, other.Max);
		}

		bool IsEmpty() const { return Min.x > Max.x || Min.y > Max.y || Min.z > Max.z; }

		// The chance that a random ray crossing the parent box also crosses this
		// one is the ratio of their surface areas, which is why surface area and
		// not volume is what the split heuristic weighs the primitive counts with.
		float SurfaceArea() const;
	};

	// What the builder needs to know about one sphere or triangle.
	struct BvhPrimitive
	{
		BvhBounds Bounds;
		glm::vec3 Centroid;
	};

	// One node exactly as the shader reads it. An interior node stores the two
	// child indices, a count of zero and its split axis; a leaf stores the first
	// primitive in the reordered primitive buffer plus how many belong to it.
	struct alignas(16) BvhNode
	{
		glm::vec4 BoundsMin;
		glm::vec4 BoundsMax;
		glm::uvec4 Links;
	};

	static_assert(sizeof(BvhNode) == 48);

	struct BvhBuildSettings
	{
		// The surface area heuristic splits where the two child boxes are cheapest
		// to trace; the median split cuts the count in half along the widest axis
		// and is kept as the baseline the heuristic has to beat.
		bool UseSah = true;
		// Ranges this small always become leaves.
		uint32_t MinLeafSize = 2;
		// Past this size a leaf is split even when the heuristic would keep it,
		// because every ray entering a long leaf pays for all of it.
		uint32_t MaxLeafSize = 4;
		// The shader traverses with a fixed size stack holding one pending sibling
		// per level, so bounding the depth is what keeps it from losing geometry.
		uint32_t MaxDepth = 30;
		// Grows every node box so perfectly flat geometry never produces a
		// zero-thickness box that the slab test could miss on a float edge.
		float BoundsPadding = 0.0f;
	};

	struct BvhBuildResult
	{
		std::vector<BvhNode> Nodes;
		// Primitives are uploaded in this order, so a leaf can point at a
		// contiguous range of the GPU buffer instead of an indirection list.
		std::vector<uint32_t> PrimitiveOrder;
		uint32_t Depth = 0;
		// The expected number of primitive tests one random ray entering the root
		// costs, under the same heuristic the split search minimises.
		float Cost = 0.0f;
	};

	// Two invariants the shader depends on: the left child always holds the
	// primitives with the smaller centroid along the node's split axis, and no
	// node is deeper than the traversal stack can follow. A tree over N
	// primitives never needs more than 2N-1 nodes.
	BvhBuildResult BuildBvh(
		const std::vector<BvhPrimitive>& primitives,
		const BvhBuildSettings& settings);
}
