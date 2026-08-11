#include "ComputeRenderer.h"

#include "Walnut/Application.h"
#include "Walnut/Image.h"

#include "backends/imgui_impl_vulkan.h"

#include "../../vendor/stb_image/stb_image.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <unordered_map>
#include <vector>

namespace
{
	// SceneSettings.x is 1 when the BVH should be traversed, .y is the number of
	// lights in the light buffer, .z is 1 when each hit samples a single light
	// chosen by weight instead of all of them and .w is the bounce limit.
	struct alignas(16) PushConstants
	{
		glm::vec4 CameraPosition;
		glm::vec4 CameraForward;
		uint32_t FrameIndex;
		float VerticalFov;
		float Exposure;
		uint32_t SphereCount;
		glm::uvec4 SceneSettings;
		uint32_t TriangleCount;
		uint32_t Padding0;
		uint32_t Padding1;
		uint32_t Padding2;
	};

	// The triangle count grows this from 64 to 80 bytes, still leaving 48 bytes
	// inside the 128-byte push constant capacity Vulkan guarantees.
	static_assert(sizeof(PushConstants) == 80);

	struct alignas(16) GpuSphere
	{
		glm::vec4 CenterRadius;
		glm::vec4 AlbedoReflectivity;
		// x stores roughness, y stores MaterialType and z stores optical density.
		glm::vec4 RoughnessMaterial;
	};

	static_assert(sizeof(GpuSphere) == 48);

	struct alignas(16) GpuTriangle
	{
		glm::vec4 Vertex0;
		glm::vec4 Vertex1;
		glm::vec4 Vertex2;
		glm::vec4 AlbedoReflectivity;
		glm::vec4 RoughnessMaterial;
		// Normal0.w is one when all three OBJ corner normals are available.
		glm::vec4 Normal0;
		glm::vec4 Normal1;
		glm::vec4 Normal2;
		// TexCoord2.w is one when all three OBJ corners have UV coordinates.
		glm::vec4 TexCoord01;
		glm::vec4 TexCoord2;
	};

	static_assert(sizeof(GpuTriangle) == 160);

	// One node of the bounding volume hierarchy. An interior node stores the two
	// child indices and a SphereCount of zero; a leaf stores the first sphere in
	// the BVH ordered sphere buffer plus how many spheres belong to it.
	struct alignas(16) GpuBvhNode
	{
		glm::vec4 BoundsMin;
		glm::vec4 BoundsMax;
		glm::uvec4 Links;
	};

	static_assert(sizeof(GpuBvhNode) == 48);

	// The same three-vec4 packing the spheres use. Position and intensity share
	// one vec4 because std430 would pad a lone vec3 out to 16 bytes anyway.
	struct alignas(16) GpuSphereLight
	{
		glm::vec4 PositionIntensity;
		glm::vec4 Color;
		// x stores radius, z the selection CDF and w its probability.
		glm::vec4 RadiusSampling;
	};

	static_assert(sizeof(GpuSphereLight) == 48);

	constexpr uint32_t BvhLeafSphereCount = 2;

	// A leaf is allowed to hold more than the minimum when the heuristic says
	// splitting would not pay for itself, but not without limit: past this many
	// spheres a leaf costs more than the extra box test, so the split is forced.
	constexpr uint32_t BvhMaxLeafSphereCount = 4;

	// The shader traverses with a fixed size stack of BVH_STACK_SIZE entries and
	// silently drops children it cannot push. That stack holds at most one pending
	// sibling per level, so bounding the depth here is what keeps the traversal
	// from losing geometry. A median split is balanced and would never come close,
	// but the surface area heuristic follows the spheres instead of the count and
	// can build a deep, lopsided branch, so the bound has to be enforced.
	constexpr uint32_t BvhMaxDepth = 30;

	// Testing every position between two spheres would make the build quadratic.
	// Sorting the spheres into a fixed number of slices along the axis and only
	// cutting between slices keeps it linear per axis, and the candidate it finds
	// is close enough to the best one that the difference does not show.
	constexpr uint32_t BvhSahBinCount = 12;

	// The cost of descending into one interior node, measured in sphere tests.
	// It only decides how eagerly the build stops splitting, because it is the one
	// term that does not shrink when the boxes get tighter.
	constexpr float BvhTraversalCost = 0.125f;

	// An axis aligned box grown one point or one box at a time. Kept separate from
	// the node struct because the build needs a running box per bin, which never
	// reaches the GPU.
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

		bool IsEmpty() const { return Min.x > Max.x; }

		// The chance that a random ray crossing the parent box also crosses this
		// one is the ratio of their surface areas, which is why surface area and
		// not volume is what the split heuristic weighs the sphere counts with.
		float SurfaceArea() const
		{
			if (IsEmpty())
				return 0.0f;

			const glm::vec3 extent = Max - Min;
			return 2.0f * (
				extent.x * extent.y +
				extent.y * extent.z +
				extent.z * extent.x);
		}
	};

	// One slice of the split axis: how many spheres landed in it and the box that
	// encloses them.
	struct BvhSahBin
	{
		BvhBounds Bounds;
		uint32_t Count = 0;
	};

	// Which slice a centroid falls into. The multiply is hoisted out by the
	// caller because it is the same for every sphere in the range, and the clamp
	// catches the centroid sitting exactly on the upper bound.
	uint32_t SahBinIndex(float centroid, float axisMin, float binScale)
	{
		const float slice = (centroid - axisMin) * binScale;
		if (slice <= 0.0f)
			return 0;

		return std::min(
			BvhSahBinCount - 1u,
			static_cast<uint32_t>(slice));
	}

	// One pending subtree during the build: the slice of the sphere order array
	// it owns, the node that describes it and how deep it sits in the tree.
	struct BvhBuildEntry
	{
		uint32_t Start;
		uint32_t Count;
		uint32_t NodeIndex;
		uint32_t Depth;
	};

	ComputeRenderer::Sphere SanitizeSphere(
		const ComputeRenderer::Sphere& sphere)
	{
		ComputeRenderer::Sphere result = sphere;
		result.Radius = std::clamp(result.Radius, 0.05f, 200.0f);
		result.Albedo = glm::clamp(
			result.Albedo,
			glm::vec3(0.0f),
			glm::vec3(1.0f));
		result.Reflectivity =
			std::clamp(result.Reflectivity, 0.0f, 1.0f);
		result.Roughness =
			std::clamp(result.Roughness, 0.0f, 1.0f);
		result.IndexOfRefraction = std::isfinite(result.IndexOfRefraction)
			? std::clamp(result.IndexOfRefraction, 1.0f, 2.5f)
			: 1.5f;
		if (result.Type != ComputeRenderer::MaterialType::Legacy &&
			result.Type != ComputeRenderer::MaterialType::Diffuse &&
			result.Type != ComputeRenderer::MaterialType::Metal &&
			result.Type != ComputeRenderer::MaterialType::Dielectric)
		{
			result.Type = ComputeRenderer::MaterialType::Legacy;
		}
		return result;
	}

	ComputeRenderer::SphereLight SanitizeSphereLight(
		const ComputeRenderer::SphereLight& light)
	{
		ComputeRenderer::SphereLight result = light;
		result.Color = glm::clamp(
			result.Color,
			glm::vec3(0.0f),
			glm::vec3(1.0f));
		result.Radius = std::clamp(result.Radius, 0.05f, 20.0f);
		result.Intensity = std::clamp(result.Intensity, 0.0f, 100.0f);
		return result;
	}

	// The limits match the UI sliders, so a camera that came out of a scene file
	// is always a camera the controls can still represent. Clamping the pitch
	// away from straight up also keeps the forward vector from lining up with the
	// world up vector, which would collapse the camera basis.
	ComputeRenderer::Camera SanitizeCamera(
		const ComputeRenderer::Camera& camera)
	{
		ComputeRenderer::Camera result = camera;
		result.Pitch = std::clamp(result.Pitch, -89.0f, 89.0f);
		result.VerticalFov = std::clamp(result.VerticalFov, 20.0f, 90.0f);
		return result;
	}

	float SanitizeExposure(float exposure)
	{
		return std::clamp(
			exposure,
			ComputeRenderer::MinExposure,
			ComputeRenderer::MaxExposure);
	}

	bool IsFinite(const ComputeRenderer::Sphere& sphere)
	{
		return
			std::isfinite(sphere.Center.x) &&
			std::isfinite(sphere.Center.y) &&
			std::isfinite(sphere.Center.z) &&
			std::isfinite(sphere.Radius) &&
			std::isfinite(sphere.Albedo.r) &&
			std::isfinite(sphere.Albedo.g) &&
			std::isfinite(sphere.Albedo.b) &&
			std::isfinite(sphere.Reflectivity) &&
			std::isfinite(sphere.Roughness) &&
			std::isfinite(sphere.IndexOfRefraction);
	}

	bool IsFinite(const ComputeRenderer::SphereLight& light)
	{
		return
			std::isfinite(light.Position.x) &&
			std::isfinite(light.Position.y) &&
			std::isfinite(light.Position.z) &&
			std::isfinite(light.Color.r) &&
			std::isfinite(light.Color.g) &&
			std::isfinite(light.Color.b) &&
			std::isfinite(light.Radius) &&
			std::isfinite(light.Intensity);
	}

	bool IsFinite(const ComputeRenderer::Camera& camera)
	{
		return
			std::isfinite(camera.Position.x) &&
			std::isfinite(camera.Position.y) &&
			std::isfinite(camera.Position.z) &&
			std::isfinite(camera.Yaw) &&
			std::isfinite(camera.Pitch) &&
			std::isfinite(camera.VerticalFov);
	}

	constexpr float TimingSmoothingFactor = 0.05f;

	// Raw per-frame timings jitter far too much to read in the UI, so every
	// sample is folded into an exponential moving average instead.
	float SmoothTiming(float averageMs, float sampleMs)
	{
		if (averageMs <= 0.0f)
			return sampleMs;

		return averageMs + (sampleMs - averageMs) * TimingSmoothingFactor;
	}

	std::vector<uint32_t> ReadShaderFile(const std::string& path)
	{
		std::ifstream file(path, std::ios::ate | std::ios::binary);
		if (!file)
			throw std::runtime_error("Compute shader could not be opened: " + path);

		const size_t fileSize = static_cast<size_t>(file.tellg());
		if (fileSize == 0 || fileSize % sizeof(uint32_t) != 0)
			throw std::runtime_error("Compute shader contains invalid SPIR-V data: " + path);

		std::vector<uint32_t> code(fileSize / sizeof(uint32_t));
		file.seekg(0);
		file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(fileSize));
		return code;
	}
}

ComputeRenderer::~ComputeRenderer()
{
	Release();
}

void ComputeRenderer::Init(const std::string& shaderPath, uint32_t width, uint32_t height)
{
	m_Width = width;
	m_Height = height;
	m_FrameIndex = 0;

	// Keeps the cached forward vector consistent with the default angles instead
	// of relying on the two initializers agreeing with each other.
	UpdateCameraBasis();

	CreateOutputImages();
	CreateSceneBuffer();
	const uint32_t whitePixel = 0xffffffffu;
	m_TextureImage = std::make_unique<Walnut::Image>(
		1,
		1,
		Walnut::ImageFormat::RGBA,
		&whitePixel);
	CreateComputeDescriptors();
	CreateComputePipeline(shaderPath);
	CreateTimestampQueryPool();
}

void ComputeRenderer::CreateTimestampQueryPool()
{
	VkPhysicalDevice physicalDevice = Walnut::Application::GetPhysicalDevice();

	VkPhysicalDeviceProperties deviceProperties{};
	vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

	// A timestampPeriod of zero would make the tick to millisecond conversion
	// meaningless, so treat it the same way as missing hardware support.
	if (deviceProperties.limits.timestampComputeAndGraphics != VK_TRUE ||
		deviceProperties.limits.timestampPeriod <= 0.0f)
	{
		return;
	}

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(
		physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(
		physicalDevice, &queueFamilyCount, queueFamilies.data());

	const uint32_t queueFamilyIndex = Walnut::Application::GetQueueFamily();
	if (queueFamilyIndex >= queueFamilyCount)
		return;

	const uint32_t validBits =
		queueFamilies[queueFamilyIndex].timestampValidBits;
	if (validBits == 0)
		return;

	// Only the low validBits of every timestamp carry data; the rest must be
	// masked off before two timestamps can be subtracted.
	m_TimestampValidMask = validBits >= 64
		? ~static_cast<uint64_t>(0)
		: (static_cast<uint64_t>(1) << validBits) - 1;
	m_TimestampPeriodNs = deviceProperties.limits.timestampPeriod;

	VkQueryPoolCreateInfo queryPoolInfo{};
	queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
	queryPoolInfo.queryCount = TimestampQueryCount;
	check_vk_result(vkCreateQueryPool(
		Walnut::Application::GetDevice(),
		&queryPoolInfo,
		nullptr,
		&m_TimestampQueryPool));
}

void ComputeRenderer::ReadGpuComputeTime()
{
	if (m_TimestampQueryPool == VK_NULL_HANDLE)
		return;

	// FlushCommandBuffer already waited on a fence, so both queries have
	// finished and this read never blocks.
	std::array<uint64_t, TimestampQueryCount> timestamps{};
	const VkResult result = vkGetQueryPoolResults(
		Walnut::Application::GetDevice(),
		m_TimestampQueryPool,
		0,
		TimestampQueryCount,
		sizeof(timestamps),
		timestamps.data(),
		sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
	if (result != VK_SUCCESS)
		return;

	const uint64_t begin = timestamps[0] & m_TimestampValidMask;
	const uint64_t end = timestamps[1] & m_TimestampValidMask;
	if (end < begin)
		return;

	const float elapsedMs =
		static_cast<float>(end - begin) * m_TimestampPeriodNs / 1000000.0f;
	m_GpuComputeTimeMs = SmoothTiming(m_GpuComputeTimeMs, elapsedMs);
}

void ComputeRenderer::CreateSceneBuffer()
{
	m_Spheres = {
		Sphere{
			{ 0.0f, 0.0f, 0.0f },
			1.0f,
			{ 0.85f, 0.18f, 0.12f },
			0.15f,
			0.65f,
			MaterialType::Diffuse,
			1.5f },
		Sphere{
			{ -2.1f, 0.0f, -1.0f },
			1.0f,
			{ 0.12f, 0.35f, 0.85f },
			0.9f,
			0.05f,
			MaterialType::Metal,
			1.5f },
		Sphere{
			{ 2.1f, 0.0f, -1.0f },
			1.0f,
			{ 0.92f, 1.0f, 0.95f },
			1.0f,
			0.0f,
			MaterialType::Dielectric,
			1.5f },
		Sphere{
			{ 0.0f, -101.0f, 0.0f },
			100.0f,
			{ 0.55f, 0.55f, 0.55f },
			0.15f,
			0.55f,
			MaterialType::Diffuse,
			1.5f }
	};

	const VkDeviceSize bufferSize = sizeof(GpuSphere) * MaxSphereCount;
	CreateHostBuffer(bufferSize, m_SphereBuffer, m_SphereBufferMemory);

	CreateBvhBuffer();
	CreateLightBuffer();
	CreateTriangleBuffer();
	UploadSceneBuffer();
}

void ComputeRenderer::CreateTriangleBuffer()
{
	// A first standalone triangle proves that the renderer supports planar
	// primitives before model loading and triangle BVH construction are added.
	m_Triangles = {
		Triangle{
			{ -1.25f, 0.1f, -3.0f },
			{ 1.25f, 0.1f, -3.0f },
			{ 0.0f, 2.2f, -3.0f },
			{ 0.8f, 0.65f, 0.15f },
			0.0f,
			0.5f,
			MaterialType::Diffuse,
			1.5f }
	};
	m_ModelTriangles = m_Triangles;

	const VkDeviceSize bufferSize =
		sizeof(GpuTriangle) * MaxTriangleCount;
	CreateHostBuffer(bufferSize, m_TriangleBuffer, m_TriangleBufferMemory);
	CreateTriangleBvhBuffer();
	UploadTriangleBuffer();
}

void ComputeRenderer::CreateTriangleBvhBuffer()
{
	const VkDeviceSize bufferSize =
		sizeof(GpuBvhNode) * MaxTriangleBvhNodeCount;
	CreateHostBuffer(
		bufferSize,
		m_TriangleBvhBuffer,
		m_TriangleBvhBufferMemory);
}

void ComputeRenderer::CreateBvhBuffer()
{
	const VkDeviceSize bufferSize = sizeof(GpuBvhNode) * MaxBvhNodeCount;
	CreateHostBuffer(bufferSize, m_BvhBuffer, m_BvhBufferMemory);
}

void ComputeRenderer::CreateLightBuffer()
{
	// A warm key light on one side and a dimmer cool fill on the other, so the
	// scene shows two distinct shadow directions out of the box.
	m_Lights = {
		SphereLight{
			{ -2.5f, 5.0f, 2.0f },
			{ 1.0f, 0.95f, 0.85f },
			1.5f,
			24.0f },
		SphereLight{
			{ 3.5f, 4.0f, -2.0f },
			{ 0.45f, 0.6f, 1.0f },
			1.0f,
			12.0f }
	};

	const VkDeviceSize bufferSize = sizeof(GpuSphereLight) * MaxLightCount;
	CreateHostBuffer(bufferSize, m_LightBuffer, m_LightBufferMemory);
	UploadLightBuffer();
}

// Both scene buffers are small and rewritten whenever the scene changes, so
// they live in host visible memory and skip the staging buffer dance.
void ComputeRenderer::CreateHostBuffer(
	VkDeviceSize size,
	VkBuffer& buffer,
	VkDeviceMemory& memory) const
{
	VkDevice device = Walnut::Application::GetDevice();

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = size;
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	check_vk_result(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

	VkMemoryRequirements memoryRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

	VkMemoryAllocateInfo allocationInfo{};
	allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocationInfo.allocationSize = memoryRequirements.size;
	allocationInfo.memoryTypeIndex = FindMemoryType(
		memoryRequirements.memoryTypeBits,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	check_vk_result(vkAllocateMemory(
		device, &allocationInfo, nullptr, &memory));
	check_vk_result(vkBindBufferMemory(device, buffer, memory, 0));
}

void ComputeRenderer::WriteHostBuffer(
	VkDeviceMemory memory,
	const void* data,
	VkDeviceSize size) const
{
	VkDevice device = Walnut::Application::GetDevice();

	void* mappedMemory = nullptr;
	check_vk_result(vkMapMemory(device, memory, 0, size, 0, &mappedMemory));
	std::memcpy(mappedMemory, data, static_cast<size_t>(size));
	vkUnmapMemory(device, memory);
}

// Searches for the cheapest place to cut one range of spheres in two.
//
// The heuristic behind the search: a ray that has already entered this node's box
// enters a child box with a probability equal to the ratio of their surface areas,
// and once inside it pays one test per sphere the child holds. So the expected
// cost of a cut is one descent plus each side's sphere count weighted by its share
// of the parent area. Cutting where that sum is smallest packs the spheres a ray
// is likely to miss into a small box it can reject with a single test, which is
// what an even count split has no way to notice.
//
// Instead of trying every gap between two spheres, the range is sorted into a
// fixed number of slices along the axis and only the cuts between slices are
// scored. That keeps the search linear in the sphere count per axis while landing
// close enough to the true optimum that the difference does not show.
//
// The spheres with the smaller coordinate always end up first, because the shader
// relies on the left child being the near side when the ray travels along +axis.
ComputeRenderer::SahSplitResult ComputeRenderer::FindSahSplit(
	uint32_t start,
	uint32_t count,
	float parentSurfaceArea,
	uint32_t& splitAxis,
	uint32_t& leftCount)
{
	if (parentSurfaceArea <= 0.0f)
		return SahSplitResult::NoCandidate;

	BvhBounds centroidBounds;
	for (uint32_t offset = 0; offset < count; offset++)
		centroidBounds.Grow(m_Spheres[m_SphereOrder[start + offset]].Center);

	const glm::vec3 centroidExtent = centroidBounds.Max - centroidBounds.Min;

	float bestCost = std::numeric_limits<float>::max();
	uint32_t bestAxis = 3;
	uint32_t bestCut = 0;

	for (uint32_t axis = 0; axis < 3; axis++)
	{
		// Every centroid shares this coordinate, so no cut along it separates
		// anything and the bin scale would divide by zero.
		if (centroidExtent[axis] <= 0.0f)
			continue;

		const float axisMin = centroidBounds.Min[axis];
		const float binScale =
			static_cast<float>(BvhSahBinCount) / centroidExtent[axis];

		std::array<BvhSahBin, BvhSahBinCount> bins{};
		for (uint32_t offset = 0; offset < count; offset++)
		{
			const Sphere& sphere = m_Spheres[m_SphereOrder[start + offset]];
			BvhSahBin& bin = bins[SahBinIndex(
				sphere.Center[axis],
				axisMin,
				binScale)];
			bin.Count++;
			bin.Bounds.Grow(sphere.Center - glm::vec3(sphere.Radius));
			bin.Bounds.Grow(sphere.Center + glm::vec3(sphere.Radius));
		}

		// One sweep from the left records, for every cut, the box and the count of
		// everything before it. Without it each cut would rescan the bins and the
		// search would be quadratic in the bin count for no reason.
		std::array<float, BvhSahBinCount - 1> leftAreas{};
		std::array<uint32_t, BvhSahBinCount - 1> leftCounts{};
		BvhBounds leftBounds;
		uint32_t leftSoFar = 0;
		for (uint32_t binIndex = 0; binIndex + 1 < BvhSahBinCount; binIndex++)
		{
			leftBounds.Grow(bins[binIndex].Bounds);
			leftSoFar += bins[binIndex].Count;
			leftAreas[binIndex] = leftBounds.SurfaceArea();
			leftCounts[binIndex] = leftSoFar;
		}

		// The right side is swept backwards for the same reason, which is also why
		// the two sweeps have to run in opposite directions.
		BvhBounds rightBounds;
		uint32_t rightSoFar = 0;
		for (uint32_t binIndex = BvhSahBinCount - 1; binIndex > 0; binIndex--)
		{
			rightBounds.Grow(bins[binIndex].Bounds);
			rightSoFar += bins[binIndex].Count;

			const uint32_t cut = binIndex - 1;
			// An empty side is not a split at all, and scoring it would let a cut
			// that changes nothing look free.
			if (leftCounts[cut] == 0 || rightSoFar == 0)
				continue;

			const float cost =
				BvhTraversalCost +
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
		return SahSplitResult::NoCandidate;

	// Keeping the range whole costs one sphere test per sphere. A split only earns
	// its extra node and extra box test when it beats that. Past the leaf limit it
	// is taken anyway, because every ray that enters a long leaf pays for all of
	// it, and because a bounded leaf keeps the tree from degenerating.
	if (bestCost >= static_cast<float>(count) && count <= BvhMaxLeafSphereCount)
		return SahSplitResult::Leaf;

	const float axisMin = centroidBounds.Min[bestAxis];
	const float binScale =
		static_cast<float>(BvhSahBinCount) / centroidExtent[bestAxis];
	const auto rangeBegin = m_SphereOrder.begin() + start;
	const auto rangeEnd = rangeBegin + count;
	const auto splitPoint = std::partition(
		rangeBegin,
		rangeEnd,
		[this, bestAxis, bestCut, axisMin, binScale](uint32_t sphereIndex)
		{
			return SahBinIndex(
				m_Spheres[sphereIndex].Center[bestAxis],
				axisMin,
				binScale) <= bestCut;
		});

	leftCount = static_cast<uint32_t>(splitPoint - rangeBegin);

	// The bin counts already proved both sides hold something, and the partition
	// asks the same question of the same numbers, so this only guards against the
	// two disagreeing and building a child that owns no spheres.
	if (leftCount == 0 || leftCount >= count)
		return SahSplitResult::NoCandidate;

	splitAxis = bestAxis;
	return SahSplitResult::Split;
}

// Orders the range by centroid along its widest axis and cuts the count in half.
// Splitting by index rather than by a spatial plane always succeeds and always
// stays balanced, which is exactly what is needed when the heuristic finds no
// plane at all because every centroid coincides.
void ComputeRenderer::MedianSplit(
	uint32_t start,
	uint32_t count,
	const glm::vec3& centroidExtent,
	uint32_t& splitAxis,
	uint32_t& leftCount)
{
	uint32_t widestAxis = 0;
	if (centroidExtent.y > centroidExtent[widestAxis])
		widestAxis = 1;
	if (centroidExtent.z > centroidExtent[widestAxis])
		widestAxis = 2;

	const auto rangeBegin = m_SphereOrder.begin() + start;
	const auto rangeMiddle = rangeBegin + count / 2;
	const auto rangeEnd = rangeBegin + count;
	std::nth_element(
		rangeBegin,
		rangeMiddle,
		rangeEnd,
		[this, widestAxis](uint32_t leftIndex, uint32_t rightIndex)
		{
			return
				m_Spheres[leftIndex].Center[widestAxis] <
				m_Spheres[rightIndex].Center[widestAxis];
		});

	splitAxis = widestAxis;
	leftCount = count / 2;
}

// Groups the spheres into a binary tree of axis aligned bounding boxes so a ray
// can reject a whole branch with one box test instead of touching every sphere.
// Where each range is cut is chosen by the surface area heuristic, with the older
// median split kept reachable from the UI as the thing it has to beat.
//
// Two invariants the shader depends on: the left child always holds the spheres
// with the smaller coordinate along the node's split axis, and no node is deeper
// than the traversal stack can follow.
void ComputeRenderer::BuildBvh()
{
	const std::chrono::steady_clock::time_point buildBegin =
		std::chrono::steady_clock::now();

	const uint32_t sphereCount = GetSphereCount();

	m_SphereOrder.resize(sphereCount);
	for (uint32_t sphereIndex = 0; sphereIndex < sphereCount; sphereIndex++)
		m_SphereOrder[sphereIndex] = sphereIndex;

	m_BvhNodeCount = 0;
	m_BvhDepth = 0;
	m_BvhCost = 0.0f;

	std::array<GpuBvhNode, MaxBvhNodeCount> nodes{};

	if (sphereCount > 0)
	{
		// The same sum the split search minimises, accumulated over the finished
		// tree: what one random ray that enters the root box is expected to cost.
		float costSum = 0.0f;
		float rootSurfaceArea = 0.0f;

		std::vector<BvhBuildEntry> pending;
		pending.push_back({ 0, sphereCount, 0, 1 });
		m_BvhNodeCount = 1;

		while (!pending.empty())
		{
			const BvhBuildEntry entry = pending.back();
			pending.pop_back();
			m_BvhDepth = std::max(m_BvhDepth, entry.Depth);

			BvhBounds nodeBounds;
			BvhBounds centroidBounds;
			for (uint32_t offset = 0; offset < entry.Count; offset++)
			{
				const Sphere& sphere =
					m_Spheres[m_SphereOrder[entry.Start + offset]];
				nodeBounds.Grow(sphere.Center - glm::vec3(sphere.Radius));
				nodeBounds.Grow(sphere.Center + glm::vec3(sphere.Radius));
				centroidBounds.Grow(sphere.Center);
			}

			GpuBvhNode& node = nodes[entry.NodeIndex];
			node.BoundsMin = glm::vec4(nodeBounds.Min, 0.0f);
			node.BoundsMax = glm::vec4(nodeBounds.Max, 0.0f);

			const float nodeSurfaceArea = nodeBounds.SurfaceArea();
			// The root is the first entry pushed and the first one popped, so its
			// area is known before any child needs to be weighed against it.
			if (entry.NodeIndex == 0)
				rootSurfaceArea = nodeSurfaceArea;

			// Every split adds two nodes, so stop early enough that the fixed
			// capacity can never be exceeded, and never build deeper than the
			// shader stack can follow.
			const bool canSplit =
				entry.Count > BvhLeafSphereCount &&
				entry.Depth < BvhMaxDepth &&
				m_BvhNodeCount + 2 <= MaxBvhNodeCount;

			uint32_t splitAxis = 0;
			uint32_t leftCount = 0;
			bool splitRange = false;

			if (canSplit && m_UseSahSplit)
			{
				const SahSplitResult result = FindSahSplit(
					entry.Start,
					entry.Count,
					nodeSurfaceArea,
					splitAxis,
					leftCount);

				splitRange = result == SahSplitResult::Split;

				// No plane separates these centroids, so no split can shrink either
				// child box and a leaf is the honest answer. It still has to be a
				// bounded one, so past the leaf limit the median split takes over
				// purely to keep the range from growing without end.
				if (result == SahSplitResult::NoCandidate &&
					entry.Count > BvhMaxLeafSphereCount)
				{
					MedianSplit(
						entry.Start,
						entry.Count,
						centroidBounds.Max - centroidBounds.Min,
						splitAxis,
						leftCount);
					splitRange = true;
				}
			}
			else if (canSplit)
			{
				MedianSplit(
					entry.Start,
					entry.Count,
					centroidBounds.Max - centroidBounds.Min,
					splitAxis,
					leftCount);
				splitRange = true;
			}

			if (!splitRange)
			{
				node.Links = glm::uvec4(entry.Start, 0, entry.Count, 0);
				costSum +=
					nodeSurfaceArea * static_cast<float>(entry.Count);
				continue;
			}

			const uint32_t leftChildIndex = m_BvhNodeCount;
			const uint32_t rightChildIndex = m_BvhNodeCount + 1;
			m_BvhNodeCount += 2;

			// A SphereCount of zero marks an interior node. The split axis rides in
			// the slot that used to be padding, so the shader can tell which child
			// lies on the near side of the ray without loading anything extra.
			node.Links = glm::uvec4(
				leftChildIndex,
				rightChildIndex,
				0,
				splitAxis);
			costSum += nodeSurfaceArea * BvhTraversalCost;

			pending.push_back({
				entry.Start,
				leftCount,
				leftChildIndex,
				entry.Depth + 1 });
			pending.push_back({
				entry.Start + leftCount,
				entry.Count - leftCount,
				rightChildIndex,
				entry.Depth + 1 });
		}

		m_BvhCost = rootSurfaceArea > 0.0f
			? costSum / rootSurfaceArea
			: static_cast<float>(sphereCount);
	}

	WriteHostBuffer(m_BvhBufferMemory, nodes.data(), sizeof(nodes));

	const std::chrono::duration<float, std::milli> buildDuration =
		std::chrono::steady_clock::now() - buildBegin;
	m_BvhBuildTimeMs = buildDuration.count();
}

void ComputeRenderer::UploadSceneBuffer()
{
	// The tree decides the order the spheres are stored in, so it has to be
	// rebuilt before the sphere buffer is written.
	BuildBvh();

	std::array<GpuSphere, MaxSphereCount> gpuSpheres{};
	for (size_t sphereIndex = 0; sphereIndex < m_Spheres.size(); sphereIndex++)
	{
		const Sphere& sphere = m_Spheres[m_SphereOrder[sphereIndex]];
		gpuSpheres[sphereIndex].CenterRadius =
			glm::vec4(sphere.Center, sphere.Radius);
		gpuSpheres[sphereIndex].AlbedoReflectivity =
			glm::vec4(sphere.Albedo, sphere.Reflectivity);
		gpuSpheres[sphereIndex].RoughnessMaterial =
			glm::vec4(
				sphere.Roughness,
				static_cast<float>(sphere.Type),
				sphere.IndexOfRefraction,
				0.0f);
	}

	WriteHostBuffer(m_SphereBufferMemory, gpuSpheres.data(), sizeof(gpuSpheres));
}

void ComputeRenderer::UploadLightBuffer()
{
	std::array<GpuSphereLight, MaxLightCount> gpuLights{};
	for (size_t lightIndex = 0; lightIndex < m_Lights.size(); lightIndex++)
	{
		const SphereLight& light = m_Lights[lightIndex];
		gpuLights[lightIndex].PositionIntensity =
			glm::vec4(light.Position, light.Intensity);
		gpuLights[lightIndex].Color = glm::vec4(light.Color, 0.0f);
		gpuLights[lightIndex].RadiusSampling =
			glm::vec4(light.Radius, 0.0f, 0.0f, 0.0f);
	}

	// The stochastic path picks one light per hit, and how often it picks each
	// one decides how noisy that single sample is. Choosing in proportion to how
	// much a light can contribute keeps the estimator unbiased while making the
	// large contributors the ones that actually get sampled.
	float totalWeight = 0.0f;
	std::array<float, MaxLightCount> weights{};
	for (size_t lightIndex = 0; lightIndex < m_Lights.size(); lightIndex++)
	{
		const SphereLight& light = m_Lights[lightIndex];
		// Everything the shader multiplies into the result that the CPU can know
		// ahead of time. Distance, incidence angle and visibility are properties
		// of the shading point, so they cannot appear here; this is a bound on
		// the light rather than its actual contribution.
		weights[lightIndex] =
			light.Intensity * light.Radius * light.Radius *
			(light.Color.r + light.Color.g + light.Color.b);
		totalWeight += weights[lightIndex];
	}

	// A zero weight means zero intensity or a black colour, and either one makes
	// the shader's contribution exactly zero. Such a light gets a zero width
	// slice below, so it is never picked; dropping a term that is always zero
	// costs no accuracy. When every light is like that the whole sum is zero and
	// the slices stay empty, which the shader treats as nothing to sample.
	if (totalWeight > 0.0f)
	{
		float cumulativeWeight = 0.0f;
		for (size_t lightIndex = 0; lightIndex < m_Lights.size(); lightIndex++)
		{
			cumulativeWeight += weights[lightIndex];
			gpuLights[lightIndex].RadiusSampling.z =
				cumulativeWeight / totalWeight;
			gpuLights[lightIndex].RadiusSampling.w =
				weights[lightIndex] / totalWeight;
		}

		// Rounding can leave the last bound a hair below one, which would let a
		// random number land past every slice. Pinning it closes that gap.
		gpuLights[m_Lights.size() - 1].RadiusSampling.z = 1.0f;
	}

	WriteHostBuffer(m_LightBufferMemory, gpuLights.data(), sizeof(gpuLights));
}

void ComputeRenderer::UploadTriangleBuffer()
{
	BuildTriangleBvh();

	std::vector<GpuTriangle> gpuTriangles(MaxTriangleCount);
	for (size_t triangleIndex = 0;
		triangleIndex < m_Triangles.size();
		triangleIndex++)
	{
		const Triangle& triangle =
			m_Triangles[m_TriangleOrder[triangleIndex]];
		GpuTriangle& gpuTriangle = gpuTriangles[triangleIndex];
		gpuTriangle.Vertex0 = glm::vec4(triangle.Vertex0, 0.0f);
		gpuTriangle.Vertex1 = glm::vec4(triangle.Vertex1, 0.0f);
		gpuTriangle.Vertex2 = glm::vec4(triangle.Vertex2, 0.0f);
		gpuTriangle.AlbedoReflectivity =
			glm::vec4(triangle.Albedo, triangle.Reflectivity);
		gpuTriangle.RoughnessMaterial = glm::vec4(
			triangle.Roughness,
			static_cast<float>(triangle.Type),
			triangle.IndexOfRefraction,
			0.0f);
		gpuTriangle.Normal0 = glm::vec4(
			triangle.Normal0,
			triangle.HasVertexNormals ? 1.0f : 0.0f);
		gpuTriangle.Normal1 = glm::vec4(triangle.Normal1, 0.0f);
		gpuTriangle.Normal2 = glm::vec4(triangle.Normal2, 0.0f);
		gpuTriangle.TexCoord01 = glm::vec4(
			triangle.TexCoord0,
			triangle.TexCoord1);
		gpuTriangle.TexCoord2 = glm::vec4(
			triangle.TexCoord2,
			triangle.UsesImageTexture ? 1.0f : 0.0f,
			triangle.HasTexCoords ? 1.0f : 0.0f);
	}

	WriteHostBuffer(
		m_TriangleBufferMemory,
		gpuTriangles.data(),
		gpuTriangles.size() * sizeof(GpuTriangle));
}

void ComputeRenderer::BuildTriangleBvh()
{
	const uint32_t triangleCount = GetTriangleCount();
	m_TriangleOrder.resize(triangleCount);
	for (uint32_t triangleIndex = 0;
		triangleIndex < triangleCount;
		triangleIndex++)
	{
		m_TriangleOrder[triangleIndex] = triangleIndex;
	}

	m_TriangleBvhNodeCount = 0;
	m_TriangleBvhDepth = 0;
	std::vector<GpuBvhNode> nodes(MaxTriangleBvhNodeCount);
	if (triangleCount == 0)
	{
		WriteHostBuffer(
			m_TriangleBvhBufferMemory,
			nodes.data(),
			nodes.size() * sizeof(GpuBvhNode));
		return;
	}

	constexpr uint32_t leafTriangleCount = 4;
	std::vector<BvhBuildEntry> pending;
	pending.push_back({ 0, triangleCount, 0, 1 });
	m_TriangleBvhNodeCount = 1;

	while (!pending.empty())
	{
		const BvhBuildEntry entry = pending.back();
		pending.pop_back();
		m_TriangleBvhDepth = std::max(
			m_TriangleBvhDepth,
			entry.Depth);

		BvhBounds nodeBounds;
		BvhBounds centroidBounds;
		for (uint32_t offset = 0; offset < entry.Count; offset++)
		{
			const Triangle& triangle =
				m_Triangles[m_TriangleOrder[entry.Start + offset]];
			nodeBounds.Grow(triangle.Vertex0);
			nodeBounds.Grow(triangle.Vertex1);
			nodeBounds.Grow(triangle.Vertex2);
			centroidBounds.Grow(
				(triangle.Vertex0 + triangle.Vertex1 + triangle.Vertex2) /
				3.0f);
		}

		// A perfectly flat triangle has a zero-size box on one axis. A tiny
		// expansion makes the slab test conservative around floating-point edges.
		const glm::vec3 boundsPadding(1e-4f);
		GpuBvhNode& node = nodes[entry.NodeIndex];
		node.BoundsMin = glm::vec4(nodeBounds.Min - boundsPadding, 0.0f);
		node.BoundsMax = glm::vec4(nodeBounds.Max + boundsPadding, 0.0f);

		if (entry.Count <= leafTriangleCount)
		{
			node.Links = glm::uvec4(entry.Start, 0, entry.Count, 0);
			continue;
		}

		const glm::vec3 centroidExtent =
			centroidBounds.Max - centroidBounds.Min;
		uint32_t splitAxis = 0;
		if (centroidExtent.y > centroidExtent.x)
			splitAxis = 1;
		if (centroidExtent.z > centroidExtent[splitAxis])
			splitAxis = 2;

		const uint32_t leftCount = entry.Count / 2;
		const auto rangeBegin = m_TriangleOrder.begin() + entry.Start;
		const auto rangeMiddle = rangeBegin + leftCount;
		const auto rangeEnd = rangeBegin + entry.Count;
		std::nth_element(
			rangeBegin,
			rangeMiddle,
			rangeEnd,
			[this, splitAxis](uint32_t leftIndex, uint32_t rightIndex)
			{
				const Triangle& left = m_Triangles[leftIndex];
				const Triangle& right = m_Triangles[rightIndex];
				const float leftCentroid =
					(left.Vertex0[splitAxis] + left.Vertex1[splitAxis] +
						left.Vertex2[splitAxis]) / 3.0f;
				const float rightCentroid =
					(right.Vertex0[splitAxis] + right.Vertex1[splitAxis] +
						right.Vertex2[splitAxis]) / 3.0f;
				return leftCentroid < rightCentroid;
			});

		const uint32_t leftChildIndex = m_TriangleBvhNodeCount;
		const uint32_t rightChildIndex = m_TriangleBvhNodeCount + 1;
		m_TriangleBvhNodeCount += 2;
		node.Links = glm::uvec4(
			leftChildIndex,
			rightChildIndex,
			0,
			splitAxis);
		pending.push_back({
			entry.Start,
			leftCount,
			leftChildIndex,
			entry.Depth + 1 });
		pending.push_back({
			entry.Start + leftCount,
			entry.Count - leftCount,
			rightChildIndex,
			entry.Depth + 1 });
	}

	WriteHostBuffer(
		m_TriangleBvhBufferMemory,
		nodes.data(),
		nodes.size() * sizeof(GpuBvhNode));
}

const ComputeRenderer::Sphere& ComputeRenderer::GetSphere(uint32_t index) const
{
	return m_Spheres.at(index);
}

void ComputeRenderer::SetSphere(uint32_t index, const Sphere& sphere)
{
	m_Spheres.at(index) = SanitizeSphere(sphere);
	UploadSceneBuffer();
	ResetAccumulation();
}

bool ComputeRenderer::AddSphere()
{
	if (m_Spheres.size() >= MaxSphereCount)
		return false;

	m_Spheres.push_back({
		{ 0.0f, 0.0f, -3.0f },
		1.0f,
		{ 0.8f, 0.6f, 0.2f },
		0.1f,
		0.5f,
		MaterialType::Diffuse,
		1.5f
	});
	UploadSceneBuffer();
	ResetAccumulation();
	return true;
}

bool ComputeRenderer::RemoveSphere(uint32_t index)
{
	if (index >= m_Spheres.size())
		return false;

	m_Spheres.erase(m_Spheres.begin() + index);
	UploadSceneBuffer();
	ResetAccumulation();
	return true;
}

bool ComputeRenderer::LoadObj(
	const std::string& path,
	std::string& errorMessage)
{
	errorMessage.clear();
	std::ifstream file(path);
	if (!file)
	{
		errorMessage = "OBJ file could not be opened.";
		return false;
	}

	std::vector<glm::vec3> vertices;
	std::vector<glm::vec3> normals;
	std::vector<glm::vec2> texCoords;
	std::unordered_map<std::string, glm::vec3> materialAlbedos;
	std::unordered_map<std::string, std::filesystem::path> materialTextures;
	glm::vec3 currentAlbedo{ 0.8f, 0.65f, 0.15f };
	std::filesystem::path currentTexturePath;
	std::filesystem::path modelTexturePath;
	std::vector<Triangle> loadedTriangles;
	std::string line;
	uint32_t lineNumber = 0;

	while (std::getline(file, line))
	{
		lineNumber++;
		std::istringstream lineStream(line);
		std::string command;
		lineStream >> command;
		if (command.empty() || command[0] == '#')
			continue;

		if (command == "v")
		{
			glm::vec3 vertex{};
			if (!(lineStream >> vertex.x >> vertex.y >> vertex.z) ||
				!std::isfinite(vertex.x) ||
				!std::isfinite(vertex.y) ||
				!std::isfinite(vertex.z))
			{
				errorMessage = "Invalid vertex at OBJ line " +
					std::to_string(lineNumber) + ".";
				return false;
			}
			vertices.push_back(vertex);
			continue;
		}
		if (command == "vn")
		{
			glm::vec3 normal{};
			if (!(lineStream >> normal.x >> normal.y >> normal.z) ||
				!std::isfinite(normal.x) ||
				!std::isfinite(normal.y) ||
				!std::isfinite(normal.z) ||
				glm::dot(normal, normal) <= 1e-12f)
			{
				errorMessage = "Invalid normal at OBJ line " +
					std::to_string(lineNumber) + ".";
				return false;
			}
			normals.push_back(glm::normalize(normal));
			continue;
		}
		if (command == "vt")
		{
			glm::vec2 texCoord{};
			if (!(lineStream >> texCoord.x >> texCoord.y) ||
				!std::isfinite(texCoord.x) ||
				!std::isfinite(texCoord.y))
			{
				errorMessage = "Invalid texture coordinate at OBJ line " +
					std::to_string(lineNumber) + ".";
				return false;
			}
			texCoords.push_back(texCoord);
			continue;
		}
		if (command == "mtllib")
		{
			std::string libraryName;
			if (!(lineStream >> libraryName))
			{
				errorMessage = "Material library is missing at OBJ line " +
					std::to_string(lineNumber) + ".";
				return false;
			}

			const std::filesystem::path materialPath =
				std::filesystem::path(path).parent_path() / libraryName;
			std::ifstream materialFile(materialPath);
			if (!materialFile)
			{
				errorMessage = "MTL file could not be opened: " +
					materialPath.string();
				return false;
			}

			std::string materialLine;
			std::string materialName;
			while (std::getline(materialFile, materialLine))
			{
				std::istringstream materialStream(materialLine);
				std::string materialCommand;
				materialStream >> materialCommand;
				if (materialCommand == "newmtl")
				{
					materialStream >> materialName;
				}
				else if (materialCommand == "Kd" && !materialName.empty())
				{
					glm::vec3 albedo{};
					if (!(materialStream >> albedo.r >> albedo.g >> albedo.b) ||
						!std::isfinite(albedo.r) ||
						!std::isfinite(albedo.g) ||
						!std::isfinite(albedo.b))
					{
						errorMessage = "MTL diffuse color is invalid.";
						return false;
					}
					materialAlbedos[materialName] = glm::clamp(
						albedo,
						glm::vec3(0.0f),
						glm::vec3(1.0f));
				}
				else if (materialCommand == "map_Kd" && !materialName.empty())
				{
					std::string textureName;
					std::getline(materialStream >> std::ws, textureName);
					if (textureName.empty())
					{
						errorMessage = "MTL diffuse texture path is empty.";
						return false;
					}
					materialTextures[materialName] =
						materialPath.parent_path() / textureName;
				}
			}
			if (!materialFile.eof())
			{
				errorMessage = "MTL file could not be read completely.";
				return false;
			}
			continue;
		}
		if (command == "usemtl")
		{
			std::string materialName;
			if (!(lineStream >> materialName))
			{
				errorMessage = "Material name is missing at OBJ line " +
					std::to_string(lineNumber) + ".";
				return false;
			}
			const auto material = materialAlbedos.find(materialName);
			if (material == materialAlbedos.end())
			{
				errorMessage = "OBJ references an unknown material at line " +
					std::to_string(lineNumber) + ".";
				return false;
			}
			currentAlbedo = material->second;
			const auto texture = materialTextures.find(materialName);
			currentTexturePath = texture == materialTextures.end()
				? std::filesystem::path{}
				: texture->second;
			continue;
		}

		if (command != "f")
			continue;

		struct FaceVertex
		{
			uint32_t PositionIndex = 0;
			int64_t TexCoordIndex = -1;
			int64_t NormalIndex = -1;
		};
		std::vector<FaceVertex> faceVertices;
		std::string vertexReference;
		while (lineStream >> vertexReference)
		{
			const size_t firstSlash = vertexReference.find('/');
			const size_t secondSlash = firstSlash == std::string::npos
				? std::string::npos
				: vertexReference.find('/', firstSlash + 1);
			const std::string positionReference =
				vertexReference.substr(0, firstSlash);
			int64_t objIndex = 0;
			try
			{
				size_t parsedCharacters = 0;
				objIndex = std::stoll(positionReference, &parsedCharacters);
				if (parsedCharacters != positionReference.size())
					throw std::invalid_argument("trailing characters");
			}
			catch (const std::exception&)
			{
				errorMessage = "Invalid face index at OBJ line " +
					std::to_string(lineNumber) + ".";
				return false;
			}

			// Positive OBJ indices start at one. Negative indices are relative to
			// the end of the vertex list, with -1 naming the newest vertex.
			const int64_t resolvedIndex = objIndex > 0
				? objIndex - 1
				: static_cast<int64_t>(vertices.size()) + objIndex;
			if (objIndex == 0 || resolvedIndex < 0 ||
				resolvedIndex >= static_cast<int64_t>(vertices.size()))
			{
				errorMessage = "Face index is out of range at OBJ line " +
					std::to_string(lineNumber) + ".";
				return false;
			}

			FaceVertex faceVertex;
			faceVertex.PositionIndex = static_cast<uint32_t>(resolvedIndex);
			if (firstSlash != std::string::npos &&
				firstSlash + 1 < vertexReference.size() &&
				(firstSlash + 1 != secondSlash))
			{
				const size_t textureLength = secondSlash == std::string::npos
					? std::string::npos
					: secondSlash - firstSlash - 1;
				const std::string textureReference = vertexReference.substr(
					firstSlash + 1,
					textureLength);
				int64_t textureObjIndex = 0;
				try
				{
					size_t parsedCharacters = 0;
					textureObjIndex = std::stoll(
						textureReference,
						&parsedCharacters);
					if (parsedCharacters != textureReference.size())
						throw std::invalid_argument("trailing characters");
				}
				catch (const std::exception&)
				{
					errorMessage = "Invalid texture index at OBJ line " +
						std::to_string(lineNumber) + ".";
					return false;
				}
				const int64_t resolvedTextureIndex = textureObjIndex > 0
					? textureObjIndex - 1
					: static_cast<int64_t>(texCoords.size()) + textureObjIndex;
				if (textureObjIndex == 0 || resolvedTextureIndex < 0 ||
					resolvedTextureIndex >= static_cast<int64_t>(texCoords.size()))
				{
					errorMessage = "Texture index is out of range at OBJ line " +
						std::to_string(lineNumber) + ".";
					return false;
				}
				faceVertex.TexCoordIndex = resolvedTextureIndex;
			}
			if (secondSlash != std::string::npos &&
				secondSlash + 1 < vertexReference.size())
			{
				const std::string normalReference =
					vertexReference.substr(secondSlash + 1);
				int64_t normalObjIndex = 0;
				try
				{
					size_t parsedCharacters = 0;
					normalObjIndex = std::stoll(
						normalReference,
						&parsedCharacters);
					if (parsedCharacters != normalReference.size())
						throw std::invalid_argument("trailing characters");
				}
				catch (const std::exception&)
				{
					errorMessage = "Invalid normal index at OBJ line " +
						std::to_string(lineNumber) + ".";
					return false;
				}

				const int64_t resolvedNormalIndex = normalObjIndex > 0
					? normalObjIndex - 1
					: static_cast<int64_t>(normals.size()) + normalObjIndex;
				if (normalObjIndex == 0 || resolvedNormalIndex < 0 ||
					resolvedNormalIndex >= static_cast<int64_t>(normals.size()))
				{
					errorMessage = "Normal index is out of range at OBJ line " +
						std::to_string(lineNumber) + ".";
					return false;
				}
				faceVertex.NormalIndex = resolvedNormalIndex;
			}
			faceVertices.push_back(faceVertex);
		}

		if (faceVertices.size() < 3)
		{
			errorMessage = "Face has fewer than three vertices at OBJ line " +
				std::to_string(lineNumber) + ".";
			return false;
		}

		// A triangle fan turns (0, 1, 2, 3) into (0, 1, 2) and (0, 2, 3).
		// This is exact for triangles and convex polygon faces.
		for (size_t corner = 1; corner + 1 < faceVertices.size(); corner++)
		{
			if (loadedTriangles.size() >= MaxTriangleCount)
			{
				errorMessage = "OBJ exceeds the triangle capacity.";
				return false;
			}

			Triangle triangle{
				vertices[faceVertices[0].PositionIndex],
				vertices[faceVertices[corner].PositionIndex],
				vertices[faceVertices[corner + 1].PositionIndex],
				currentAlbedo,
				0.0f,
				0.5f,
				MaterialType::Diffuse,
				1.5f
			};
			triangle.HasVertexNormals =
				faceVertices[0].NormalIndex >= 0 &&
				faceVertices[corner].NormalIndex >= 0 &&
				faceVertices[corner + 1].NormalIndex >= 0;
			if (triangle.HasVertexNormals)
			{
				triangle.Normal0 = normals[faceVertices[0].NormalIndex];
				triangle.Normal1 = normals[faceVertices[corner].NormalIndex];
				triangle.Normal2 = normals[faceVertices[corner + 1].NormalIndex];
			}
			triangle.HasTexCoords =
				faceVertices[0].TexCoordIndex >= 0 &&
				faceVertices[corner].TexCoordIndex >= 0 &&
				faceVertices[corner + 1].TexCoordIndex >= 0;
			if (triangle.HasTexCoords)
			{
				triangle.TexCoord0 = texCoords[faceVertices[0].TexCoordIndex];
				triangle.TexCoord1 = texCoords[faceVertices[corner].TexCoordIndex];
				triangle.TexCoord2 = texCoords[faceVertices[corner + 1].TexCoordIndex];
			}
			triangle.UsesImageTexture =
				triangle.HasTexCoords && !currentTexturePath.empty();
			if (triangle.UsesImageTexture)
			{
				if (!modelTexturePath.empty() &&
					modelTexturePath != currentTexturePath)
				{
					errorMessage =
						"OBJ uses more than one diffuse texture; this renderer supports one.";
					return false;
				}
				modelTexturePath = currentTexturePath;
			}
			loadedTriangles.push_back(triangle);
		}
	}

	if (!file.eof())
	{
		errorMessage = "OBJ file could not be read completely.";
		return false;
	}
	if (vertices.empty())
	{
		errorMessage = "OBJ contains no vertices.";
		return false;
	}
	if (loadedTriangles.empty())
	{
		errorMessage = "OBJ contains no faces.";
		return false;
	}

	std::unique_ptr<Walnut::Image> loadedTexture;
	if (!modelTexturePath.empty())
	{
		int textureWidth = 0;
		int textureHeight = 0;
		int textureChannels = 0;
		if (!stbi_info(
			modelTexturePath.string().c_str(),
			&textureWidth,
			&textureHeight,
			&textureChannels) ||
			textureWidth <= 0 || textureHeight <= 0)
		{
			errorMessage = "Diffuse texture could not be decoded: " +
				modelTexturePath.string();
			return false;
		}
		loadedTexture = std::make_unique<Walnut::Image>(
			modelTexturePath.string());
	}

	// Commit only after the whole file passes validation. A malformed model
	// therefore cannot erase the triangles that are currently being displayed.
	m_ModelTriangles = std::move(loadedTriangles);
	m_ModelPath = path;
	if (loadedTexture)
		m_TextureImage = std::move(loadedTexture);
	else
	{
		const uint32_t whitePixel = 0xffffffffu;
		m_TextureImage = std::make_unique<Walnut::Image>(
			1,
			1,
			Walnut::ImageFormat::RGBA,
			&whitePixel);
	}
	UpdateTextureDescriptor();
	ApplyModelTransform();
	return true;
}

void ComputeRenderer::SetModelTransform(const ModelTransform& transform)
{
	ModelTransform sanitized = transform;
	if (!std::isfinite(sanitized.Position.x) ||
		!std::isfinite(sanitized.Position.y) ||
		!std::isfinite(sanitized.Position.z))
	{
		sanitized.Position = { 0.0f, 0.0f, 0.0f };
	}
	if (!std::isfinite(sanitized.Rotation.x) ||
		!std::isfinite(sanitized.Rotation.y) ||
		!std::isfinite(sanitized.Rotation.z))
	{
		sanitized.Rotation = { 0.0f, 0.0f, 0.0f };
	}
	if (!std::isfinite(sanitized.Scale.x) ||
		!std::isfinite(sanitized.Scale.y) ||
		!std::isfinite(sanitized.Scale.z))
	{
		sanitized.Scale = { 1.0f, 1.0f, 1.0f };
	}
	sanitized.Scale = glm::clamp(
		sanitized.Scale,
		glm::vec3(0.01f),
		glm::vec3(100.0f));

	if (sanitized.Position == m_ModelTransform.Position &&
		sanitized.Rotation == m_ModelTransform.Rotation &&
		sanitized.Scale == m_ModelTransform.Scale)
	{
		return;
	}

	m_ModelTransform = sanitized;
	ApplyModelTransform();
}

void ComputeRenderer::ApplyModelTransform()
{
	glm::mat4 transform(1.0f);
	transform = glm::translate(transform, m_ModelTransform.Position);
	transform = glm::rotate(
		transform,
		glm::radians(m_ModelTransform.Rotation.z),
		{ 0.0f, 0.0f, 1.0f });
	transform = glm::rotate(
		transform,
		glm::radians(m_ModelTransform.Rotation.y),
		{ 0.0f, 1.0f, 0.0f });
	transform = glm::rotate(
		transform,
		glm::radians(m_ModelTransform.Rotation.x),
		{ 1.0f, 0.0f, 0.0f });
	transform = glm::scale(transform, m_ModelTransform.Scale);
	const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform)));

	m_Triangles = m_ModelTriangles;
	for (Triangle& triangle : m_Triangles)
	{
		triangle.Vertex0 = glm::vec3(transform * glm::vec4(triangle.Vertex0, 1.0f));
		triangle.Vertex1 = glm::vec3(transform * glm::vec4(triangle.Vertex1, 1.0f));
		triangle.Vertex2 = glm::vec3(transform * glm::vec4(triangle.Vertex2, 1.0f));
		if (triangle.HasVertexNormals)
		{
			triangle.Normal0 = glm::normalize(normalMatrix * triangle.Normal0);
			triangle.Normal1 = glm::normalize(normalMatrix * triangle.Normal1);
			triangle.Normal2 = glm::normalize(normalMatrix * triangle.Normal2);
		}
	}

	UploadTriangleBuffer();
	ResetAccumulation();
}

bool ComputeRenderer::SaveScene(
	const std::string& path,
	std::string& errorMessage) const
{
	errorMessage.clear();
	const std::filesystem::path scenePath(path);
	if (scenePath.has_parent_path())
	{
		std::error_code directoryError;
		std::filesystem::create_directories(
			scenePath.parent_path(),
			directoryError);
		if (directoryError)
		{
			errorMessage = "Scene directory could not be created.";
			return false;
		}
	}

	std::ofstream file(scenePath, std::ios::trunc);
	if (!file)
	{
		errorMessage = "Scene file could not be opened for writing.";
		return false;
	}

	file << std::setprecision(std::numeric_limits<float>::max_digits10);
	file << "WALNUT_RAY_SCENE 8\n";
	file << "SPHERE_COUNT " << m_Spheres.size() << '\n';
	for (const Sphere& sphere : m_Spheres)
	{
		file << "SPHERE "
			<< sphere.Center.x << ' '
			<< sphere.Center.y << ' '
			<< sphere.Center.z << ' '
			<< sphere.Radius << ' '
			<< sphere.Albedo.r << ' '
			<< sphere.Albedo.g << ' '
			<< sphere.Albedo.b << ' '
			<< sphere.Reflectivity << ' '
			<< sphere.Roughness << '\n';
	}
	file << "LIGHT_COUNT " << m_Lights.size() << '\n';
	for (const SphereLight& light : m_Lights)
	{
		file << "LIGHT "
			<< light.Position.x << ' '
			<< light.Position.y << ' '
			<< light.Position.z << ' '
			<< light.Color.r << ' '
			<< light.Color.g << ' '
			<< light.Color.b << ' '
			<< light.Radius * 2.0f << ' '
			<< light.Radius * 2.0f << ' '
			<< light.Intensity << '\n';
	}
	// Appended after the lights the same way the light block was appended after
	// the spheres, so each version only adds to the end of the previous format.
	file << "CAMERA "
		<< m_Camera.Position.x << ' '
		<< m_Camera.Position.y << ' '
		<< m_Camera.Position.z << ' '
		<< m_Camera.Yaw << ' '
		<< m_Camera.Pitch << ' '
		<< m_Camera.VerticalFov << '\n';
	// Exposure belongs in the file for the same reason the camera does: it is part
	// of the picture the user set up, not of the machine they set it up on.
	file << "EXPOSURE " << m_Exposure << '\n';
	// Material types are appended as their own block, so the sphere lines remain
	// byte-for-byte compatible with versions 1-4 and old fields never move.
	file << "MATERIAL_COUNT " << m_Spheres.size() << '\n';
	for (const Sphere& sphere : m_Spheres)
	{
		file << "MATERIAL "
			<< static_cast<uint32_t>(sphere.Type)
			<< '\n';
	}
	// Version 6 appends optical density independently from material type. Keeping
	// the blocks separate lets version 5 files retain their explicit material
	// choices while receiving the default glass density when they are loaded.
	file << "IOR_COUNT " << m_Spheres.size() << '\n';
	for (const Sphere& sphere : m_Spheres)
		file << "IOR " << sphere.IndexOfRefraction << '\n';
	// Version 7 appends the exact spherical light radii. The two legacy size
	// fields above remain in place so every older block keeps its layout.
	file << "LIGHT_RADIUS_COUNT " << m_Lights.size() << '\n';
	for (const SphereLight& light : m_Lights)
		file << "LIGHT_RADIUS " << light.Radius << '\n';
	// Version 8 stores the external model reference and its world placement.
	// std::quoted preserves spaces without inventing a separate escaping format.
	file << "MODEL_PATH " << std::quoted(m_ModelPath) << '\n';
	file << "MODEL_TRANSFORM "
		<< m_ModelTransform.Position.x << ' '
		<< m_ModelTransform.Position.y << ' '
		<< m_ModelTransform.Position.z << ' '
		<< m_ModelTransform.Rotation.x << ' '
		<< m_ModelTransform.Rotation.y << ' '
		<< m_ModelTransform.Rotation.z << ' '
		<< m_ModelTransform.Scale.x << ' '
		<< m_ModelTransform.Scale.y << ' '
		<< m_ModelTransform.Scale.z << '\n';
	file.flush();

	if (!file)
	{
		errorMessage = "Scene file could not be written completely.";
		return false;
	}

	return true;
}

bool ComputeRenderer::LoadScene(
	const std::string& path,
	std::string& errorMessage)
{
	errorMessage.clear();
	std::ifstream file(path);
	if (!file)
	{
		errorMessage = "Scene file could not be opened.";
		return false;
	}

	std::string label;
	uint32_t version = 0;
	if (!(file >> label >> version) ||
		label != "WALNUT_RAY_SCENE" ||
		version < 1 || version > 8)
	{
		errorMessage = "Scene header or version is invalid.";
		return false;
	}

	uint64_t sphereCount = 0;
	if (!(file >> label >> sphereCount) || label != "SPHERE_COUNT")
	{
		errorMessage = "Sphere count is missing or invalid.";
		return false;
	}
	if (sphereCount > MaxSphereCount)
	{
		errorMessage = "Scene exceeds the sphere capacity.";
		return false;
	}

	std::vector<Sphere> loadedSpheres;
	loadedSpheres.reserve(static_cast<size_t>(sphereCount));
	for (uint64_t sphereIndex = 0; sphereIndex < sphereCount; sphereIndex++)
	{
		Sphere sphere{};
		if (!(file >> label) || label != "SPHERE" ||
			!(file
				>> sphere.Center.x
				>> sphere.Center.y
				>> sphere.Center.z
				>> sphere.Radius
				>> sphere.Albedo.r
				>> sphere.Albedo.g
				>> sphere.Albedo.b
				>> sphere.Reflectivity
				>> sphere.Roughness))
		{
			errorMessage = "Sphere data is missing or invalid.";
			return false;
		}
		if (!IsFinite(sphere))
		{
			errorMessage = "Sphere data contains a non-finite number.";
			return false;
		}
		loadedSpheres.push_back(SanitizeSphere(sphere));
	}

	// Version 1 stored exactly one light and no count line, so older scene files
	// still load and simply come back as a one light scene.
	uint64_t lightCount = 1;
	if (version >= 2)
	{
		if (!(file >> label >> lightCount) || label != "LIGHT_COUNT")
		{
			errorMessage = "Light count is missing or invalid.";
			return false;
		}
		if (lightCount > MaxLightCount)
		{
			errorMessage = "Scene exceeds the light capacity.";
			return false;
		}
	}

	std::vector<SphereLight> loadedLights;
	loadedLights.reserve(static_cast<size_t>(lightCount));
	for (uint64_t lightIndex = 0; lightIndex < lightCount; lightIndex++)
	{
		SphereLight loadedLight{};
		glm::vec2 legacySize{};
		if (!(file >> label) || label != "LIGHT" ||
			!(file
				>> loadedLight.Position.x
				>> loadedLight.Position.y
				>> loadedLight.Position.z
				>> loadedLight.Color.r
				>> loadedLight.Color.g
				>> loadedLight.Color.b
				>> legacySize.x
				>> legacySize.y
				>> loadedLight.Intensity))
		{
			errorMessage = "Light data is missing or invalid.";
			return false;
		}
		loadedLight.Radius = 0.25f * (legacySize.x + legacySize.y);
		if (!IsFinite(loadedLight))
		{
			errorMessage = "Light data contains a non-finite number.";
			return false;
		}
		loadedLights.push_back(SanitizeSphereLight(loadedLight));
	}

	// Versions 1 and 2 carry no camera, so those files keep whatever view the
	// user is looking from instead of snapping back to the default one.
	Camera loadedCamera = m_Camera;
	if (version >= 3)
	{
		if (!(file >> label) || label != "CAMERA" ||
			!(file
				>> loadedCamera.Position.x
				>> loadedCamera.Position.y
				>> loadedCamera.Position.z
				>> loadedCamera.Yaw
				>> loadedCamera.Pitch
				>> loadedCamera.VerticalFov))
		{
			errorMessage = "Camera data is missing or invalid.";
			return false;
		}
		if (!IsFinite(loadedCamera))
		{
			errorMessage = "Camera data contains a non-finite number.";
			return false;
		}
	}

	// Versions before 4 carry no exposure, so those files keep whatever the user
	// has dialled in, the same way older files keep the current camera.
	float loadedExposure = m_Exposure;
	if (version >= 4)
	{
		if (!(file >> label >> loadedExposure) || label != "EXPOSURE")
		{
			errorMessage = "Exposure is missing or invalid.";
			return false;
		}
		if (!std::isfinite(loadedExposure))
		{
			errorMessage = "Exposure is not a finite number.";
			return false;
		}
	}

	// Version 5 appends one explicit material type per sphere. Older files keep
	// the Legacy default, which preserves the exact hybrid shading they used when
	// they were saved instead of guessing whether a reflective sphere was metal.
	if (version >= 5)
	{
		uint64_t materialCount = 0;
		if (!(file >> label >> materialCount) || label != "MATERIAL_COUNT" ||
			materialCount != sphereCount)
		{
			errorMessage = "Material count is missing or does not match the spheres.";
			return false;
		}

		for (uint64_t sphereIndex = 0; sphereIndex < materialCount; sphereIndex++)
		{
			uint32_t materialType = 0;
			if (!(file >> label >> materialType) || label != "MATERIAL" ||
				materialType > static_cast<uint32_t>(MaterialType::Dielectric))
			{
				errorMessage = "Material type is missing or invalid.";
				return false;
			}

			loadedSpheres[static_cast<size_t>(sphereIndex)].Type =
				static_cast<MaterialType>(materialType);
		}
	}

	// Version 6 stores optical density after the material block. Version 5 already
	// knows the material type but has no IOR, so the Sphere default of 1.5 remains.
	if (version >= 6)
	{
		uint64_t iorCount = 0;
		if (!(file >> label >> iorCount) || label != "IOR_COUNT" ||
			iorCount != sphereCount)
		{
			errorMessage = "IOR count is missing or does not match the spheres.";
			return false;
		}

		for (uint64_t sphereIndex = 0; sphereIndex < iorCount; sphereIndex++)
		{
			float indexOfRefraction = 0.0f;
			if (!(file >> label >> indexOfRefraction) || label != "IOR")
			{
				errorMessage = "Index of refraction is missing or invalid.";
				return false;
			}
			if (!std::isfinite(indexOfRefraction))
			{
				errorMessage = "Index of refraction is not a finite number.";
				return false;
			}

			loadedSpheres[static_cast<size_t>(sphereIndex)].IndexOfRefraction =
				std::clamp(indexOfRefraction, 1.0f, 2.5f);
		}
	}

	// Version 7 replaces the invisible rectangular emitters with visible sphere
	// lights. Older versions use half the average legacy side length as a stable
	// approximation, while new files restore their exact radius here.
	if (version >= 7)
	{
		uint64_t radiusCount = 0;
		if (!(file >> label >> radiusCount) || label != "LIGHT_RADIUS_COUNT" ||
			radiusCount != lightCount)
		{
			errorMessage = "Light radius count is missing or does not match the lights.";
			return false;
		}

		for (uint64_t lightIndex = 0; lightIndex < radiusCount; lightIndex++)
		{
			float radius = 0.0f;
			if (!(file >> label >> radius) || label != "LIGHT_RADIUS")
			{
				errorMessage = "Light radius is missing or invalid.";
				return false;
			}
			if (!std::isfinite(radius))
			{
				errorMessage = "Light radius is not a finite number.";
				return false;
			}

			loadedLights[static_cast<size_t>(lightIndex)].Radius =
				std::clamp(radius, 0.05f, 20.0f);
		}
	}

	std::string loadedModelPath = m_ModelPath;
	ModelTransform loadedModelTransform = m_ModelTransform;
	if (version >= 8)
	{
		if (!(file >> label >> std::quoted(loadedModelPath)) ||
			label != "MODEL_PATH" || loadedModelPath.empty())
		{
			errorMessage = "Model path is missing or invalid.";
			return false;
		}
		if (!(file >> label) || label != "MODEL_TRANSFORM" ||
			!(file
				>> loadedModelTransform.Position.x
				>> loadedModelTransform.Position.y
				>> loadedModelTransform.Position.z
				>> loadedModelTransform.Rotation.x
				>> loadedModelTransform.Rotation.y
				>> loadedModelTransform.Rotation.z
				>> loadedModelTransform.Scale.x
				>> loadedModelTransform.Scale.y
				>> loadedModelTransform.Scale.z))
		{
			errorMessage = "Model transform is missing or invalid.";
			return false;
		}

		const glm::vec3 transformValues[] = {
			loadedModelTransform.Position,
			loadedModelTransform.Rotation,
			loadedModelTransform.Scale
		};
		for (const glm::vec3& value : transformValues)
		{
			if (!std::isfinite(value.x) ||
				!std::isfinite(value.y) ||
				!std::isfinite(value.z))
			{
				errorMessage = "Model transform contains a non-finite number.";
				return false;
			}
		}
	}

	std::string unexpectedData;
	if (file >> unexpectedData)
	{
		errorMessage = "Scene file contains unexpected trailing data.";
		return false;
	}

	if (version >= 8)
	{
		std::string modelError;
		if (!LoadObj(loadedModelPath, modelError))
		{
			errorMessage = "Scene model could not be loaded: " + modelError;
			return false;
		}
		SetModelTransform(loadedModelTransform);
	}

	m_Spheres = std::move(loadedSpheres);
	m_Lights = std::move(loadedLights);
	m_Camera = SanitizeCamera(loadedCamera);
	m_Exposure = SanitizeExposure(loadedExposure);
	UpdateCameraBasis();
	UploadSceneBuffer();
	UploadLightBuffer();
	ResetAccumulation();
	return true;
}



const ComputeRenderer::SphereLight& ComputeRenderer::GetLight(uint32_t index) const
{
	return m_Lights.at(index);
}

void ComputeRenderer::SetLight(uint32_t index, const SphereLight& light)
{
	m_Lights.at(index) = SanitizeSphereLight(light);
	UploadLightBuffer();
	ResetAccumulation();
}

bool ComputeRenderer::AddLight()
{
	if (m_Lights.size() >= MaxLightCount)
		return false;

	m_Lights.push_back({
		{ 0.0f, 5.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f },
		1.0f,
		15.0f
	});
	UploadLightBuffer();
	ResetAccumulation();
	return true;
}

bool ComputeRenderer::RemoveLight(uint32_t index)
{
	if (index >= m_Lights.size())
		return false;

	m_Lights.erase(m_Lights.begin() + index);
	UploadLightBuffer();
	ResetAccumulation();
	return true;
}

uint32_t ComputeRenderer::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const
{
	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(Walnut::Application::GetPhysicalDevice(), &memoryProperties);

	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
	{
		const bool typeMatches = (typeFilter & (1 << i)) != 0;
		const bool propertiesMatch =
			(memoryProperties.memoryTypes[i].propertyFlags & properties) == properties;

		if (typeMatches && propertiesMatch)
			return i;
	}

	throw std::runtime_error("A suitable Vulkan memory type could not be found.");
}

void ComputeRenderer::SetCamera(const Camera& camera)
{
	m_Camera = SanitizeCamera(camera);
	UpdateCameraBasis();
	ResetAccumulation();
}

// Which split built the tree changes how the spheres are grouped and therefore
// the order they are stored in, so both the tree and the sphere buffer have to be
// rewritten. It cannot change which sphere a ray hits first, so the accumulated
// samples still mean what they meant and are deliberately kept.
void ComputeRenderer::SetSahSplitEnabled(bool enabled)
{
	if (m_UseSahSplit == enabled)
		return;

	m_UseSahSplit = enabled;
	UploadSceneBuffer();
}

// Exposure only scales the averaged colour on its way to the screen, so unlike a
// camera or scene change it leaves the meaning of the accumulated samples intact
// and must not reset them.
void ComputeRenderer::SetExposure(float exposure)
{
	m_Exposure = SanitizeExposure(exposure);
}

// The bounce limit changes how far a sample can travel through reflection and
// refraction chains, so samples made with the old limit cannot stay in the same
// progressive average.
void ComputeRenderer::SetBounceCount(uint32_t bounceCount)
{
	const uint32_t sanitizedBounceCount = std::clamp(
		bounceCount,
		MinBounceCount,
		MaxBounceCount);
	if (m_BounceCount == sanitizedBounceCount)
		return;

	m_BounceCount = sanitizedBounceCount;
	ResetAccumulation();
}

// Turns the two angles the UI edits into the forward vector the shader needs.
// Yaw turns the camera around the world up axis and pitch tilts it, so this is a
// spherical to Cartesian conversion. Zero yaw and zero pitch has to come out as
// -Z because that is the direction the camera looks in by default.
void ComputeRenderer::UpdateCameraBasis()
{
	const float yaw = glm::radians(m_Camera.Yaw);
	const float pitch = glm::radians(m_Camera.Pitch);

	glm::vec3 forward;
	forward.x = glm::cos(pitch) * glm::sin(yaw);
	forward.y = glm::sin(pitch);
	forward.z = -glm::cos(pitch) * glm::cos(yaw);
	m_CameraForward = glm::normalize(forward);
}

void ComputeRenderer::ResetAccumulation()
{
	m_FrameIndex = 0;
}

void ComputeRenderer::Resize(uint32_t width, uint32_t height)
{
	if (width == 0 || height == 0 ||
		(width == m_Width && height == m_Height))
	{
		return;
	}

	VkDevice device = Walnut::Application::GetDevice();
	check_vk_result(vkDeviceWaitIdle(device));

	VkImage oldOutputImage = m_OutputImage;
	VkDeviceMemory oldOutputImageMemory = m_OutputImageMemory;
	VkImageView oldOutputImageView = m_OutputImageView;
	VkImage oldAccumulationImage = m_AccumulationImage;
	VkDeviceMemory oldAccumulationImageMemory = m_AccumulationImageMemory;
	VkImageView oldAccumulationImageView = m_AccumulationImageView;

	m_Width = width;
	m_Height = height;
	m_OutputImage = VK_NULL_HANDLE;
	m_OutputImageMemory = VK_NULL_HANDLE;
	m_OutputImageView = VK_NULL_HANDLE;
	m_AccumulationImage = VK_NULL_HANDLE;
	m_AccumulationImageMemory = VK_NULL_HANDLE;
	m_AccumulationImageView = VK_NULL_HANDLE;

	CreateOutputImages();
	UpdateComputeImageDescriptors();

	vkDestroyImageView(device, oldOutputImageView, nullptr);
	vkDestroyImage(device, oldOutputImage, nullptr);
	vkFreeMemory(device, oldOutputImageMemory, nullptr);
	vkDestroyImageView(device, oldAccumulationImageView, nullptr);
	vkDestroyImage(device, oldAccumulationImage, nullptr);
	vkFreeMemory(device, oldAccumulationImageMemory, nullptr);

	ResetAccumulation();
}

void ComputeRenderer::CreateImage(
	VkFormat format,
	VkImageUsageFlags usage,
	VkImage& image,
	VkDeviceMemory& memory,
	VkImageView& imageView)
{
	VkDevice device = Walnut::Application::GetDevice();

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = format;
	imageInfo.extent = { m_Width, m_Height, 1 };
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = usage;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	check_vk_result(vkCreateImage(device, &imageInfo, nullptr, &image));

	VkMemoryRequirements memoryRequirements;
	vkGetImageMemoryRequirements(device, image, &memoryRequirements);

	VkMemoryAllocateInfo allocationInfo{};
	allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocationInfo.allocationSize = memoryRequirements.size;
	allocationInfo.memoryTypeIndex = FindMemoryType(
		memoryRequirements.memoryTypeBits,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	check_vk_result(vkAllocateMemory(device, &allocationInfo, nullptr, &memory));
	check_vk_result(vkBindImageMemory(device, image, memory, 0));

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.layerCount = 1;
	check_vk_result(vkCreateImageView(device, &viewInfo, nullptr, &imageView));
}

void ComputeRenderer::CreateOutputImages()
{
	VkDevice device = Walnut::Application::GetDevice();

	CreateImage(
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		m_OutputImage,
		m_OutputImageMemory,
		m_OutputImageView);

	CreateImage(
		VK_FORMAT_R32G32B32A32_SFLOAT,
		VK_IMAGE_USAGE_STORAGE_BIT,
		m_AccumulationImage,
		m_AccumulationImageMemory,
		m_AccumulationImageView);

	if (m_OutputSampler == VK_NULL_HANDLE)
	{
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		samplerInfo.maxLod = 1.0f;
		check_vk_result(vkCreateSampler(
			device, &samplerInfo, nullptr, &m_OutputSampler));
	}

	VkCommandBuffer commandBuffer = Walnut::Application::GetCommandBuffer(true);

	VkImageMemoryBarrier layoutBarriers[2]{};
	for (VkImageMemoryBarrier& barrier : layoutBarriers)
	{
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.layerCount = 1;
	}
	layoutBarriers[0].image = m_OutputImage;
	layoutBarriers[1].image = m_AccumulationImage;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		2, layoutBarriers);

	Walnut::Application::FlushCommandBuffer(commandBuffer);

	if (m_ImGuiDescriptorSet == VK_NULL_HANDLE)
	{
		m_ImGuiDescriptorSet = ImGui_ImplVulkan_AddTexture(
			m_OutputSampler,
			m_OutputImageView,
			VK_IMAGE_LAYOUT_GENERAL);
	}
	else
	{
		UpdateImGuiImageDescriptor();
	}
}

void ComputeRenderer::UpdateComputeImageDescriptors()
{
	VkDescriptorImageInfo descriptorImages[2]{};
	descriptorImages[0].imageView = m_OutputImageView;
	descriptorImages[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	descriptorImages[1].imageView = m_AccumulationImageView;
	descriptorImages[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet descriptorWrites[2]{};
	for (uint32_t bindingIndex = 0; bindingIndex < 2; bindingIndex++)
	{
		descriptorWrites[bindingIndex].sType =
			VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[bindingIndex].dstSet = m_ComputeDescriptorSet;
		descriptorWrites[bindingIndex].dstBinding = bindingIndex;
		descriptorWrites[bindingIndex].descriptorCount = 1;
		descriptorWrites[bindingIndex].descriptorType =
			VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorWrites[bindingIndex].pImageInfo =
			&descriptorImages[bindingIndex];
	}

	vkUpdateDescriptorSets(
		Walnut::Application::GetDevice(),
		2,
		descriptorWrites,
		0,
		nullptr);
}

void ComputeRenderer::UpdateImGuiImageDescriptor()
{
	VkDescriptorImageInfo descriptorImage{};
	descriptorImage.sampler = m_OutputSampler;
	descriptorImage.imageView = m_OutputImageView;
	descriptorImage.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet descriptorWrite{};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = m_ImGuiDescriptorSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.descriptorType =
		VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorWrite.pImageInfo = &descriptorImage;

	vkUpdateDescriptorSets(
		Walnut::Application::GetDevice(),
		1,
		&descriptorWrite,
		0,
		nullptr);
}

void ComputeRenderer::CreateComputeDescriptors()
{
	VkDevice device = Walnut::Application::GetDevice();

	VkDescriptorSetLayoutBinding descriptorBindings[8]{};
	for (uint32_t bindingIndex = 0; bindingIndex < 2; bindingIndex++)
	{
		descriptorBindings[bindingIndex].binding = bindingIndex;
		descriptorBindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorBindings[bindingIndex].descriptorCount = 1;
		descriptorBindings[bindingIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	for (uint32_t bindingIndex = 2; bindingIndex < 7; bindingIndex++)
	{
		descriptorBindings[bindingIndex].binding = bindingIndex;
		descriptorBindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorBindings[bindingIndex].descriptorCount = 1;
		descriptorBindings[bindingIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	descriptorBindings[7].binding = 7;
	descriptorBindings[7].descriptorType =
		VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorBindings[7].descriptorCount = 1;
	descriptorBindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 8;
	layoutInfo.pBindings = descriptorBindings;
	check_vk_result(vkCreateDescriptorSetLayout(
		device, &layoutInfo, nullptr, &m_ComputeDescriptorSetLayout));

	VkDescriptorPoolSize poolSizes[3]{};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[0].descriptorCount = 2;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[1].descriptorCount = 5;
	poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	poolSizes[2].descriptorCount = 1;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1;
	poolInfo.poolSizeCount = 3;
	poolInfo.pPoolSizes = poolSizes;
	check_vk_result(vkCreateDescriptorPool(
		device, &poolInfo, nullptr, &m_ComputeDescriptorPool));

	VkDescriptorSetAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocateInfo.descriptorPool = m_ComputeDescriptorPool;
	allocateInfo.descriptorSetCount = 1;
	allocateInfo.pSetLayouts = &m_ComputeDescriptorSetLayout;
	check_vk_result(vkAllocateDescriptorSets(
		device, &allocateInfo, &m_ComputeDescriptorSet));

	VkDescriptorImageInfo descriptorImages[2]{};
	descriptorImages[0].imageView = m_OutputImageView;
	descriptorImages[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	descriptorImages[1].imageView = m_AccumulationImageView;
	descriptorImages[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	VkDescriptorBufferInfo descriptorBuffers[5]{};
	descriptorBuffers[0].buffer = m_SphereBuffer;
	descriptorBuffers[0].offset = 0;
	descriptorBuffers[0].range = VK_WHOLE_SIZE;
	descriptorBuffers[1].buffer = m_BvhBuffer;
	descriptorBuffers[1].offset = 0;
	descriptorBuffers[1].range = VK_WHOLE_SIZE;
	descriptorBuffers[2].buffer = m_LightBuffer;
	descriptorBuffers[2].offset = 0;
	descriptorBuffers[2].range = VK_WHOLE_SIZE;
	descriptorBuffers[3].buffer = m_TriangleBuffer;
	descriptorBuffers[3].offset = 0;
	descriptorBuffers[3].range = VK_WHOLE_SIZE;
	descriptorBuffers[4].buffer = m_TriangleBvhBuffer;
	descriptorBuffers[4].offset = 0;
	descriptorBuffers[4].range = VK_WHOLE_SIZE;

	VkWriteDescriptorSet descriptorWrites[7]{};
	for (uint32_t bindingIndex = 0; bindingIndex < 2; bindingIndex++)
	{
		descriptorWrites[bindingIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[bindingIndex].dstSet = m_ComputeDescriptorSet;
		descriptorWrites[bindingIndex].dstBinding = bindingIndex;
		descriptorWrites[bindingIndex].descriptorCount = 1;
		descriptorWrites[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorWrites[bindingIndex].pImageInfo = &descriptorImages[bindingIndex];
	}
	for (uint32_t bindingIndex = 2; bindingIndex < 7; bindingIndex++)
	{
		descriptorWrites[bindingIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[bindingIndex].dstSet = m_ComputeDescriptorSet;
		descriptorWrites[bindingIndex].dstBinding = bindingIndex;
		descriptorWrites[bindingIndex].descriptorCount = 1;
		descriptorWrites[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[bindingIndex].pBufferInfo = &descriptorBuffers[bindingIndex - 2];
	}
	vkUpdateDescriptorSets(device, 7, descriptorWrites, 0, nullptr);
	UpdateTextureDescriptor();
}

void ComputeRenderer::UpdateTextureDescriptor()
{
	if (m_ComputeDescriptorSet == VK_NULL_HANDLE || !m_TextureImage)
		return;

	VkDescriptorImageInfo textureInfo{};
	textureInfo.sampler = m_TextureImage->GetSampler();
	textureInfo.imageView = m_TextureImage->GetImageView();
	textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkWriteDescriptorSet textureWrite{};
	textureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	textureWrite.dstSet = m_ComputeDescriptorSet;
	textureWrite.dstBinding = 7;
	textureWrite.descriptorCount = 1;
	textureWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	textureWrite.pImageInfo = &textureInfo;
	vkUpdateDescriptorSets(
		Walnut::Application::GetDevice(),
		1,
		&textureWrite,
		0,
		nullptr);
}

void ComputeRenderer::CreateComputePipeline(const std::string& shaderPath)
{
	VkDevice device = Walnut::Application::GetDevice();
	const std::vector<uint32_t> shaderCode = ReadShaderFile(shaderPath);

	VkShaderModuleCreateInfo shaderInfo{};
	shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
	shaderInfo.pCode = shaderCode.data();

	VkShaderModule shaderModule = VK_NULL_HANDLE;
	check_vk_result(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule));

	VkPushConstantRange pushConstantRange{};
	pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pushConstantRange.offset = 0;
	pushConstantRange.size = sizeof(PushConstants);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &m_ComputeDescriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
	check_vk_result(vkCreatePipelineLayout(
		device, &pipelineLayoutInfo, nullptr, &m_ComputePipelineLayout));

	VkPipelineShaderStageCreateInfo shaderStage{};
	shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStage.module = shaderModule;
	shaderStage.pName = "main";

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage = shaderStage;
	pipelineInfo.layout = m_ComputePipelineLayout;
	check_vk_result(vkCreateComputePipelines(
		device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_ComputePipeline));

	vkDestroyShaderModule(device, shaderModule, nullptr);
}

void ComputeRenderer::Render()
{
	const std::chrono::steady_clock::time_point cpuRenderBegin =
		std::chrono::steady_clock::now();

	m_FrameIndex++;
	PushConstants pushConstants{};
	pushConstants.CameraPosition = glm::vec4(m_Camera.Position, 1.0f);
	pushConstants.CameraForward = glm::vec4(m_CameraForward, 0.0f);
	pushConstants.FrameIndex = m_FrameIndex;
	pushConstants.VerticalFov = m_Camera.VerticalFov;
	pushConstants.Exposure = m_Exposure;
	pushConstants.SphereCount = GetSphereCount();
	// A node count of zero would leave the shader without a root to start from,
	// so the brute force loop stays as the fallback.
	const bool traverseBvh = m_UseBvh && m_BvhNodeCount > 0;
	pushConstants.SceneSettings = glm::uvec4(
		traverseBvh ? 1u : 0u,
		GetLightCount(),
		m_UseStochasticLights ? 1u : 0u,
		m_BounceCount);
	pushConstants.TriangleCount = GetTriangleCount();

	VkCommandBuffer commandBuffer = Walnut::Application::GetCommandBuffer(true);

	// Queries hold results from the previous frame and must be reset before
	// they can be written again.
	if (m_TimestampQueryPool != VK_NULL_HANDLE)
	{
		vkCmdResetQueryPool(
			commandBuffer,
			m_TimestampQueryPool,
			0,
			TimestampQueryCount);
	}

	vkCmdBindPipeline(
		commandBuffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		m_ComputePipeline);

	vkCmdBindDescriptorSets(
		commandBuffer,
		VK_PIPELINE_BIND_POINT_COMPUTE,
		m_ComputePipelineLayout,
		0,
		1,
		&m_ComputeDescriptorSet,
		0,
		nullptr);

	vkCmdPushConstants(
		commandBuffer,
		m_ComputePipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT,
		0,
		sizeof(PushConstants),
		&pushConstants);

	const uint32_t groupCountX = (m_Width + 7) / 8;
	const uint32_t groupCountY = (m_Height + 7) / 8;

	// The two timestamps bracket only the dispatch, so their difference is the
	// time the GPU itself spent tracing rays.
	if (m_TimestampQueryPool != VK_NULL_HANDLE)
	{
		vkCmdWriteTimestamp(
			commandBuffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			m_TimestampQueryPool,
			0);
	}

	vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

	if (m_TimestampQueryPool != VK_NULL_HANDLE)
	{
		vkCmdWriteTimestamp(
			commandBuffer,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			m_TimestampQueryPool,
			1);
	}

	VkImageMemoryBarrier computeBarriers[2]{};
	for (VkImageMemoryBarrier& barrier : computeBarriers)
	{
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
		barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.layerCount = 1;
	}

	computeBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	computeBarriers[0].image = m_OutputImage;
	computeBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	computeBarriers[1].image = m_AccumulationImage;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		2, computeBarriers);

	Walnut::Application::FlushCommandBuffer(commandBuffer);

	ReadGpuComputeTime();

	// This includes the fence wait inside FlushCommandBuffer, so it measures
	// how long the whole render call blocks the CPU rather than CPU-only work.
	// Compute and present share one queue, so on a vsync limited swapchain this
	// wait also absorbs the wait for the display and tracks the frame time
	// instead of the dispatch cost. Use m_GpuComputeTimeMs to judge the shader.
	const std::chrono::duration<float, std::milli> cpuRenderDuration =
		std::chrono::steady_clock::now() - cpuRenderBegin;
	m_CpuRenderTimeMs =
		SmoothTiming(m_CpuRenderTimeMs, cpuRenderDuration.count());
}


void ComputeRenderer::Release()
{
	if (m_OutputImage == VK_NULL_HANDLE)
		return;

	VkPipeline pipeline = m_ComputePipeline;
	VkPipelineLayout pipelineLayout = m_ComputePipelineLayout;
	VkDescriptorPool descriptorPool = m_ComputeDescriptorPool;
	VkDescriptorSetLayout descriptorSetLayout = m_ComputeDescriptorSetLayout;
	VkSampler sampler = m_OutputSampler;
	VkImageView outputImageView = m_OutputImageView;
	VkImage outputImage = m_OutputImage;
	VkDeviceMemory outputImageMemory = m_OutputImageMemory;
	VkImageView accumulationImageView = m_AccumulationImageView;
	VkImage accumulationImage = m_AccumulationImage;
	VkDeviceMemory accumulationImageMemory = m_AccumulationImageMemory;
	VkBuffer sphereBuffer = m_SphereBuffer;
	VkDeviceMemory sphereBufferMemory = m_SphereBufferMemory;
	VkBuffer bvhBuffer = m_BvhBuffer;
	VkDeviceMemory bvhBufferMemory = m_BvhBufferMemory;
	VkBuffer lightBuffer = m_LightBuffer;
	VkDeviceMemory lightBufferMemory = m_LightBufferMemory;
	VkBuffer triangleBuffer = m_TriangleBuffer;
	VkDeviceMemory triangleBufferMemory = m_TriangleBufferMemory;
	VkBuffer triangleBvhBuffer = m_TriangleBvhBuffer;
	VkDeviceMemory triangleBvhBufferMemory = m_TriangleBvhBufferMemory;
	VkQueryPool timestampQueryPool = m_TimestampQueryPool;

	Walnut::Application::SubmitResourceFree(
		[pipeline, pipelineLayout, descriptorPool, descriptorSetLayout,
		 sampler, outputImageView, outputImage, outputImageMemory,
		 accumulationImageView, accumulationImage, accumulationImageMemory,
			 sphereBuffer, sphereBufferMemory, bvhBuffer, bvhBufferMemory,
			 lightBuffer, lightBufferMemory, triangleBuffer,
			 triangleBufferMemory, triangleBvhBuffer,
			 triangleBvhBufferMemory, timestampQueryPool]()
		{
			VkDevice device = Walnut::Application::GetDevice();
			if (timestampQueryPool != VK_NULL_HANDLE)
				vkDestroyQueryPool(device, timestampQueryPool, nullptr);
			vkDestroyPipeline(device, pipeline, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyDescriptorPool(device, descriptorPool, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
			vkDestroySampler(device, sampler, nullptr);
			vkDestroyImageView(device, outputImageView, nullptr);
			vkDestroyImage(device, outputImage, nullptr);
			vkFreeMemory(device, outputImageMemory, nullptr);
			vkDestroyImageView(device, accumulationImageView, nullptr);
			vkDestroyImage(device, accumulationImage, nullptr);
			vkFreeMemory(device, accumulationImageMemory, nullptr);
			vkDestroyBuffer(device, sphereBuffer, nullptr);
			vkFreeMemory(device, sphereBufferMemory, nullptr);
			vkDestroyBuffer(device, bvhBuffer, nullptr);
			vkFreeMemory(device, bvhBufferMemory, nullptr);
			vkDestroyBuffer(device, lightBuffer, nullptr);
			vkFreeMemory(device, lightBufferMemory, nullptr);
			vkDestroyBuffer(device, triangleBuffer, nullptr);
			vkFreeMemory(device, triangleBufferMemory, nullptr);
			vkDestroyBuffer(device, triangleBvhBuffer, nullptr);
			vkFreeMemory(device, triangleBvhBufferMemory, nullptr);
		});

	m_OutputImage = VK_NULL_HANDLE;
	m_TimestampQueryPool = VK_NULL_HANDLE;
}
