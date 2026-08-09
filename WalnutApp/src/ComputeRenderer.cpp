#include "ComputeRenderer.h"

#include "Walnut/Application.h"

#include "backends/imgui_impl_vulkan.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{
	// SceneSettings.x is 1 when the BVH should be traversed, .y is the number of
	// lights in the light buffer, .z is 1 when each hit samples a single light
	// chosen by weight instead of all of them. The last slot is reserved padding.
	struct alignas(16) PushConstants
	{
		glm::vec4 CameraPosition;
		glm::vec4 CameraForward;
		uint32_t FrameIndex;
		float VerticalFov;
		float Exposure;
		uint32_t SphereCount;
		glm::uvec4 SceneSettings;
	};

	// Moving the light out of the push constant and into a storage buffer took
	// this back down to 64 bytes, half of the 128 bytes Vulkan guarantees.
	static_assert(sizeof(PushConstants) == 64);

	struct alignas(16) GpuSphere
	{
		glm::vec4 CenterRadius;
		glm::vec4 AlbedoReflectivity;
		glm::vec4 RoughnessPadding;
	};

	static_assert(sizeof(GpuSphere) == 48);

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
	struct alignas(16) GpuAreaLight
	{
		glm::vec4 PositionIntensity;
		glm::vec4 Color;
		glm::vec4 SizePadding;
	};

	static_assert(sizeof(GpuAreaLight) == 48);

	constexpr uint32_t BvhLeafSphereCount = 2;

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
		return result;
	}

	ComputeRenderer::AreaLight SanitizeAreaLight(
		const ComputeRenderer::AreaLight& light)
	{
		ComputeRenderer::AreaLight result = light;
		result.Color = glm::clamp(
			result.Color,
			glm::vec3(0.0f),
			glm::vec3(1.0f));
		result.Size = glm::clamp(
			result.Size,
			glm::vec2(0.05f),
			glm::vec2(20.0f));
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
			std::isfinite(sphere.Roughness);
	}

	bool IsFinite(const ComputeRenderer::AreaLight& light)
	{
		return
			std::isfinite(light.Position.x) &&
			std::isfinite(light.Position.y) &&
			std::isfinite(light.Position.z) &&
			std::isfinite(light.Color.r) &&
			std::isfinite(light.Color.g) &&
			std::isfinite(light.Color.b) &&
			std::isfinite(light.Size.x) &&
			std::isfinite(light.Size.y) &&
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
			0.65f },
		Sphere{
			{ -2.1f, 0.0f, -1.0f },
			1.0f,
			{ 0.12f, 0.35f, 0.85f },
			0.75f,
			0.05f },
		Sphere{
			{ 2.1f, 0.0f, -1.0f },
			1.0f,
			{ 0.15f, 0.75f, 0.28f },
			0.35f,
			0.35f },
		Sphere{
			{ 0.0f, -101.0f, 0.0f },
			100.0f,
			{ 0.55f, 0.55f, 0.55f },
			0.15f,
			0.55f }
	};

	const VkDeviceSize bufferSize = sizeof(GpuSphere) * MaxSphereCount;
	CreateHostBuffer(bufferSize, m_SphereBuffer, m_SphereBufferMemory);

	CreateBvhBuffer();
	CreateLightBuffer();
	UploadSceneBuffer();
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
		AreaLight{
			{ -2.5f, 5.0f, 2.0f },
			{ 1.0f, 0.95f, 0.85f },
			{ 3.0f, 3.0f },
			24.0f },
		AreaLight{
			{ 3.5f, 4.0f, -2.0f },
			{ 0.45f, 0.6f, 1.0f },
			{ 2.0f, 2.0f },
			12.0f }
	};

	const VkDeviceSize bufferSize = sizeof(GpuAreaLight) * MaxLightCount;
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

// Groups the spheres into a binary tree of axis aligned bounding boxes so a ray
// can reject a whole branch with one box test instead of touching every sphere.
// The split is a median split: the range is ordered by centroid along its widest
// axis and cut in half. Splitting by index rather than by a spatial plane keeps
// the tree balanced even when many centroids coincide.
void ComputeRenderer::BuildBvh()
{
	const uint32_t sphereCount = GetSphereCount();

	m_SphereOrder.resize(sphereCount);
	for (uint32_t sphereIndex = 0; sphereIndex < sphereCount; sphereIndex++)
		m_SphereOrder[sphereIndex] = sphereIndex;

	m_BvhNodeCount = 0;
	m_BvhDepth = 0;

	std::array<GpuBvhNode, MaxBvhNodeCount> nodes{};

	if (sphereCount > 0)
	{
		std::vector<BvhBuildEntry> pending;
		pending.push_back({ 0, sphereCount, 0, 1 });
		m_BvhNodeCount = 1;

		while (!pending.empty())
		{
			const BvhBuildEntry entry = pending.back();
			pending.pop_back();
			m_BvhDepth = std::max(m_BvhDepth, entry.Depth);

			glm::vec3 boundsMin(std::numeric_limits<float>::max());
			glm::vec3 boundsMax(std::numeric_limits<float>::lowest());
			glm::vec3 centroidMin(std::numeric_limits<float>::max());
			glm::vec3 centroidMax(std::numeric_limits<float>::lowest());
			for (uint32_t offset = 0; offset < entry.Count; offset++)
			{
				const Sphere& sphere =
					m_Spheres[m_SphereOrder[entry.Start + offset]];
				boundsMin = glm::min(
					boundsMin, sphere.Center - glm::vec3(sphere.Radius));
				boundsMax = glm::max(
					boundsMax, sphere.Center + glm::vec3(sphere.Radius));
				centroidMin = glm::min(centroidMin, sphere.Center);
				centroidMax = glm::max(centroidMax, sphere.Center);
			}

			GpuBvhNode& node = nodes[entry.NodeIndex];
			node.BoundsMin = glm::vec4(boundsMin, 0.0f);
			node.BoundsMax = glm::vec4(boundsMax, 0.0f);

			// Every split adds two nodes, so stop early enough that the fixed
			// capacity can never be exceeded.
			const bool canSplit =
				entry.Count > BvhLeafSphereCount &&
				m_BvhNodeCount + 2 <= MaxBvhNodeCount;
			if (!canSplit)
			{
				node.Links = glm::uvec4(entry.Start, 0, entry.Count, 0);
				continue;
			}

			const glm::vec3 centroidExtent = centroidMax - centroidMin;
			uint32_t splitAxis = 0;
			if (centroidExtent.y > centroidExtent[splitAxis])
				splitAxis = 1;
			if (centroidExtent.z > centroidExtent[splitAxis])
				splitAxis = 2;

			const auto rangeBegin = m_SphereOrder.begin() + entry.Start;
			const auto rangeMiddle = rangeBegin + entry.Count / 2;
			const auto rangeEnd = rangeBegin + entry.Count;
			std::nth_element(
				rangeBegin,
				rangeMiddle,
				rangeEnd,
				[this, splitAxis](uint32_t leftIndex, uint32_t rightIndex)
				{
					return
						m_Spheres[leftIndex].Center[splitAxis] <
						m_Spheres[rightIndex].Center[splitAxis];
				});

			const uint32_t leftChildIndex = m_BvhNodeCount;
			const uint32_t rightChildIndex = m_BvhNodeCount + 1;
			m_BvhNodeCount += 2;

			// A SphereCount of zero marks an interior node.
			node.Links = glm::uvec4(leftChildIndex, rightChildIndex, 0, 0);

			const uint32_t leftCount = entry.Count / 2;
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
	}

	WriteHostBuffer(m_BvhBufferMemory, nodes.data(), sizeof(nodes));
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
		gpuSpheres[sphereIndex].RoughnessPadding =
			glm::vec4(sphere.Roughness, 0.0f, 0.0f, 0.0f);
	}

	WriteHostBuffer(m_SphereBufferMemory, gpuSpheres.data(), sizeof(gpuSpheres));
}

void ComputeRenderer::UploadLightBuffer()
{
	std::array<GpuAreaLight, MaxLightCount> gpuLights{};
	for (size_t lightIndex = 0; lightIndex < m_Lights.size(); lightIndex++)
	{
		const AreaLight& light = m_Lights[lightIndex];
		gpuLights[lightIndex].PositionIntensity =
			glm::vec4(light.Position, light.Intensity);
		gpuLights[lightIndex].Color = glm::vec4(light.Color, 0.0f);
		gpuLights[lightIndex].SizePadding =
			glm::vec4(light.Size, 0.0f, 0.0f);
	}

	// The stochastic path picks one light per hit, and how often it picks each
	// one decides how noisy that single sample is. Choosing in proportion to how
	// much a light can contribute keeps the estimator unbiased while making the
	// large contributors the ones that actually get sampled.
	float totalWeight = 0.0f;
	std::array<float, MaxLightCount> weights{};
	for (size_t lightIndex = 0; lightIndex < m_Lights.size(); lightIndex++)
	{
		const AreaLight& light = m_Lights[lightIndex];
		// Everything the shader multiplies into the result that the CPU can know
		// ahead of time. Distance, incidence angle and visibility are properties
		// of the shading point, so they cannot appear here; this is a bound on
		// the light rather than its actual contribution.
		weights[lightIndex] =
			light.Intensity *
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
			gpuLights[lightIndex].SizePadding.z =
				cumulativeWeight / totalWeight;
			gpuLights[lightIndex].SizePadding.w =
				weights[lightIndex] / totalWeight;
		}

		// Rounding can leave the last bound a hair below one, which would let a
		// random number land past every slice. Pinning it closes that gap.
		gpuLights[m_Lights.size() - 1].SizePadding.z = 1.0f;
	}

	WriteHostBuffer(m_LightBufferMemory, gpuLights.data(), sizeof(gpuLights));
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
		0.5f
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
	file << "WALNUT_RAY_SCENE 4\n";
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
	for (const AreaLight& light : m_Lights)
	{
		file << "LIGHT "
			<< light.Position.x << ' '
			<< light.Position.y << ' '
			<< light.Position.z << ' '
			<< light.Color.r << ' '
			<< light.Color.g << ' '
			<< light.Color.b << ' '
			<< light.Size.x << ' '
			<< light.Size.y << ' '
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
		version < 1 || version > 4)
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

	std::vector<AreaLight> loadedLights;
	loadedLights.reserve(static_cast<size_t>(lightCount));
	for (uint64_t lightIndex = 0; lightIndex < lightCount; lightIndex++)
	{
		AreaLight loadedLight{};
		if (!(file >> label) || label != "LIGHT" ||
			!(file
				>> loadedLight.Position.x
				>> loadedLight.Position.y
				>> loadedLight.Position.z
				>> loadedLight.Color.r
				>> loadedLight.Color.g
				>> loadedLight.Color.b
				>> loadedLight.Size.x
				>> loadedLight.Size.y
				>> loadedLight.Intensity))
		{
			errorMessage = "Area light data is missing or invalid.";
			return false;
		}
		if (!IsFinite(loadedLight))
		{
			errorMessage = "Area light data contains a non-finite number.";
			return false;
		}
		loadedLights.push_back(SanitizeAreaLight(loadedLight));
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

	std::string unexpectedData;
	if (file >> unexpectedData)
	{
		errorMessage = "Scene file contains unexpected trailing data.";
		return false;
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

const ComputeRenderer::AreaLight& ComputeRenderer::GetLight(uint32_t index) const
{
	return m_Lights.at(index);
}

void ComputeRenderer::SetLight(uint32_t index, const AreaLight& light)
{
	m_Lights.at(index) = SanitizeAreaLight(light);
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
		{ 2.0f, 2.0f },
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

// Exposure only scales the averaged colour on its way to the screen, so unlike a
// camera or scene change it leaves the meaning of the accumulated samples intact
// and must not reset them.
void ComputeRenderer::SetExposure(float exposure)
{
	m_Exposure = SanitizeExposure(exposure);
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

	VkDescriptorSetLayoutBinding descriptorBindings[5]{};
	for (uint32_t bindingIndex = 0; bindingIndex < 2; bindingIndex++)
	{
		descriptorBindings[bindingIndex].binding = bindingIndex;
		descriptorBindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorBindings[bindingIndex].descriptorCount = 1;
		descriptorBindings[bindingIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	for (uint32_t bindingIndex = 2; bindingIndex < 5; bindingIndex++)
	{
		descriptorBindings[bindingIndex].binding = bindingIndex;
		descriptorBindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorBindings[bindingIndex].descriptorCount = 1;
		descriptorBindings[bindingIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 5;
	layoutInfo.pBindings = descriptorBindings;
	check_vk_result(vkCreateDescriptorSetLayout(
		device, &layoutInfo, nullptr, &m_ComputeDescriptorSetLayout));

	VkDescriptorPoolSize poolSizes[2]{};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[0].descriptorCount = 2;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[1].descriptorCount = 3;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1;
	poolInfo.poolSizeCount = 2;
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
	VkDescriptorBufferInfo descriptorBuffers[3]{};
	descriptorBuffers[0].buffer = m_SphereBuffer;
	descriptorBuffers[0].offset = 0;
	descriptorBuffers[0].range = VK_WHOLE_SIZE;
	descriptorBuffers[1].buffer = m_BvhBuffer;
	descriptorBuffers[1].offset = 0;
	descriptorBuffers[1].range = VK_WHOLE_SIZE;
	descriptorBuffers[2].buffer = m_LightBuffer;
	descriptorBuffers[2].offset = 0;
	descriptorBuffers[2].range = VK_WHOLE_SIZE;

	VkWriteDescriptorSet descriptorWrites[5]{};
	for (uint32_t bindingIndex = 0; bindingIndex < 2; bindingIndex++)
	{
		descriptorWrites[bindingIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[bindingIndex].dstSet = m_ComputeDescriptorSet;
		descriptorWrites[bindingIndex].dstBinding = bindingIndex;
		descriptorWrites[bindingIndex].descriptorCount = 1;
		descriptorWrites[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorWrites[bindingIndex].pImageInfo = &descriptorImages[bindingIndex];
	}
	for (uint32_t bindingIndex = 2; bindingIndex < 5; bindingIndex++)
	{
		descriptorWrites[bindingIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[bindingIndex].dstSet = m_ComputeDescriptorSet;
		descriptorWrites[bindingIndex].dstBinding = bindingIndex;
		descriptorWrites[bindingIndex].descriptorCount = 1;
		descriptorWrites[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrites[bindingIndex].pBufferInfo = &descriptorBuffers[bindingIndex - 2];
	}
	vkUpdateDescriptorSets(device, 5, descriptorWrites, 0, nullptr);
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
		0);

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
	VkQueryPool timestampQueryPool = m_TimestampQueryPool;

	Walnut::Application::SubmitResourceFree(
		[pipeline, pipelineLayout, descriptorPool, descriptorSetLayout,
		 sampler, outputImageView, outputImage, outputImageMemory,
		 accumulationImageView, accumulationImage, accumulationImageMemory,
		 sphereBuffer, sphereBufferMemory, bvhBuffer, bvhBufferMemory,
		 lightBuffer, lightBufferMemory, timestampQueryPool]()
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
		});

	m_OutputImage = VK_NULL_HANDLE;
	m_TimestampQueryPool = VK_NULL_HANDLE;
}
