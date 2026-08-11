#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "Walnut/Image.h"

class ComputeRenderer
{
public:
	enum class MaterialType : uint32_t
	{
		Legacy = 0,
		Diffuse = 1,
		Metal = 2,
		Dielectric = 3
	};

	struct Sphere
	{
		glm::vec3 Center;
		float Radius;
		glm::vec3 Albedo;
		float Reflectivity;
		float Roughness;
		MaterialType Type = MaterialType::Legacy;
		float IndexOfRefraction = 1.5f;
	};

	struct SphereLight
	{
		glm::vec3 Position;
		glm::vec3 Color;
		float Radius;
		float Intensity;
	};

	struct Triangle
	{
		glm::vec3 Vertex0;
		glm::vec3 Vertex1;
		glm::vec3 Vertex2;
		glm::vec3 Albedo;
		float Reflectivity;
		float Roughness;
		MaterialType Type = MaterialType::Diffuse;
		float IndexOfRefraction = 1.5f;
		glm::vec3 Normal0{ 0.0f };
		glm::vec3 Normal1{ 0.0f };
		glm::vec3 Normal2{ 0.0f };
		bool HasVertexNormals = false;
		glm::vec2 TexCoord0{ 0.0f };
		glm::vec2 TexCoord1{ 0.0f };
		glm::vec2 TexCoord2{ 0.0f };
		bool HasTexCoords = false;
		bool UsesImageTexture = false;
	};

	struct ModelTransform
	{
		glm::vec3 Position{ 0.0f };
		glm::vec3 Rotation{ 0.0f };
		glm::vec3 Scale{ 1.0f };
	};

	// Yaw and pitch are stored in degrees instead of a forward vector because
	// they are what the UI edits and what a scene file can round trip; the
	// renderer derives the forward vector from them.
	struct Camera
	{
		glm::vec3 Position = { 0.0f, 0.0f, 3.0f };
		float Yaw = 0.0f;
		float Pitch = 0.0f;
		float VerticalFov = 45.0f;
	};

	static constexpr uint32_t MaxSphereCount = 512;
	static constexpr uint32_t MaxTriangleCount = 4096;
	static constexpr uint32_t MaxTriangleBvhNodeCount = 2 * MaxTriangleCount;

	// A binary tree over N primitives needs at most 2N-1 nodes, so this capacity
	// can always hold a tree built over MaxSphereCount spheres.
	static constexpr uint32_t MaxBvhNodeCount = 2 * MaxSphereCount;

	// Every light costs one shadow ray per bounce, so this capacity is kept far
	// smaller than the sphere capacity.
	static constexpr uint32_t MaxLightCount = 8;

	// The UI slider and the clamp a scene file goes through read the same two
	// numbers, so an exposure that came out of a file is always one the control
	// can still represent.
	static constexpr float MinExposure = 0.1f;
	static constexpr float MaxExposure = 4.0f;
	static constexpr uint32_t MinBounceCount = 1;
	static constexpr uint32_t MaxBounceCount = 10;

	~ComputeRenderer();

	void Init(const std::string& shaderPath, uint32_t width, uint32_t height);
	void Render();
	void Resize(uint32_t width, uint32_t height);
	const Camera& GetCamera() const { return m_Camera; }
	void SetCamera(const Camera& camera);
	const glm::vec3& GetCameraForward() const { return m_CameraForward; }
	float GetExposure() const { return m_Exposure; }
	void SetExposure(float exposure);
	uint32_t GetBounceCount() const { return m_BounceCount; }
	void SetBounceCount(uint32_t bounceCount);
	const Sphere& GetSphere(uint32_t index) const;
	void SetSphere(uint32_t index, const Sphere& sphere);
	bool AddSphere();
	bool RemoveSphere(uint32_t index);
	bool SaveScene(const std::string& path, std::string& errorMessage) const;
	bool LoadScene(const std::string& path, std::string& errorMessage);
	bool LoadObj(const std::string& path, std::string& errorMessage);
	bool LoadEnvironmentMap(const std::string& path, std::string& errorMessage);
	void ClearEnvironmentMap();
	float GetEnvironmentIntensity() const { return m_EnvironmentIntensity; }
	void SetEnvironmentIntensity(float intensity);
	float GetEnvironmentRotation() const { return m_EnvironmentRotation; }
	void SetEnvironmentRotation(float rotationDegrees);
	const std::string& GetModelPath() const { return m_ModelPath; }
	const ModelTransform& GetModelTransform() const { return m_ModelTransform; }
	void SetModelTransform(const ModelTransform& transform);
	const SphereLight& GetLight(uint32_t index) const;
	void SetLight(uint32_t index, const SphereLight& light);
	bool AddLight();
	bool RemoveLight(uint32_t index);

	void SetBvhEnabled(bool enabled) { m_UseBvh = enabled; }
	bool IsBvhEnabled() const { return m_UseBvh; }
	// Which split the tree is built with changes how many boxes and spheres a ray
	// has to test, never which sphere ends up nearest, so it costs a rebuild and
	// an upload but leaves the accumulated samples meaningful.
	void SetSahSplitEnabled(bool enabled);
	bool IsSahSplitEnabled() const { return m_UseSahSplit; }
	// Changing the sampling strategy changes what every sample means, so the
	// already accumulated frames cannot be mixed with the new ones.
	void SetStochasticLightsEnabled(bool enabled)
	{
		if (m_UseStochasticLights == enabled)
			return;

		m_UseStochasticLights = enabled;
		ResetAccumulation();
	}
	bool AreStochasticLightsEnabled() const { return m_UseStochasticLights; }
	void ResetAccumulation();

	VkDescriptorSet GetImageDescriptorSet() const { return m_ImGuiDescriptorSet; }
	uint32_t GetWidth() const { return m_Width; }
	uint32_t GetHeight() const { return m_Height; }
	uint32_t GetFrameIndex() const { return m_FrameIndex; }
	uint32_t GetSphereCount() const { return static_cast<uint32_t>(m_Spheres.size()); }
	uint32_t GetTriangleCount() const { return static_cast<uint32_t>(m_Triangles.size()); }
	uint32_t GetTriangleBvhNodeCount() const { return m_TriangleBvhNodeCount; }
	uint32_t GetTriangleBvhDepth() const { return m_TriangleBvhDepth; }
	uint32_t GetLightCount() const { return static_cast<uint32_t>(m_Lights.size()); }
	uint32_t GetBvhNodeCount() const { return m_BvhNodeCount; }
	uint32_t GetBvhDepth() const { return m_BvhDepth; }
	// The expected number of sphere tests one random ray costs, under the same
	// heuristic the split search minimises. It compares two trees over the same
	// scene without waiting for a timing average to settle.
	float GetBvhCost() const { return m_BvhCost; }
	float GetBvhBuildTimeMs() const { return m_BvhBuildTimeMs; }
	float GetCpuRenderTimeMs() const { return m_CpuRenderTimeMs; }
	float GetGpuComputeTimeMs() const { return m_GpuComputeTimeMs; }
	bool AreGpuTimestampsSupported() const { return m_TimestampQueryPool != VK_NULL_HANDLE; }
	bool IsRayQuerySupported() const { return m_RayQuerySupported; }
	bool IsRayQueryEnabled() const { return m_UseRayQuery; }
	void SetRayQueryEnabled(bool enabled);

private:
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
	void UpdateCameraBasis();
	void CreateImage(
		VkFormat format,
		VkImageUsageFlags usage,
		VkImage& image,
		VkDeviceMemory& memory,
		VkImageView& imageView);
	void CreateOutputImages();
	void UpdateComputeImageDescriptors();
	void UpdateImGuiImageDescriptor();
	void CreateSceneBuffer();
	void CreateBvhBuffer();
	void CreateLightBuffer();
	void CreateTriangleBuffer();
	void CreateTriangleBvhBuffer();
	void CreateHostBuffer(
		VkDeviceSize size,
		VkBuffer& buffer,
		VkDeviceMemory& memory) const;
	void WriteHostBuffer(
		VkDeviceMemory memory,
		const void* data,
		VkDeviceSize size) const;
	void UploadSceneBuffer();
	void UploadLightBuffer();
	void UploadTriangleBuffer();
	void BuildTriangleAccelerationStructure();
	void ReleaseTriangleAccelerationStructure();
	VkDeviceAddress GetBufferDeviceAddress(VkBuffer buffer) const;
	void CreateAddressBuffer(
		VkDeviceSize size,
		VkBufferUsageFlags usage,
		VkMemoryPropertyFlags properties,
		VkBuffer& buffer,
		VkDeviceMemory& memory) const;
	void BuildTriangleBvh();
	void ApplyModelTransform();

	// What the surface area heuristic concluded about one range of spheres.
	// NoCandidate is not a failure: it means every centroid fell on the same
	// plane, so no cut can separate them and the median split has to take over.
	enum class SahSplitResult
	{
		Split,
		Leaf,
		NoCandidate
	};

	SahSplitResult FindSahSplit(
		uint32_t start,
		uint32_t count,
		float parentSurfaceArea,
		uint32_t& splitAxis,
		uint32_t& leftCount);
	void MedianSplit(
		uint32_t start,
		uint32_t count,
		const glm::vec3& centroidExtent,
		uint32_t& splitAxis,
		uint32_t& leftCount);
	void BuildBvh();
	void CreateComputeDescriptors();
	void UpdateTextureDescriptor();
	void UpdateEnvironmentDescriptor();
	void CreateComputePipeline(const std::string& shaderPath);
	void CreateTimestampQueryPool();
	void ReadGpuComputeTime();
	void Release();

private:
	static constexpr uint32_t TimestampQueryCount = 2;

	uint32_t m_Width = 0;
	uint32_t m_Height = 0;
	uint32_t m_FrameIndex = 0;
	Camera m_Camera;
	// Derived from the camera yaw and pitch, cached so the basis is only
	// recomputed when the camera actually changes.
	glm::vec3 m_CameraForward = { 0.0f, 0.0f, -1.0f };
	// Owned here rather than in the UI for the same reason the camera is: a scene
	// load changes it, and a second copy in the UI would go stale.
	float m_Exposure = 1.0f;
	uint32_t m_BounceCount = 3;
	std::vector<Sphere> m_Spheres;
	std::vector<Triangle> m_Triangles;
	std::vector<Triangle> m_ModelTriangles;
	std::string m_ModelPath;
	std::unique_ptr<Walnut::Image> m_TextureImage;
	std::unique_ptr<Walnut::Image> m_EnvironmentImage;
	bool m_HasEnvironmentMap = false;
	float m_EnvironmentIntensity = 1.0f;
	float m_EnvironmentRotation = 0.0f;
	ModelTransform m_ModelTransform;
	std::vector<uint32_t> m_TriangleOrder;
	uint32_t m_TriangleBvhNodeCount = 0;
	uint32_t m_TriangleBvhDepth = 0;
	std::vector<SphereLight> m_Lights;

	bool m_UseBvh = true;
	bool m_RayQuerySupported = false;
	bool m_UseRayQuery = false;
	// Sampling one light per hit keeps the shadow ray cost independent of the
	// light count, at the price of extra noise that accumulation removes. The
	// light is chosen in proportion to its weight, which cut the noise by up to
	// 5.97 times against choosing uniformly and never did worse: with equal
	// weights the two agree exactly, because equal weights make equal slices.
	//
	// Whether one light beats summing all of them still depends on the scene. At
	// 8 lights, correcting for the noise the reference itself carries, one light
	// needs 1.9 times more frames when a single light dominates a hundred to one
	// but 48 times more when all eight are equal, against roughly 3 times cheaper
	// frames either way. So it wins clearly on the lopsided scene and loses on the
	// even one, which is why it stays available but off by default. It is also
	// what lets the light count grow past this cap at a fixed cost per frame.
	bool m_UseStochasticLights = false;
	// The surface area heuristic splits where the two child boxes are cheapest to
	// trace rather than where the sphere count is even, so it follows how the
	// spheres are actually spread out instead of assuming they are uniform. The
	// median split stays reachable from the UI because it is the thing the
	// heuristic has to beat, and on an evenly spread scene the two nearly tie.
	bool m_UseSahSplit = true;
	// Spheres are uploaded in BVH leaf order, so a leaf can point at a
	// contiguous range of the GPU buffer instead of an indirection list.
	std::vector<uint32_t> m_SphereOrder;
	uint32_t m_BvhNodeCount = 0;
	uint32_t m_BvhDepth = 0;
	float m_BvhCost = 0.0f;
	float m_BvhBuildTimeMs = 0.0f;

	VkBuffer m_SphereBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_SphereBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_BvhBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_BvhBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_LightBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_LightBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_TriangleBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_TriangleBufferMemory = VK_NULL_HANDLE;
	VkBuffer m_TriangleBvhBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_TriangleBvhBufferMemory = VK_NULL_HANDLE;

	VkImage m_OutputImage = VK_NULL_HANDLE;
	VkDeviceMemory m_OutputImageMemory = VK_NULL_HANDLE;
	VkImageView m_OutputImageView = VK_NULL_HANDLE;

	VkImage m_AccumulationImage = VK_NULL_HANDLE;
	VkDeviceMemory m_AccumulationImageMemory = VK_NULL_HANDLE;
	VkImageView m_AccumulationImageView = VK_NULL_HANDLE;

	VkSampler m_OutputSampler = VK_NULL_HANDLE;
	VkDescriptorSet m_ImGuiDescriptorSet = VK_NULL_HANDLE;

	VkDescriptorSetLayout m_ComputeDescriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool m_ComputeDescriptorPool = VK_NULL_HANDLE;
	VkDescriptorSet m_ComputeDescriptorSet = VK_NULL_HANDLE;
	VkBuffer m_RayQueryVertexBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_RayQueryVertexMemory = VK_NULL_HANDLE;
	VkBuffer m_BlasBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_BlasMemory = VK_NULL_HANDLE;
	VkAccelerationStructureKHR m_Blas = VK_NULL_HANDLE;
	VkBuffer m_TlasBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_TlasMemory = VK_NULL_HANDLE;
	VkAccelerationStructureKHR m_Tlas = VK_NULL_HANDLE;
	VkBuffer m_TlasInstanceBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_TlasInstanceMemory = VK_NULL_HANDLE;

	VkPipelineLayout m_ComputePipelineLayout = VK_NULL_HANDLE;
	VkPipeline m_ComputePipeline = VK_NULL_HANDLE;

	VkQueryPool m_TimestampQueryPool = VK_NULL_HANDLE;
	uint64_t m_TimestampValidMask = 0;
	float m_TimestampPeriodNs = 0.0f;
	float m_CpuRenderTimeMs = 0.0f;
	float m_GpuComputeTimeMs = 0.0f;
};
