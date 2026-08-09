#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

class ComputeRenderer
{
public:
	struct Sphere
	{
		glm::vec3 Center;
		float Radius;
		glm::vec3 Albedo;
		float Reflectivity;
		float Roughness;
	};

	struct AreaLight
	{
		glm::vec3 Position;
		glm::vec3 Color;
		glm::vec2 Size;
		float Intensity;
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

	// A binary tree over N primitives needs at most 2N-1 nodes, so this capacity
	// can always hold a tree built over MaxSphereCount spheres.
	static constexpr uint32_t MaxBvhNodeCount = 2 * MaxSphereCount;

	// Every light costs one shadow ray per bounce, so this capacity is kept far
	// smaller than the sphere capacity.
	static constexpr uint32_t MaxLightCount = 8;

	~ComputeRenderer();

	void Init(const std::string& shaderPath, uint32_t width, uint32_t height);
	void Render();
	void Resize(uint32_t width, uint32_t height);
	const Camera& GetCamera() const { return m_Camera; }
	void SetCamera(const Camera& camera);
	const glm::vec3& GetCameraForward() const { return m_CameraForward; }
	void SetExposure(float exposure) { m_Exposure = exposure; }
	const Sphere& GetSphere(uint32_t index) const;
	void SetSphere(uint32_t index, const Sphere& sphere);
	bool AddSphere();
	bool RemoveSphere(uint32_t index);
	bool SaveScene(const std::string& path, std::string& errorMessage) const;
	bool LoadScene(const std::string& path, std::string& errorMessage);
	const AreaLight& GetLight(uint32_t index) const;
	void SetLight(uint32_t index, const AreaLight& light);
	bool AddLight();
	bool RemoveLight(uint32_t index);
	void SetBvhEnabled(bool enabled) { m_UseBvh = enabled; }
	bool IsBvhEnabled() const { return m_UseBvh; }
	void ResetAccumulation();

	VkDescriptorSet GetImageDescriptorSet() const { return m_ImGuiDescriptorSet; }
	uint32_t GetWidth() const { return m_Width; }
	uint32_t GetHeight() const { return m_Height; }
	uint32_t GetFrameIndex() const { return m_FrameIndex; }
	uint32_t GetSphereCount() const { return static_cast<uint32_t>(m_Spheres.size()); }
	uint32_t GetLightCount() const { return static_cast<uint32_t>(m_Lights.size()); }
	uint32_t GetBvhNodeCount() const { return m_BvhNodeCount; }
	uint32_t GetBvhDepth() const { return m_BvhDepth; }
	float GetCpuRenderTimeMs() const { return m_CpuRenderTimeMs; }
	float GetGpuComputeTimeMs() const { return m_GpuComputeTimeMs; }
	bool AreGpuTimestampsSupported() const { return m_TimestampQueryPool != VK_NULL_HANDLE; }

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
	void BuildBvh();
	void CreateComputeDescriptors();
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
	float m_Exposure = 1.0f;
	std::vector<Sphere> m_Spheres;
	std::vector<AreaLight> m_Lights;

	bool m_UseBvh = true;
	// Spheres are uploaded in BVH leaf order, so a leaf can point at a
	// contiguous range of the GPU buffer instead of an indirection list.
	std::vector<uint32_t> m_SphereOrder;
	uint32_t m_BvhNodeCount = 0;
	uint32_t m_BvhDepth = 0;

	VkBuffer m_SphereBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_SphereBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_BvhBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_BvhBufferMemory = VK_NULL_HANDLE;

	VkBuffer m_LightBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_LightBufferMemory = VK_NULL_HANDLE;

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

	VkPipelineLayout m_ComputePipelineLayout = VK_NULL_HANDLE;
	VkPipeline m_ComputePipeline = VK_NULL_HANDLE;

	VkQueryPool m_TimestampQueryPool = VK_NULL_HANDLE;
	uint64_t m_TimestampValidMask = 0;
	float m_TimestampPeriodNs = 0.0f;
	float m_CpuRenderTimeMs = 0.0f;
	float m_GpuComputeTimeMs = 0.0f;
};
