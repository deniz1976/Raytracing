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

	static constexpr uint32_t MaxSphereCount = 64;

	~ComputeRenderer();

	void Init(const std::string& shaderPath, uint32_t width, uint32_t height);
	void Render();
	void Resize(uint32_t width, uint32_t height);
	void SetCamera(
		const glm::vec3& position,
		const glm::vec3& forward,
		float verticalFov);
	void SetExposure(float exposure) { m_Exposure = exposure; }
	const Sphere& GetSphere(uint32_t index) const;
	void SetSphere(uint32_t index, const Sphere& sphere);
	bool AddSphere();
	bool RemoveSphere(uint32_t index);
	bool SaveScene(const std::string& path, std::string& errorMessage) const;
	bool LoadScene(const std::string& path, std::string& errorMessage);
	const AreaLight& GetAreaLight() const { return m_AreaLight; }
	void SetAreaLight(const AreaLight& light);
	void ResetAccumulation();

	VkDescriptorSet GetImageDescriptorSet() const { return m_ImGuiDescriptorSet; }
	uint32_t GetWidth() const { return m_Width; }
	uint32_t GetHeight() const { return m_Height; }
	uint32_t GetFrameIndex() const { return m_FrameIndex; }
	uint32_t GetSphereCount() const { return static_cast<uint32_t>(m_Spheres.size()); }

private:
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
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
	void UploadSceneBuffer();
	void CreateComputeDescriptors();
	void CreateComputePipeline(const std::string& shaderPath);
	void Release();

private:
	uint32_t m_Width = 0;
	uint32_t m_Height = 0;
	uint32_t m_FrameIndex = 0;
	glm::vec3 m_CameraPosition = { 0.0f, 0.0f, 3.0f };
	glm::vec3 m_CameraForward = { 0.0f, 0.0f, -1.0f };
	float m_VerticalFov = 45.0f;
	float m_Exposure = 1.0f;
	std::vector<Sphere> m_Spheres;
	AreaLight m_AreaLight = {
		{ -2.5f, 5.0f, 2.0f },
		{ 1.0f, 0.95f, 0.85f },
		{ 3.0f, 3.0f },
		24.0f
	};

	VkBuffer m_SphereBuffer = VK_NULL_HANDLE;
	VkDeviceMemory m_SphereBufferMemory = VK_NULL_HANDLE;

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
};
