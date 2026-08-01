#pragma once

#include <cstdint>
#include <string>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

class ComputeRenderer
{
public:
	~ComputeRenderer();

	void Init(const std::string& shaderPath, uint32_t width, uint32_t height);
	void Render();
	void SetCamera(
		const glm::vec3& position,
		const glm::vec3& forward,
		float verticalFov);
	void SetExposure(float exposure) { m_Exposure = exposure; }
	void ResetAccumulation();

	VkDescriptorSet GetImageDescriptorSet() const { return m_ImGuiDescriptorSet; }
	uint32_t GetWidth() const { return m_Width; }
	uint32_t GetHeight() const { return m_Height; }
	uint32_t GetFrameIndex() const { return m_FrameIndex; }

private:
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
	void CreateImage(
		VkFormat format,
		VkImageUsageFlags usage,
		VkImage& image,
		VkDeviceMemory& memory,
		VkImageView& imageView);
	void CreateOutputImages();
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
