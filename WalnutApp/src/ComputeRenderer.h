#pragma once

#include <cstdint>
#include <string>

#include <vulkan/vulkan.h>

class ComputeRenderer
{
public:
	~ComputeRenderer();

	void Init(const std::string& shaderPath, uint32_t width, uint32_t height);
	void Render();

	VkDescriptorSet GetImageDescriptorSet() const { return m_ImGuiDescriptorSet; }
	uint32_t GetWidth() const { return m_Width; }
	uint32_t GetHeight() const { return m_Height; }

private:
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
	void CreateOutputImage();
	void CreateComputeDescriptors();
	void CreateComputePipeline(const std::string& shaderPath);
	void Release();

private:
	uint32_t m_Width = 0;
	uint32_t m_Height = 0;

	VkImage m_OutputImage = VK_NULL_HANDLE;
	VkDeviceMemory m_OutputImageMemory = VK_NULL_HANDLE;
	VkImageView m_OutputImageView = VK_NULL_HANDLE;
	VkSampler m_OutputSampler = VK_NULL_HANDLE;
	VkDescriptorSet m_ImGuiDescriptorSet = VK_NULL_HANDLE;

	VkDescriptorSetLayout m_ComputeDescriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorPool m_ComputeDescriptorPool = VK_NULL_HANDLE;
	VkDescriptorSet m_ComputeDescriptorSet = VK_NULL_HANDLE;

	VkPipelineLayout m_ComputePipelineLayout = VK_NULL_HANDLE;
	VkPipeline m_ComputePipeline = VK_NULL_HANDLE;
};
