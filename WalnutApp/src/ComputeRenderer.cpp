#include "ComputeRenderer.h"

#include "Walnut/Application.h"

#include "backends/imgui_impl_vulkan.h"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace
{
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

	CreateOutputImage();
	CreateComputeDescriptors();
	CreateComputePipeline(shaderPath);
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

void ComputeRenderer::CreateOutputImage()
{
	VkDevice device = Walnut::Application::GetDevice();

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	imageInfo.extent = { m_Width, m_Height, 1 };
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	check_vk_result(vkCreateImage(device, &imageInfo, nullptr, &m_OutputImage));

	VkMemoryRequirements memoryRequirements;
	vkGetImageMemoryRequirements(device, m_OutputImage, &memoryRequirements);

	VkMemoryAllocateInfo allocationInfo{};
	allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocationInfo.allocationSize = memoryRequirements.size;
	allocationInfo.memoryTypeIndex = FindMemoryType(
		memoryRequirements.memoryTypeBits,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	check_vk_result(vkAllocateMemory(device, &allocationInfo, nullptr, &m_OutputImageMemory));
	check_vk_result(vkBindImageMemory(device, m_OutputImage, m_OutputImageMemory, 0));

	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = m_OutputImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.layerCount = 1;
	check_vk_result(vkCreateImageView(device, &viewInfo, nullptr, &m_OutputImageView));

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerInfo.maxLod = 1.0f;
	check_vk_result(vkCreateSampler(device, &samplerInfo, nullptr, &m_OutputSampler));

	VkCommandBuffer commandBuffer = Walnut::Application::GetCommandBuffer(true);

	VkImageMemoryBarrier layoutBarrier{};
	layoutBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	layoutBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	layoutBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	layoutBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	layoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	layoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	layoutBarrier.image = m_OutputImage;
	layoutBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	layoutBarrier.subresourceRange.levelCount = 1;
	layoutBarrier.subresourceRange.layerCount = 1;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		1, &layoutBarrier);

	Walnut::Application::FlushCommandBuffer(commandBuffer);

	m_ImGuiDescriptorSet = ImGui_ImplVulkan_AddTexture(
		m_OutputSampler,
		m_OutputImageView,
		VK_IMAGE_LAYOUT_GENERAL);
}

void ComputeRenderer::CreateComputeDescriptors()
{
	VkDevice device = Walnut::Application::GetDevice();

	VkDescriptorSetLayoutBinding imageBinding{};
	imageBinding.binding = 0;
	imageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	imageBinding.descriptorCount = 1;
	imageBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &imageBinding;
	check_vk_result(vkCreateDescriptorSetLayout(
		device, &layoutInfo, nullptr, &m_ComputeDescriptorSetLayout));

	VkDescriptorPoolSize poolSize{};
	poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSize.descriptorCount = 1;

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1;
	poolInfo.poolSizeCount = 1;
	poolInfo.pPoolSizes = &poolSize;
	check_vk_result(vkCreateDescriptorPool(
		device, &poolInfo, nullptr, &m_ComputeDescriptorPool));

	VkDescriptorSetAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocateInfo.descriptorPool = m_ComputeDescriptorPool;
	allocateInfo.descriptorSetCount = 1;
	allocateInfo.pSetLayouts = &m_ComputeDescriptorSetLayout;
	check_vk_result(vkAllocateDescriptorSets(
		device, &allocateInfo, &m_ComputeDescriptorSet));

	VkDescriptorImageInfo descriptorImage{};
	descriptorImage.imageView = m_OutputImageView;
	descriptorImage.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet descriptorWrite{};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = m_ComputeDescriptorSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	descriptorWrite.pImageInfo = &descriptorImage;
	vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
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

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &m_ComputeDescriptorSetLayout;
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
	VkCommandBuffer commandBuffer = Walnut::Application::GetCommandBuffer(true);

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

	const uint32_t groupCountX = (m_Width + 7) / 8;
	const uint32_t groupCountY = (m_Height + 7) / 8;
	vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

	VkImageMemoryBarrier computeToImGuiBarrier{};
	computeToImGuiBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	computeToImGuiBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	computeToImGuiBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	computeToImGuiBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
	computeToImGuiBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	computeToImGuiBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	computeToImGuiBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	computeToImGuiBarrier.image = m_OutputImage;
	computeToImGuiBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	computeToImGuiBarrier.subresourceRange.levelCount = 1;
	computeToImGuiBarrier.subresourceRange.layerCount = 1;

	vkCmdPipelineBarrier(
		commandBuffer,
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		0,
		0, nullptr,
		0, nullptr,
		1, &computeToImGuiBarrier);

	Walnut::Application::FlushCommandBuffer(commandBuffer);
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
	VkImageView imageView = m_OutputImageView;
	VkImage image = m_OutputImage;
	VkDeviceMemory imageMemory = m_OutputImageMemory;

	Walnut::Application::SubmitResourceFree(
		[pipeline, pipelineLayout, descriptorPool, descriptorSetLayout,
		 sampler, imageView, image, imageMemory]()
		{
			VkDevice device = Walnut::Application::GetDevice();
			vkDestroyPipeline(device, pipeline, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyDescriptorPool(device, descriptorPool, nullptr);
			vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
			vkDestroySampler(device, sampler, nullptr);
			vkDestroyImageView(device, imageView, nullptr);
			vkDestroyImage(device, image, nullptr);
			vkFreeMemory(device, imageMemory, nullptr);
		});

	m_OutputImage = VK_NULL_HANDLE;
}
