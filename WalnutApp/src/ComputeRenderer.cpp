#include "ComputeRenderer.h"

#include "Walnut/Application.h"

#include "backends/imgui_impl_vulkan.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace
{
	struct alignas(16) PushConstants
	{
		glm::vec4 CameraPosition;
		glm::vec4 CameraForward;
		uint32_t FrameIndex;
		float VerticalFov;
		float Exposure;
		uint32_t SphereCount;
	};

	static_assert(sizeof(PushConstants) == 48);

	struct alignas(16) GpuSphere
	{
		glm::vec4 CenterRadius;
		glm::vec4 AlbedoReflectivity;
		glm::vec4 RoughnessPadding;
	};

	static_assert(sizeof(GpuSphere) == 48);

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

	CreateOutputImages();
	CreateSceneBuffer();
	CreateComputeDescriptors();
	CreateComputePipeline(shaderPath);
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

	const VkDeviceSize bufferSize = sizeof(GpuSphere) * m_Spheres.size();
	VkDevice device = Walnut::Application::GetDevice();

	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = bufferSize;
	bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	check_vk_result(vkCreateBuffer(
		device, &bufferInfo, nullptr, &m_SphereBuffer));

	VkMemoryRequirements memoryRequirements;
	vkGetBufferMemoryRequirements(
		device, m_SphereBuffer, &memoryRequirements);

	VkMemoryAllocateInfo allocationInfo{};
	allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocationInfo.allocationSize = memoryRequirements.size;
	allocationInfo.memoryTypeIndex = FindMemoryType(
		memoryRequirements.memoryTypeBits,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
			VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	check_vk_result(vkAllocateMemory(
		device, &allocationInfo, nullptr, &m_SphereBufferMemory));
	check_vk_result(vkBindBufferMemory(
		device, m_SphereBuffer, m_SphereBufferMemory, 0));

	UploadSceneBuffer();
}

void ComputeRenderer::UploadSceneBuffer()
{
	std::array<GpuSphere, SphereCount> gpuSpheres{};
	for (size_t sphereIndex = 0; sphereIndex < m_Spheres.size(); sphereIndex++)
	{
		const Sphere& sphere = m_Spheres[sphereIndex];
		gpuSpheres[sphereIndex].CenterRadius =
			glm::vec4(sphere.Center, sphere.Radius);
		gpuSpheres[sphereIndex].AlbedoReflectivity =
			glm::vec4(sphere.Albedo, sphere.Reflectivity);
		gpuSpheres[sphereIndex].RoughnessPadding =
			glm::vec4(sphere.Roughness, 0.0f, 0.0f, 0.0f);
	}

	const VkDeviceSize bufferSize = sizeof(gpuSpheres);
	VkDevice device = Walnut::Application::GetDevice();

	void* mappedMemory = nullptr;
	check_vk_result(vkMapMemory(
		device,
		m_SphereBufferMemory,
		0,
		bufferSize,
		0,
		&mappedMemory));
	std::memcpy(
		mappedMemory,
		gpuSpheres.data(),
		static_cast<size_t>(bufferSize));
	vkUnmapMemory(device, m_SphereBufferMemory);
}

const ComputeRenderer::Sphere& ComputeRenderer::GetSphere(uint32_t index) const
{
	return m_Spheres.at(index);
}

void ComputeRenderer::SetSphere(uint32_t index, const Sphere& sphere)
{
	Sphere sanitizedSphere = sphere;
	sanitizedSphere.Radius = std::clamp(sanitizedSphere.Radius, 0.05f, 200.0f);
	sanitizedSphere.Albedo = glm::clamp(
		sanitizedSphere.Albedo,
		glm::vec3(0.0f),
		glm::vec3(1.0f));
	sanitizedSphere.Reflectivity =
		std::clamp(sanitizedSphere.Reflectivity, 0.0f, 1.0f);
	sanitizedSphere.Roughness =
		std::clamp(sanitizedSphere.Roughness, 0.0f, 1.0f);

	m_Spheres.at(index) = sanitizedSphere;
	UploadSceneBuffer();
	ResetAccumulation();
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

void ComputeRenderer::SetCamera(
	const glm::vec3& position,
	const glm::vec3& forward,
	float verticalFov)
{
	m_CameraPosition = position;
	m_CameraForward = glm::normalize(forward);
	m_VerticalFov = verticalFov;
	ResetAccumulation();
}

void ComputeRenderer::ResetAccumulation()
{
	m_FrameIndex = 0;
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

	m_ImGuiDescriptorSet = ImGui_ImplVulkan_AddTexture(
		m_OutputSampler,
		m_OutputImageView,
		VK_IMAGE_LAYOUT_GENERAL);
}

void ComputeRenderer::CreateComputeDescriptors()
{
	VkDevice device = Walnut::Application::GetDevice();

	VkDescriptorSetLayoutBinding descriptorBindings[3]{};
	for (uint32_t bindingIndex = 0; bindingIndex < 2; bindingIndex++)
	{
		descriptorBindings[bindingIndex].binding = bindingIndex;
		descriptorBindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorBindings[bindingIndex].descriptorCount = 1;
		descriptorBindings[bindingIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	}
	descriptorBindings[2].binding = 2;
	descriptorBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorBindings[2].descriptorCount = 1;
	descriptorBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = 3;
	layoutInfo.pBindings = descriptorBindings;
	check_vk_result(vkCreateDescriptorSetLayout(
		device, &layoutInfo, nullptr, &m_ComputeDescriptorSetLayout));

	VkDescriptorPoolSize poolSizes[2]{};
	poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	poolSizes[0].descriptorCount = 2;
	poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	poolSizes[1].descriptorCount = 1;

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
	VkDescriptorBufferInfo descriptorBuffer{};
	descriptorBuffer.buffer = m_SphereBuffer;
	descriptorBuffer.offset = 0;
	descriptorBuffer.range = VK_WHOLE_SIZE;

	VkWriteDescriptorSet descriptorWrites[3]{};
	for (uint32_t bindingIndex = 0; bindingIndex < 2; bindingIndex++)
	{
		descriptorWrites[bindingIndex].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrites[bindingIndex].dstSet = m_ComputeDescriptorSet;
		descriptorWrites[bindingIndex].dstBinding = bindingIndex;
		descriptorWrites[bindingIndex].descriptorCount = 1;
		descriptorWrites[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		descriptorWrites[bindingIndex].pImageInfo = &descriptorImages[bindingIndex];
	}
	descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrites[2].dstSet = m_ComputeDescriptorSet;
	descriptorWrites[2].dstBinding = 2;
	descriptorWrites[2].descriptorCount = 1;
	descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorWrites[2].pBufferInfo = &descriptorBuffer;
	vkUpdateDescriptorSets(device, 3, descriptorWrites, 0, nullptr);
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
	m_FrameIndex++;
	PushConstants pushConstants{};
	pushConstants.CameraPosition = glm::vec4(m_CameraPosition, 1.0f);
	pushConstants.CameraForward = glm::vec4(m_CameraForward, 0.0f);
	pushConstants.FrameIndex = m_FrameIndex;
	pushConstants.VerticalFov = m_VerticalFov;
	pushConstants.Exposure = m_Exposure;
	pushConstants.SphereCount = SphereCount;

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

	vkCmdPushConstants(
		commandBuffer,
		m_ComputePipelineLayout,
		VK_SHADER_STAGE_COMPUTE_BIT,
		0,
		sizeof(PushConstants),
		&pushConstants);

	const uint32_t groupCountX = (m_Width + 7) / 8;
	const uint32_t groupCountY = (m_Height + 7) / 8;
	vkCmdDispatch(commandBuffer, groupCountX, groupCountY, 1);

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

	Walnut::Application::SubmitResourceFree(
		[pipeline, pipelineLayout, descriptorPool, descriptorSetLayout,
		 sampler, outputImageView, outputImage, outputImageMemory,
		 accumulationImageView, accumulationImage, accumulationImageMemory,
		 sphereBuffer, sphereBufferMemory]()
		{
			VkDevice device = Walnut::Application::GetDevice();
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
		});

	m_OutputImage = VK_NULL_HANDLE;
}
