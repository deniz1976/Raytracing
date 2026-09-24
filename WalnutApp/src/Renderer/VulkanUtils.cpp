#include "VulkanUtils.h"

#include "Walnut/Application.h"

#include <cstring>
#include <stdexcept>

namespace VulkanUtils
{
	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memoryProperties;
		vkGetPhysicalDeviceMemoryProperties(
			Walnut::Application::GetPhysicalDevice(),
			&memoryProperties);

		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
		{
			const bool typeMatches = (typeFilter & (1u << i)) != 0;
			const bool propertiesMatch =
				(memoryProperties.memoryTypes[i].propertyFlags & properties) == properties;
			if (typeMatches && propertiesMatch)
				return i;
		}

		throw std::runtime_error("A suitable Vulkan memory type could not be found.");
	}

	Buffer CreateBuffer(
		VkDeviceSize size,
		VkBufferUsageFlags usage,
		VkMemoryPropertyFlags properties)
	{
		VkDevice device = Walnut::Application::GetDevice();
		Buffer buffer;
		buffer.Size = size;

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		check_vk_result(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer.Handle));

		VkMemoryRequirements requirements{};
		vkGetBufferMemoryRequirements(device, buffer.Handle, &requirements);

		VkMemoryAllocateFlagsInfo flagsInfo{};
		flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
		flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

		VkMemoryAllocateInfo allocationInfo{};
		allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
			allocationInfo.pNext = &flagsInfo;
		allocationInfo.allocationSize = requirements.size;
		allocationInfo.memoryTypeIndex =
			FindMemoryType(requirements.memoryTypeBits, properties);
		check_vk_result(vkAllocateMemory(
			device, &allocationInfo, nullptr, &buffer.Memory));
		check_vk_result(vkBindBufferMemory(device, buffer.Handle, buffer.Memory, 0));
		return buffer;
	}

	Buffer CreateHostStorageBuffer(VkDeviceSize size)
	{
		return CreateBuffer(
			size,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	}

	void WriteBuffer(const Buffer& buffer, const void* data, VkDeviceSize size)
	{
		if (size == 0)
			return;
		if (size > buffer.Size)
			throw std::runtime_error("Buffer write exceeds the buffer size.");

		VkDevice device = Walnut::Application::GetDevice();
		void* mappedMemory = nullptr;
		check_vk_result(vkMapMemory(device, buffer.Memory, 0, size, 0, &mappedMemory));
		std::memcpy(mappedMemory, data, static_cast<size_t>(size));
		vkUnmapMemory(device, buffer.Memory);
	}

	VkDeviceAddress GetDeviceAddress(const Buffer& buffer)
	{
		VkBufferDeviceAddressInfo addressInfo{};
		addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		addressInfo.buffer = buffer.Handle;
		return vkGetBufferDeviceAddress(
			Walnut::Application::GetDevice(), &addressInfo);
	}

	void DestroyBuffer(Buffer& buffer)
	{
		VkDevice device = Walnut::Application::GetDevice();
		if (buffer.Handle != VK_NULL_HANDLE)
			vkDestroyBuffer(device, buffer.Handle, nullptr);
		if (buffer.Memory != VK_NULL_HANDLE)
			vkFreeMemory(device, buffer.Memory, nullptr);
		buffer = Buffer{};
	}
}
