#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

namespace VulkanUtils
{
	struct Buffer
	{
		VkBuffer Handle = VK_NULL_HANDLE;
		VkDeviceMemory Memory = VK_NULL_HANDLE;
		VkDeviceSize Size = 0;
	};

	uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

	// Creates a buffer with dedicated memory. When usage asks for a shader device
	// address, the memory is allocated with the matching flag so the address can
	// be queried.
	Buffer CreateBuffer(
		VkDeviceSize size,
		VkBufferUsageFlags usage,
		VkMemoryPropertyFlags properties);

	// The scene buffers are small and rewritten whenever the scene changes, so they
	// live in host visible, coherent memory and skip the staging buffer dance.
	Buffer CreateHostStorageBuffer(VkDeviceSize size);

	// Copies into host visible memory. Only valid while the GPU is not reading the
	// buffer, which the renderer guarantees by waiting for every dispatch.
	void WriteBuffer(const Buffer& buffer, const void* data, VkDeviceSize size);

	VkDeviceAddress GetDeviceAddress(const Buffer& buffer);

	// Destroys immediately; the caller makes sure the GPU is done with it.
	void DestroyBuffer(Buffer& buffer);
}
