#pragma once

#include "Renderer/VulkanUtils.h"

#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

// The hardware ray query path: one bottom level structure over the model's
// triangles in model space, and a top level structure with a single instance
// that carries the model transform. Moving the model therefore only rebuilds the
// one-instance top level structure, never the triangle geometry.
class TriangleAccelerationStructure
{
public:
	TriangleAccelerationStructure() = default;
	TriangleAccelerationStructure(const TriangleAccelerationStructure&) = delete;
	TriangleAccelerationStructure& operator=(const TriangleAccelerationStructure&) = delete;
	~TriangleAccelerationStructure();

	// Loads the extension entry points. Returns false when the device does not
	// expose them, in which case nothing else may be called.
	bool Init();

	// Builds both levels. vertices holds three corners per triangle, in the same
	// order as the triangle buffer, so a primitive index addresses that buffer.
	void Build(const std::vector<glm::vec3>& vertices, const glm::mat4& objectToWorld);

	// Rewrites the instance transform and rebuilds the top level structure.
	void SetTransform(const glm::mat4& objectToWorld);

	VkAccelerationStructureKHR GetTlas() const { return m_Tlas; }

	// Destroys immediately; the caller makes sure the GPU is idle.
	void Release();

private:
	void WriteInstance(const glm::mat4& objectToWorld);
	void BuildTopLevel();
	VkAccelerationStructureKHR CreateStructure(
		VkAccelerationStructureTypeKHR type,
		const VulkanUtils::Buffer& storage);

private:
	PFN_vkCreateAccelerationStructureKHR m_CreateAccelerationStructure = nullptr;
	PFN_vkDestroyAccelerationStructureKHR m_DestroyAccelerationStructure = nullptr;
	PFN_vkCmdBuildAccelerationStructuresKHR m_CmdBuildAccelerationStructures = nullptr;
	PFN_vkGetAccelerationStructureBuildSizesKHR m_GetBuildSizes = nullptr;
	PFN_vkGetAccelerationStructureDeviceAddressKHR m_GetDeviceAddress = nullptr;

	VulkanUtils::Buffer m_Vertices;
	VulkanUtils::Buffer m_BlasStorage;
	VulkanUtils::Buffer m_Instances;
	VulkanUtils::Buffer m_TlasStorage;
	// Kept alive so a transform change can rebuild the top level structure
	// without allocating anything.
	VulkanUtils::Buffer m_TlasScratch;
	VkAccelerationStructureKHR m_Blas = VK_NULL_HANDLE;
	VkAccelerationStructureKHR m_Tlas = VK_NULL_HANDLE;
	VkDeviceAddress m_BlasAddress = 0;
};
