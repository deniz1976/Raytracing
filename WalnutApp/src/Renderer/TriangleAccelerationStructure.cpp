#include "Renderer/TriangleAccelerationStructure.h"

#include "Walnut/Application.h"

namespace
{
	template<typename Function>
	Function LoadDeviceFunction(VkDevice device, const char* name)
	{
		return reinterpret_cast<Function>(vkGetDeviceProcAddr(device, name));
	}

	VkAccelerationStructureGeometryKHR InstanceGeometry(VkDeviceAddress instances)
	{
		VkAccelerationStructureGeometryKHR geometry{};
		geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
		geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
		geometry.geometry.instances.sType =
			VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
		geometry.geometry.instances.data.deviceAddress = instances;
		return geometry;
	}

	VkAccelerationStructureBuildGeometryInfoKHR BuildInfo(
		VkAccelerationStructureTypeKHR type,
		const VkAccelerationStructureGeometryKHR* geometry)
	{
		VkAccelerationStructureBuildGeometryInfoKHR info{};
		info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
		info.type = type;
		info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
		info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
		info.geometryCount = 1;
		info.pGeometries = geometry;
		return info;
	}

	VulkanUtils::Buffer CreateDeviceBuffer(VkDeviceSize size, VkBufferUsageFlags usage)
	{
		return VulkanUtils::CreateBuffer(
			size,
			usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	}

	VulkanUtils::Buffer CreateBuildInputBuffer(VkDeviceSize size)
	{
		return VulkanUtils::CreateBuffer(
			size,
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
				VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	}
}

TriangleAccelerationStructure::~TriangleAccelerationStructure()
{
	Release();
}

bool TriangleAccelerationStructure::Init()
{
	VkDevice device = Walnut::Application::GetDevice();
	m_CreateAccelerationStructure =
		LoadDeviceFunction<PFN_vkCreateAccelerationStructureKHR>(
			device, "vkCreateAccelerationStructureKHR");
	m_DestroyAccelerationStructure =
		LoadDeviceFunction<PFN_vkDestroyAccelerationStructureKHR>(
			device, "vkDestroyAccelerationStructureKHR");
	m_CmdBuildAccelerationStructures =
		LoadDeviceFunction<PFN_vkCmdBuildAccelerationStructuresKHR>(
			device, "vkCmdBuildAccelerationStructuresKHR");
	m_GetBuildSizes =
		LoadDeviceFunction<PFN_vkGetAccelerationStructureBuildSizesKHR>(
			device, "vkGetAccelerationStructureBuildSizesKHR");
	m_GetDeviceAddress =
		LoadDeviceFunction<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
			device, "vkGetAccelerationStructureDeviceAddressKHR");
	return m_CreateAccelerationStructure && m_DestroyAccelerationStructure &&
		m_CmdBuildAccelerationStructures && m_GetBuildSizes && m_GetDeviceAddress;
}

VkAccelerationStructureKHR TriangleAccelerationStructure::CreateStructure(
	VkAccelerationStructureTypeKHR type,
	const VulkanUtils::Buffer& storage)
{
	VkAccelerationStructureCreateInfoKHR createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	createInfo.buffer = storage.Handle;
	createInfo.size = storage.Size;
	createInfo.type = type;
	VkAccelerationStructureKHR structure = VK_NULL_HANDLE;
	check_vk_result(m_CreateAccelerationStructure(
		Walnut::Application::GetDevice(), &createInfo, nullptr, &structure));
	return structure;
}

void TriangleAccelerationStructure::Build(
	const std::vector<glm::vec3>& vertices,
	const glm::mat4& objectToWorld)
{
	VkDevice device = Walnut::Application::GetDevice();
	check_vk_result(vkDeviceWaitIdle(device));
	Release();
	if (vertices.empty())
		return;

	m_Vertices = CreateBuildInputBuffer(vertices.size() * sizeof(glm::vec3));
	VulkanUtils::WriteBuffer(
		m_Vertices, vertices.data(), vertices.size() * sizeof(glm::vec3));

	VkAccelerationStructureGeometryKHR geometry{};
	geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geometry.geometry.triangles.sType =
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	geometry.geometry.triangles.vertexData.deviceAddress =
		VulkanUtils::GetDeviceAddress(m_Vertices);
	geometry.geometry.triangles.vertexStride = sizeof(glm::vec3);
	geometry.geometry.triangles.maxVertex = static_cast<uint32_t>(vertices.size() - 1);
	geometry.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;

	const uint32_t primitiveCount = static_cast<uint32_t>(vertices.size() / 3);
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo =
		BuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, &geometry);
	VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
	sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	m_GetBuildSizes(
		device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&buildInfo, &primitiveCount, &sizeInfo);

	m_BlasStorage = CreateDeviceBuffer(
		sizeInfo.accelerationStructureSize,
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
	m_Blas = CreateStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, m_BlasStorage);

	VulkanUtils::Buffer scratch = CreateDeviceBuffer(
		sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	buildInfo.dstAccelerationStructure = m_Blas;
	buildInfo.scratchData.deviceAddress = VulkanUtils::GetDeviceAddress(scratch);
	VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
	rangeInfo.primitiveCount = primitiveCount;
	const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = { &rangeInfo };
	VkCommandBuffer commandBuffer = Walnut::Application::GetCommandBuffer(true);
	m_CmdBuildAccelerationStructures(commandBuffer, 1, &buildInfo, rangeInfos);
	Walnut::Application::FlushCommandBuffer(commandBuffer);
	VulkanUtils::DestroyBuffer(scratch);

	VkAccelerationStructureDeviceAddressInfoKHR blasAddressInfo{};
	blasAddressInfo.sType =
		VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	blasAddressInfo.accelerationStructure = m_Blas;
	m_BlasAddress = m_GetDeviceAddress(device, &blasAddressInfo);

	m_Instances = CreateBuildInputBuffer(sizeof(VkAccelerationStructureInstanceKHR));
	WriteInstance(objectToWorld);

	const VkAccelerationStructureGeometryKHR instanceGeometry =
		InstanceGeometry(VulkanUtils::GetDeviceAddress(m_Instances));
	const VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo =
		BuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &instanceGeometry);
	const uint32_t instanceCount = 1;
	VkAccelerationStructureBuildSizesInfoKHR tlasSizeInfo{};
	tlasSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	m_GetBuildSizes(
		device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&tlasBuildInfo, &instanceCount, &tlasSizeInfo);

	m_TlasStorage = CreateDeviceBuffer(
		tlasSizeInfo.accelerationStructureSize,
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
	m_Tlas = CreateStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, m_TlasStorage);
	m_TlasScratch = CreateDeviceBuffer(
		tlasSizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	BuildTopLevel();
}

void TriangleAccelerationStructure::SetTransform(const glm::mat4& objectToWorld)
{
	if (m_Tlas == VK_NULL_HANDLE)
		return;

	// Every renderer submission waits on its fence before returning, so the GPU
	// is not reading the instance buffer or the TLAS at this point and both can
	// be rewritten in place without a device wide wait.
	WriteInstance(objectToWorld);
	BuildTopLevel();
}

void TriangleAccelerationStructure::WriteInstance(const glm::mat4& objectToWorld)
{
	VkAccelerationStructureInstanceKHR instance{};
	// Vulkan wants the top three rows of the matrix, row major; glm is column major.
	for (int row = 0; row < 3; row++)
	{
		for (int column = 0; column < 4; column++)
			instance.transform.matrix[row][column] = objectToWorld[column][row];
	}
	// Custom index zero makes the shader's primitive index the triangle index.
	instance.instanceCustomIndex = 0;
	instance.mask = 0xff;
	instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
	instance.accelerationStructureReference = m_BlasAddress;
	VulkanUtils::WriteBuffer(m_Instances, &instance, sizeof(instance));
}

void TriangleAccelerationStructure::BuildTopLevel()
{
	const VkAccelerationStructureGeometryKHR instanceGeometry =
		InstanceGeometry(VulkanUtils::GetDeviceAddress(m_Instances));
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo =
		BuildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, &instanceGeometry);
	buildInfo.dstAccelerationStructure = m_Tlas;
	buildInfo.scratchData.deviceAddress = VulkanUtils::GetDeviceAddress(m_TlasScratch);
	VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
	rangeInfo.primitiveCount = 1;
	const VkAccelerationStructureBuildRangeInfoKHR* rangeInfos[] = { &rangeInfo };

	VkCommandBuffer commandBuffer = Walnut::Application::GetCommandBuffer(true);
	m_CmdBuildAccelerationStructures(commandBuffer, 1, &buildInfo, rangeInfos);
	Walnut::Application::FlushCommandBuffer(commandBuffer);
}

void TriangleAccelerationStructure::Release()
{
	if (!m_DestroyAccelerationStructure)
		return;

	VkDevice device = Walnut::Application::GetDevice();
	if (m_Tlas != VK_NULL_HANDLE)
		m_DestroyAccelerationStructure(device, m_Tlas, nullptr);
	if (m_Blas != VK_NULL_HANDLE)
		m_DestroyAccelerationStructure(device, m_Blas, nullptr);
	m_Tlas = VK_NULL_HANDLE;
	m_Blas = VK_NULL_HANDLE;
	m_BlasAddress = 0;
	VulkanUtils::DestroyBuffer(m_TlasScratch);
	VulkanUtils::DestroyBuffer(m_TlasStorage);
	VulkanUtils::DestroyBuffer(m_Instances);
	VulkanUtils::DestroyBuffer(m_BlasStorage);
	VulkanUtils::DestroyBuffer(m_Vertices);
}
