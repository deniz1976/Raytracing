#pragma once

#include "Renderer/TriangleAccelerationStructure.h"
#include "Renderer/VulkanUtils.h"
#include "Scene/SceneFile.h"
#include "Scene/SceneTypes.h"

#include "Walnut/Image.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

class ComputeRenderer
{
public:
	using MaterialType = RayScene::MaterialType;
	using Sphere = RayScene::Sphere;
	using SphereLight = RayScene::SphereLight;
	using Triangle = RayScene::Triangle;
	using ModelTransform = RayScene::ModelTransform;
	using Camera = RayScene::Camera;

	static constexpr uint32_t MaxSphereCount = RayScene::MaxSphereCount;
	static constexpr uint32_t MaxTriangleCount = RayScene::MaxTriangleCount;
	static constexpr uint32_t MaxLightCount = RayScene::MaxLightCount;
	static constexpr uint32_t MaxEnvironmentTexelCount = RayScene::MaxEnvironmentTexelCount;
	static constexpr float MinExposure = RayScene::MinExposure;
	static constexpr float MaxExposure = RayScene::MaxExposure;
	static constexpr uint32_t MinBounceCount = RayScene::MinBounceCount;
	static constexpr uint32_t MaxBounceCount = RayScene::MaxBounceCount;
	static constexpr float MaxEnvironmentIntensity = RayScene::MaxEnvironmentIntensity;
	static constexpr float MinModelScale = RayScene::MinModelScale;
	static constexpr float MaxModelScale = RayScene::MaxModelScale;

	ComputeRenderer() = default;
	ComputeRenderer(const ComputeRenderer&) = delete;
	ComputeRenderer& operator=(const ComputeRenderer&) = delete;
	ComputeRenderer(ComputeRenderer&&) = delete;
	ComputeRenderer& operator=(ComputeRenderer&&) = delete;
	~ComputeRenderer();

	// Loads RayTracing.comp.spv from shaderDirectory when the device supports ray
	// queries, and the RayTracingFallback.comp.spv variant otherwise. Throws a
	// std::runtime_error naming the problem when the renderer cannot start.
	void Init(const std::string& shaderDirectory, uint32_t width, uint32_t height);
	bool IsInitialized() const { return m_ComputePipeline != VK_NULL_HANDLE; }
	void Render();
	void Resize(uint32_t width, uint32_t height);

	const Camera& GetCamera() const { return m_Camera; }
	void SetCamera(const Camera& camera);
	const glm::vec3& GetCameraForward() const { return m_CameraForward; }
	float GetExposure() const { return m_Exposure; }
	void SetExposure(float exposure);
	uint32_t GetBounceCount() const { return m_BounceCount; }
	void SetBounceCount(uint32_t bounceCount);

	const Sphere& GetSphere(uint32_t index) const;
	void SetSphere(uint32_t index, const Sphere& sphere);
	bool AddSphere();
	bool RemoveSphere(uint32_t index);
	const SphereLight& GetLight(uint32_t index) const;
	void SetLight(uint32_t index, const SphereLight& light);
	bool AddLight();
	bool RemoveLight(uint32_t index);

	// Loading a scene is all or nothing: the file, its model and its environment
	// map are all read and validated before any of them replaces the current one.
	bool SaveScene(const std::string& path, std::string& errorMessage) const;
	bool LoadScene(const std::string& path, std::string& errorMessage);
	bool LoadObj(const std::string& path, std::string& errorMessage);
	// Replaces the model with the built-in triangle.
	void ClearModel();
	bool LoadEnvironmentMap(const std::string& path, std::string& errorMessage);
	void ClearEnvironmentMap();

	float GetEnvironmentIntensity() const { return m_EnvironmentIntensity; }
	void SetEnvironmentIntensity(float intensity);
	float GetEnvironmentRotation() const { return m_EnvironmentRotation; }
	void SetEnvironmentRotation(float rotationDegrees);
	const std::string& GetEnvironmentPath() const { return m_EnvironmentPath; }
	const std::string& GetModelPath() const { return m_ModelPath; }
	const ModelTransform& GetModelTransform() const { return m_ModelTransform; }
	// Only the transform uploaded to the GPU changes; the triangles, their BVH
	// and the bottom level acceleration structure stay in model space.
	void SetModelTransform(const ModelTransform& transform);

	// The tree only changes how primitives are searched, never the image.
	void SetBvhEnabled(bool enabled) { m_Settings.UseBvh = enabled; }
	bool IsBvhEnabled() const { return m_Settings.UseBvh; }
	// Which split the trees are built with changes how many boxes and primitives
	// a ray has to test, never which one ends up nearest, so it costs a rebuild
	// but leaves the accumulated samples meaningful.
	void SetSahSplitEnabled(bool enabled);
	bool IsSahSplitEnabled() const { return m_Settings.UseSahSplit; }
	// Changing the sampling strategy changes what every sample means, so the
	// already accumulated frames cannot be mixed with the new ones.
	//
	// Sampling one light per hit keeps the shadow ray cost independent of the
	// light count, at the price of extra noise that accumulation removes. The
	// light is chosen in proportion to its weight, which cut the noise by up to
	// 5.97 times against choosing uniformly. Whether one light beats summing all
	// of them depends on the scene: at 8 lights it needs 1.9 times more frames
	// when one light dominates a hundred to one but 48 times more when all eight
	// are equal, against roughly 3 times cheaper frames either way, which is why
	// it stays available but off by default.
	void SetStochasticLightsEnabled(bool enabled);
	bool AreStochasticLightsEnabled() const { return m_Settings.UseStochasticLights; }
	bool IsRayQuerySupported() const { return m_RayQuerySupported; }
	bool IsRayQueryEnabled() const { return m_Settings.UseRayQuery; }
	void SetRayQueryEnabled(bool enabled);
	void ResetAccumulation();

	VkDescriptorSet GetImageDescriptorSet() const { return m_ImGuiDescriptorSet; }
	uint32_t GetWidth() const { return m_Width; }
	uint32_t GetHeight() const { return m_Height; }
	uint32_t GetFrameIndex() const { return m_FrameIndex; }
	uint32_t GetSphereCount() const { return static_cast<uint32_t>(m_Spheres.size()); }
	uint32_t GetTriangleCount() const { return static_cast<uint32_t>(m_ModelTriangles.size()); }
	uint32_t GetLightCount() const { return static_cast<uint32_t>(m_Lights.size()); }
	uint32_t GetBvhNodeCount() const { return m_BvhNodeCount; }
	uint32_t GetBvhDepth() const { return m_BvhDepth; }
	uint32_t GetTriangleBvhNodeCount() const { return m_TriangleBvhNodeCount; }
	uint32_t GetTriangleBvhDepth() const { return m_TriangleBvhDepth; }
	// The expected number of sphere tests one random ray costs, under the same
	// heuristic the split search minimises. It compares two trees over the same
	// scene without waiting for a timing average to settle.
	float GetBvhCost() const { return m_BvhCost; }
	float GetBvhBuildTimeMs() const { return m_BvhBuildTimeMs; }
	float GetCpuRenderTimeMs() const { return m_CpuRenderTimeMs; }
	float GetGpuComputeTimeMs() const { return m_GpuComputeTimeMs; }
	bool AreGpuTimestampsSupported() const { return m_TimestampQueryPool != VK_NULL_HANDLE; }

private:
	// A model or environment that has been fully read and validated but not yet
	// made current. Preparing touches nothing the GPU is using, so a failure at
	// any point leaves the scene exactly as it was.
	struct PreparedModel
	{
		std::string Path;
		std::vector<Triangle> Triangles;
		std::unique_ptr<Walnut::Image> Texture;
	};

	struct PreparedEnvironment
	{
		std::string Path;
		std::unique_ptr<Walnut::Image> Image;
		std::vector<glm::vec4> Distribution;
		uint32_t Width = 0;
		uint32_t Height = 0;
	};

	static bool PrepareModel(
		const std::string& path,
		PreparedModel& model,
		std::string& errorMessage);
	static PreparedModel PrepareDefaultModel();
	void CommitModel(PreparedModel&& model);
	static bool PrepareEnvironment(
		const std::string& path,
		PreparedEnvironment& environment,
		std::string& errorMessage);
	static PreparedEnvironment PrepareEmptyEnvironment();
	void CommitEnvironment(PreparedEnvironment&& environment);

	RayScene::SceneDescription CaptureScene() const;
	void UpdateCameraBasis();

	void CreateSceneBuffers();
	void UploadSpheres();
	void UploadLights();
	void UploadModel();
	void UploadModelTransform();

	void CreateImage(
		VkFormat format,
		VkImageUsageFlags usage,
		VkImage& image,
		VkDeviceMemory& memory,
		VkImageView& imageView);
	void CreateOutputImages();
	void UpdateImGuiImageDescriptor();

	void CreateComputeDescriptors();
	void WriteStorageImageDescriptor(uint32_t binding, VkImageView view);
	void WriteStorageBufferDescriptor(uint32_t binding, const VulkanUtils::Buffer& buffer);
	void WriteSampledImageDescriptor(uint32_t binding, const Walnut::Image& image);
	void WriteTlasDescriptor();
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
	// Owned here rather than in the UI for the same reason the camera is: a scene
	// load changes it, and a second copy in the UI would go stale.
	float m_Exposure = 1.0f;
	uint32_t m_BounceCount = 3;
	RayScene::RenderSettings m_Settings;
	bool m_RayQuerySupported = false;

	std::vector<Sphere> m_Spheres;
	std::vector<SphereLight> m_Lights;
	// Kept in model space; see SetModelTransform.
	std::vector<Triangle> m_ModelTriangles;
	std::string m_ModelPath;
	ModelTransform m_ModelTransform;
	std::unique_ptr<Walnut::Image> m_TextureImage;
	std::unique_ptr<Walnut::Image> m_EnvironmentImage;
	std::string m_EnvironmentPath;
	uint32_t m_EnvironmentWidth = 0;
	uint32_t m_EnvironmentHeight = 0;
	float m_EnvironmentIntensity = 1.0f;
	float m_EnvironmentRotation = 0.0f;

	uint32_t m_BvhNodeCount = 0;
	uint32_t m_BvhDepth = 0;
	float m_BvhCost = 0.0f;
	float m_BvhBuildTimeMs = 0.0f;
	uint32_t m_TriangleBvhNodeCount = 0;
	uint32_t m_TriangleBvhDepth = 0;

	VulkanUtils::Buffer m_SphereBuffer;
	VulkanUtils::Buffer m_SphereBvhBuffer;
	VulkanUtils::Buffer m_LightBuffer;
	VulkanUtils::Buffer m_TriangleBuffer;
	VulkanUtils::Buffer m_TriangleBvhBuffer;
	VulkanUtils::Buffer m_EnvironmentDistributionBuffer;
	VulkanUtils::Buffer m_ModelTransformBuffer;
	TriangleAccelerationStructure m_AccelerationStructure;

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
