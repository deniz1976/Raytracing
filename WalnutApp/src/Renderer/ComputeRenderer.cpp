#include "Renderer/ComputeRenderer.h"

#include "Renderer/GpuTypes.h"
#include "Scene/Bvh.h"
#include "Scene/EnvironmentDistribution.h"
#include "Scene/ObjLoader.h"
#include "Shaders/ShaderInterface.h"

#include "Walnut/Application.h"
#include "Walnut/Image.h"

#include "backends/imgui_impl_vulkan.h"

#include "../../vendor/stb_image/stb_image.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace
{
	constexpr const char* RayQueryShaderName = "RayTracing.comp.spv";
	constexpr const char* FallbackShaderName = "RayTracingFallback.comp.spv";

	// The traversal stack holds one pending sibling per level plus the node being
	// expanded, so a tree this deep always fits.
	constexpr uint32_t BvhMaxDepth = 30;
	static_assert(BvhMaxDepth < BVH_STACK_SIZE);

	// A binary tree over N primitives needs at most 2N-1 nodes.
	constexpr uint32_t MaxSphereBvhNodeCount = 2 * RayScene::MaxSphereCount;
	constexpr uint32_t MaxTriangleBvhNodeCount = 2 * RayScene::MaxTriangleCount;

	constexpr float TimingSmoothingFactor = 0.05f;

	// Raw per-frame timings jitter far too much to read in the UI, so every
	// sample is folded into an exponential moving average instead.
	float SmoothTiming(float averageMs, float sampleMs)
	{
		if (averageMs <= 0.0f)
			return sampleMs;

		return averageMs + (sampleMs - averageMs) * TimingSmoothingFactor;
	}

	std::vector<uint32_t> ReadShaderFile(const std::string& path)
	{
		std::ifstream file(path, std::ios::ate | std::ios::binary);
		if (!file)
			return {};

		const size_t fileSize = static_cast<size_t>(file.tellg());
		if (fileSize == 0 || fileSize % sizeof(uint32_t) != 0)
			return {};

		std::vector<uint32_t> code(fileSize / sizeof(uint32_t));
		file.seekg(0);
		if (!file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(fileSize)))
			return {};
		return code;
	}

	std::unique_ptr<Walnut::Image> CreateSolidImage(Walnut::ImageFormat format, const void* pixel)
	{
		return std::make_unique<Walnut::Image>(1, 1, format, pixel);
	}

	std::unique_ptr<Walnut::Image> CreateWhiteTexture()
	{
		const uint32_t whitePixel = 0xffffffffu;
		return CreateSolidImage(Walnut::ImageFormat::RGBA, &whitePixel);
	}

	RayScene::BvhBuildSettings BvhSettings(bool useSah, float boundsPadding)
	{
		RayScene::BvhBuildSettings settings;
		settings.UseSah = useSah;
		settings.MaxDepth = BvhMaxDepth;
		settings.BoundsPadding = boundsPadding;
		return settings;
	}

	GpuTypes::Triangle PackTriangle(const RayScene::Triangle& triangle)
	{
		GpuTypes::Triangle packed;
		packed.Vertex0 = glm::vec4(triangle.Vertex0, 0.0f);
		packed.Vertex1 = glm::vec4(triangle.Vertex1, 0.0f);
		packed.Vertex2 = glm::vec4(triangle.Vertex2, 0.0f);
		packed.AlbedoReflectivity = glm::vec4(triangle.Albedo, triangle.Reflectivity);
		packed.RoughnessMaterial = glm::vec4(
			triangle.Roughness,
			static_cast<float>(triangle.Type),
			triangle.IndexOfRefraction,
			0.0f);
		packed.Normal0 = glm::vec4(triangle.Normal0, triangle.HasVertexNormals ? 1.0f : 0.0f);
		packed.Normal1 = glm::vec4(triangle.Normal1, 0.0f);
		packed.Normal2 = glm::vec4(triangle.Normal2, 0.0f);
		packed.TexCoord01 = glm::vec4(triangle.TexCoord0, triangle.TexCoord1);
		packed.TexCoord2 = glm::vec4(
			triangle.TexCoord2,
			triangle.UsesImageTexture ? 1.0f : 0.0f,
			triangle.HasTexCoords ? 1.0f : 0.0f);
		return packed;
	}

	GpuTypes::Sphere PackSphere(const RayScene::Sphere& sphere)
	{
		GpuTypes::Sphere packed;
		packed.CenterRadius = glm::vec4(sphere.Center, sphere.Radius);
		packed.AlbedoReflectivity = glm::vec4(sphere.Albedo, sphere.Reflectivity);
		packed.RoughnessMaterial = glm::vec4(
			sphere.Roughness,
			static_cast<float>(sphere.Type),
			sphere.IndexOfRefraction,
			0.0f);
		return packed;
	}
}

ComputeRenderer::~ComputeRenderer()
{
	Release();
}

void ComputeRenderer::Init(const std::string& shaderDirectory, uint32_t width, uint32_t height)
{
	// Hardware ray queries are an optional accelerator: without them the custom
	// triangle BVH does the same work in the fallback shader.
	m_RayQuerySupported =
		Walnut::Application::SupportsRayQuery() && m_AccelerationStructure.Init();
	m_Settings.UseRayQuery = false;
	m_Width = width;
	m_Height = height;
	m_FrameIndex = 0;

	// Keeps the cached forward vector consistent with the default angles instead
	// of relying on the two initializers agreeing with each other.
	UpdateCameraBasis();

	CreateOutputImages();
	CreateSceneBuffers();
	m_TextureImage = CreateWhiteTexture();
	m_EnvironmentImage = PrepareEmptyEnvironment().Image;
	CreateComputeDescriptors();

	m_Spheres = RayScene::DefaultSpheres();
	m_Lights = RayScene::DefaultLights();
	UploadSpheres();
	UploadLights();
	CommitModel(PrepareDefaultModel());
	CommitEnvironment(PrepareEmptyEnvironment());

	CreateComputePipeline(
		shaderDirectory + "/" +
		(m_RayQuerySupported ? RayQueryShaderName : FallbackShaderName));
	CreateTimestampQueryPool();
}

// ------------------------------------------------------------ scene buffers

void ComputeRenderer::CreateSceneBuffers()
{
	using VulkanUtils::CreateHostStorageBuffer;
	m_SphereBuffer = CreateHostStorageBuffer(
		sizeof(GpuTypes::Sphere) * MaxSphereCount);
	m_SphereBvhBuffer = CreateHostStorageBuffer(
		sizeof(RayScene::BvhNode) * MaxSphereBvhNodeCount);
	m_LightBuffer = CreateHostStorageBuffer(
		sizeof(GpuTypes::SphereLight) * MaxLightCount);
	m_TriangleBuffer = CreateHostStorageBuffer(
		sizeof(GpuTypes::Triangle) * MaxTriangleCount);
	m_TriangleBvhBuffer = CreateHostStorageBuffer(
		sizeof(RayScene::BvhNode) * MaxTriangleBvhNodeCount);
	m_EnvironmentDistributionBuffer = CreateHostStorageBuffer(
		sizeof(glm::vec4) * MaxEnvironmentTexelCount);
	m_ModelTransformBuffer = CreateHostStorageBuffer(sizeof(GpuTypes::ModelTransform));
}

// Groups the spheres into a binary tree of boxes so a ray can reject a whole
// branch with one box test. The tree decides the order the spheres are stored
// in, so it is rebuilt before the sphere buffer is written.
void ComputeRenderer::UploadSpheres()
{
	const auto buildBegin = std::chrono::steady_clock::now();

	std::vector<RayScene::BvhPrimitive> primitives(m_Spheres.size());
	for (size_t index = 0; index < m_Spheres.size(); index++)
	{
		const Sphere& sphere = m_Spheres[index];
		primitives[index].Bounds.Grow(sphere.Center - glm::vec3(sphere.Radius));
		primitives[index].Bounds.Grow(sphere.Center + glm::vec3(sphere.Radius));
		primitives[index].Centroid = sphere.Center;
	}
	const RayScene::BvhBuildResult tree = RayScene::BuildBvh(
		primitives, BvhSettings(m_Settings.UseSahSplit, 0.0f));

	const std::chrono::duration<float, std::milli> buildDuration =
		std::chrono::steady_clock::now() - buildBegin;
	m_BvhBuildTimeMs = buildDuration.count();
	m_BvhNodeCount = static_cast<uint32_t>(tree.Nodes.size());
	m_BvhDepth = tree.Depth;
	m_BvhCost = tree.Cost;

	std::vector<GpuTypes::Sphere> gpuSpheres;
	gpuSpheres.reserve(m_Spheres.size());
	for (uint32_t sphereIndex : tree.PrimitiveOrder)
		gpuSpheres.push_back(PackSphere(m_Spheres[sphereIndex]));

	VulkanUtils::WriteBuffer(
		m_SphereBuffer, gpuSpheres.data(), gpuSpheres.size() * sizeof(GpuTypes::Sphere));
	VulkanUtils::WriteBuffer(
		m_SphereBvhBuffer, tree.Nodes.data(), tree.Nodes.size() * sizeof(RayScene::BvhNode));
}

void ComputeRenderer::UploadLights()
{
	std::array<GpuTypes::SphereLight, MaxLightCount> gpuLights{};
	for (size_t index = 0; index < m_Lights.size(); index++)
	{
		const SphereLight& light = m_Lights[index];
		gpuLights[index].PositionIntensity = glm::vec4(light.Position, light.Intensity);
		gpuLights[index].Color = glm::vec4(light.Color, 0.0f);
		gpuLights[index].RadiusSampling = glm::vec4(light.Radius, 0.0f, 0.0f, 0.0f);
	}

	// The stochastic path picks one light per hit, and how often it picks each
	// one decides how noisy that single sample is. Choosing in proportion to how
	// much a light can contribute keeps the estimator unbiased while making the
	// large contributors the ones that actually get sampled. Distance, incidence
	// angle and visibility belong to the shading point, so this is a bound on
	// the light rather than its actual contribution.
	float totalWeight = 0.0f;
	std::array<float, MaxLightCount> weights{};
	for (size_t index = 0; index < m_Lights.size(); index++)
	{
		const SphereLight& light = m_Lights[index];
		const float samplingRadius = std::max(light.Radius, 0.01f);
		weights[index] =
			light.Intensity * samplingRadius * samplingRadius *
			(light.Color.r + light.Color.g + light.Color.b);
		totalWeight += weights[index];
	}

	// A zero weight means the light contributes exactly zero, so it gets a zero
	// width slice and is never picked. When every light is like that the slices
	// stay empty, which the shader treats as nothing to sample.
	if (totalWeight > 0.0f)
	{
		float cumulativeWeight = 0.0f;
		for (size_t index = 0; index < m_Lights.size(); index++)
		{
			cumulativeWeight += weights[index];
			gpuLights[index].RadiusSampling.z = cumulativeWeight / totalWeight;
			gpuLights[index].RadiusSampling.w = weights[index] / totalWeight;
		}

		// Rounding can leave the last bound a hair below one, which would let a
		// random number land past every slice.
		gpuLights[m_Lights.size() - 1].RadiusSampling.z = 1.0f;
	}

	VulkanUtils::WriteBuffer(m_LightBuffer, gpuLights.data(), sizeof(gpuLights));
}

// Builds everything that depends on the model's triangles: their BVH, the
// triangle buffer in BVH leaf order and, when available, the hardware
// acceleration structure over the same order. All of it stays in model space.
void ComputeRenderer::UploadModel()
{
	std::vector<RayScene::BvhPrimitive> primitives(m_ModelTriangles.size());
	for (size_t index = 0; index < m_ModelTriangles.size(); index++)
	{
		const Triangle& triangle = m_ModelTriangles[index];
		primitives[index].Bounds.Grow(triangle.Vertex0);
		primitives[index].Bounds.Grow(triangle.Vertex1);
		primitives[index].Bounds.Grow(triangle.Vertex2);
		primitives[index].Centroid =
			(triangle.Vertex0 + triangle.Vertex1 + triangle.Vertex2) / 3.0f;
	}
	// A perfectly flat triangle has a zero-size box on one axis. A tiny
	// expansion makes the slab test conservative around floating-point edges.
	const RayScene::BvhBuildResult tree = RayScene::BuildBvh(
		primitives, BvhSettings(m_Settings.UseSahSplit, 1e-4f));
	m_TriangleBvhNodeCount = static_cast<uint32_t>(tree.Nodes.size());
	m_TriangleBvhDepth = tree.Depth;

	std::vector<GpuTypes::Triangle> gpuTriangles;
	std::vector<glm::vec3> vertices;
	gpuTriangles.reserve(m_ModelTriangles.size());
	vertices.reserve(m_ModelTriangles.size() * 3);
	for (uint32_t triangleIndex : tree.PrimitiveOrder)
	{
		const Triangle& triangle = m_ModelTriangles[triangleIndex];
		gpuTriangles.push_back(PackTriangle(triangle));
		vertices.push_back(triangle.Vertex0);
		vertices.push_back(triangle.Vertex1);
		vertices.push_back(triangle.Vertex2);
	}

	VulkanUtils::WriteBuffer(
		m_TriangleBuffer,
		gpuTriangles.data(),
		gpuTriangles.size() * sizeof(GpuTypes::Triangle));
	VulkanUtils::WriteBuffer(
		m_TriangleBvhBuffer,
		tree.Nodes.data(),
		tree.Nodes.size() * sizeof(RayScene::BvhNode));

	const glm::mat4 modelMatrix = RayScene::ComposeModelMatrix(m_ModelTransform);
	const GpuTypes::ModelTransform gpuTransform{ glm::inverse(modelMatrix) };
	VulkanUtils::WriteBuffer(m_ModelTransformBuffer, &gpuTransform, sizeof(gpuTransform));

	if (m_RayQuerySupported)
	{
		m_AccelerationStructure.Build(vertices, modelMatrix);
		WriteTlasDescriptor();
	}
}

void ComputeRenderer::UploadModelTransform()
{
	const glm::mat4 modelMatrix = RayScene::ComposeModelMatrix(m_ModelTransform);
	const GpuTypes::ModelTransform gpuTransform{ glm::inverse(modelMatrix) };
	VulkanUtils::WriteBuffer(m_ModelTransformBuffer, &gpuTransform, sizeof(gpuTransform));
	if (m_RayQuerySupported)
		m_AccelerationStructure.SetTransform(modelMatrix);
}

// ------------------------------------------------------ models and HDR maps

bool ComputeRenderer::PrepareModel(
	const std::string& path,
	PreparedModel& model,
	std::string& errorMessage)
{
	RayScene::ObjModel objModel;
	if (!RayScene::LoadObjModel(path, objModel, errorMessage))
		return false;

	std::unique_ptr<Walnut::Image> texture;
	if (!objModel.DiffuseTexturePath.empty())
	{
		const std::string texturePath = objModel.DiffuseTexturePath.string();
		int textureWidth = 0;
		int textureHeight = 0;
		int textureChannels = 0;
		if (!stbi_info(texturePath.c_str(), &textureWidth, &textureHeight, &textureChannels) ||
			textureWidth <= 0 || textureHeight <= 0)
		{
			errorMessage = "Diffuse texture could not be decoded: " + texturePath;
			return false;
		}
		texture = std::make_unique<Walnut::Image>(texturePath);
	}

	model.Path = path;
	model.Triangles = std::move(objModel.Triangles);
	model.Texture = texture ? std::move(texture) : CreateWhiteTexture();
	return true;
}

ComputeRenderer::PreparedModel ComputeRenderer::PrepareDefaultModel()
{
	PreparedModel model;
	model.Triangles = RayScene::DefaultTriangles();
	model.Texture = CreateWhiteTexture();
	return model;
}

void ComputeRenderer::CommitModel(PreparedModel&& model)
{
	m_ModelPath = std::move(model.Path);
	m_ModelTriangles = std::move(model.Triangles);
	m_TextureImage = std::move(model.Texture);
	WriteSampledImageDescriptor(BINDING_MODEL_TEXTURE, *m_TextureImage);
	UploadModel();
	ResetAccumulation();
}

bool ComputeRenderer::PrepareEnvironment(
	const std::string& path,
	PreparedEnvironment& environment,
	std::string& errorMessage)
{
	if (path.empty())
	{
		errorMessage = "Environment path is empty.";
		return false;
	}
	if (!std::ifstream(path, std::ios::binary))
	{
		errorMessage = "Environment file could not be opened: " + path;
		return false;
	}
	if (!stbi_is_hdr(path.c_str()))
	{
		errorMessage = "Environment image must be a Radiance HDR file.";
		return false;
	}

	// Everything that can be rejected from the header is rejected before the
	// pixels are decoded or any GPU memory is allocated.
	int width = 0;
	int height = 0;
	int channels = 0;
	if (!stbi_info(path.c_str(), &width, &height, &channels) ||
		width <= 0 || height <= 0)
	{
		errorMessage = "Environment image could not be decoded: " + path;
		return false;
	}
	if (static_cast<uint64_t>(width) * static_cast<uint64_t>(height) >
		MaxEnvironmentTexelCount)
	{
		errorMessage = "Environment image exceeds the importance sampling texel limit.";
		return false;
	}

	int decodedChannels = 0;
	float* pixels = stbi_loadf(path.c_str(), &width, &height, &decodedChannels, 4);
	if (!pixels)
	{
		errorMessage = "Environment pixels could not be decoded: " + path;
		return false;
	}

	// The same decoded pixels feed both the sampling distribution and the GPU
	// image, so the file is only decoded once.
	const bool hasLight = RayScene::BuildEnvironmentDistribution(
		pixels,
		static_cast<uint32_t>(width),
		static_cast<uint32_t>(height),
		environment.Distribution);
	if (hasLight)
	{
		environment.Image = std::make_unique<Walnut::Image>(
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height),
			Walnut::ImageFormat::RGBA32F,
			pixels);
	}
	stbi_image_free(pixels);

	if (!hasLight)
	{
		errorMessage = "Environment image contains no positive luminance.";
		return false;
	}

	environment.Path = path;
	environment.Width = static_cast<uint32_t>(width);
	environment.Height = static_cast<uint32_t>(height);
	return true;
}

ComputeRenderer::PreparedEnvironment ComputeRenderer::PrepareEmptyEnvironment()
{
	PreparedEnvironment environment;
	const glm::vec4 blackPixel(0.0f);
	environment.Image = CreateSolidImage(Walnut::ImageFormat::RGBA32F, &blackPixel);
	return environment;
}

void ComputeRenderer::CommitEnvironment(PreparedEnvironment&& environment)
{
	VulkanUtils::WriteBuffer(
		m_EnvironmentDistributionBuffer,
		environment.Distribution.data(),
		environment.Distribution.size() * sizeof(glm::vec4));
	m_EnvironmentPath = std::move(environment.Path);
	m_EnvironmentImage = std::move(environment.Image);
	m_EnvironmentWidth = environment.Width;
	m_EnvironmentHeight = environment.Height;
	WriteSampledImageDescriptor(BINDING_ENVIRONMENT_TEXTURE, *m_EnvironmentImage);
	ResetAccumulation();
}

bool ComputeRenderer::LoadObj(const std::string& path, std::string& errorMessage)
{
	errorMessage.clear();
	PreparedModel model;
	if (!PrepareModel(path, model, errorMessage))
		return false;

	CommitModel(std::move(model));
	return true;
}

void ComputeRenderer::ClearModel()
{
	CommitModel(PrepareDefaultModel());
}

bool ComputeRenderer::LoadEnvironmentMap(const std::string& path, std::string& errorMessage)
{
	errorMessage.clear();
	PreparedEnvironment environment;
	if (!PrepareEnvironment(path, environment, errorMessage))
		return false;

	CommitEnvironment(std::move(environment));
	return true;
}

void ComputeRenderer::ClearEnvironmentMap()
{
	CommitEnvironment(PrepareEmptyEnvironment());
}

// --------------------------------------------------------------- scene I/O

RayScene::SceneDescription ComputeRenderer::CaptureScene() const
{
	RayScene::SceneDescription scene;
	scene.Spheres = m_Spheres;
	scene.Lights = m_Lights;
	scene.SceneCamera = m_Camera;
	scene.Exposure = m_Exposure;
	scene.BounceCount = m_BounceCount;
	scene.ModelPath = m_ModelPath;
	scene.Transform = m_ModelTransform;
	scene.EnvironmentPath = m_EnvironmentPath;
	scene.EnvironmentIntensity = m_EnvironmentIntensity;
	scene.EnvironmentRotation = m_EnvironmentRotation;
	scene.Settings = m_Settings;
	return scene;
}

bool ComputeRenderer::SaveScene(const std::string& path, std::string& errorMessage) const
{
	return RayScene::SaveSceneFile(path, CaptureScene(), errorMessage);
}

bool ComputeRenderer::LoadScene(const std::string& path, std::string& errorMessage)
{
	// Starting from the current scene lets older file versions keep every
	// setting they do not store.
	RayScene::SceneDescription scene = CaptureScene();
	if (!RayScene::LoadSceneFile(path, scene, errorMessage))
		return false;

	// The model and environment are only read again when the file names a
	// different one; either way nothing is committed until both are ready.
	std::optional<PreparedModel> model;
	if (scene.ModelPath != m_ModelPath)
	{
		model.emplace();
		std::string modelError;
		if (scene.ModelPath.empty())
			*model = PrepareDefaultModel();
		else if (!PrepareModel(scene.ModelPath, *model, modelError))
		{
			errorMessage = "Scene model could not be loaded: " + modelError;
			return false;
		}
	}

	std::optional<PreparedEnvironment> environment;
	if (scene.EnvironmentPath != m_EnvironmentPath)
	{
		environment.emplace();
		std::string environmentError;
		if (scene.EnvironmentPath.empty())
			*environment = PrepareEmptyEnvironment();
		else if (!PrepareEnvironment(scene.EnvironmentPath, *environment, environmentError))
		{
			errorMessage = "Environment map could not be loaded: " + environmentError;
			return false;
		}
	}

	m_Spheres = std::move(scene.Spheres);
	m_Lights = std::move(scene.Lights);
	m_Camera = scene.SceneCamera;
	m_Exposure = scene.Exposure;
	m_BounceCount = scene.BounceCount;
	m_EnvironmentIntensity = scene.EnvironmentIntensity;
	m_EnvironmentRotation = scene.EnvironmentRotation;
	const bool rebuildTriangleBvh =
		m_Settings.UseSahSplit != scene.Settings.UseSahSplit;
	m_Settings = scene.Settings;
	m_Settings.UseRayQuery = scene.Settings.UseRayQuery && m_RayQuerySupported;
	m_ModelTransform = scene.Transform;

	UpdateCameraBasis();
	UploadSpheres();
	UploadLights();
	if (model)
		CommitModel(std::move(*model));
	else if (rebuildTriangleBvh)
		UploadModel();
	else
		UploadModelTransform();
	if (environment)
		CommitEnvironment(std::move(*environment));
	ResetAccumulation();
	return true;
}

// ------------------------------------------------------------ scene editing

const ComputeRenderer::Sphere& ComputeRenderer::GetSphere(uint32_t index) const
{
	return m_Spheres.at(index);
}

void ComputeRenderer::SetSphere(uint32_t index, const Sphere& sphere)
{
	m_Spheres.at(index) = RayScene::SanitizeSphere(sphere);
	UploadSpheres();
	ResetAccumulation();
}

bool ComputeRenderer::AddSphere()
{
	if (m_Spheres.size() >= MaxSphereCount)
		return false;

	m_Spheres.push_back({
		{ 0.0f, 0.0f, -3.0f },
		1.0f,
		{ 0.8f, 0.6f, 0.2f },
		0.1f,
		0.5f,
		MaterialType::Diffuse,
		1.5f
	});
	UploadSpheres();
	ResetAccumulation();
	return true;
}

bool ComputeRenderer::RemoveSphere(uint32_t index)
{
	if (index >= m_Spheres.size())
		return false;

	m_Spheres.erase(m_Spheres.begin() + index);
	UploadSpheres();
	ResetAccumulation();
	return true;
}

const ComputeRenderer::SphereLight& ComputeRenderer::GetLight(uint32_t index) const
{
	return m_Lights.at(index);
}

void ComputeRenderer::SetLight(uint32_t index, const SphereLight& light)
{
	m_Lights.at(index) = RayScene::SanitizeSphereLight(light);
	UploadLights();
	ResetAccumulation();
}

bool ComputeRenderer::AddLight()
{
	if (m_Lights.size() >= MaxLightCount)
		return false;

	m_Lights.push_back({ { 0.0f, 5.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, 1.0f, 15.0f });
	UploadLights();
	ResetAccumulation();
	return true;
}

bool ComputeRenderer::RemoveLight(uint32_t index)
{
	if (index >= m_Lights.size())
		return false;

	m_Lights.erase(m_Lights.begin() + index);
	UploadLights();
	ResetAccumulation();
	return true;
}

void ComputeRenderer::SetModelTransform(const ModelTransform& transform)
{
	const ModelTransform sanitized = RayScene::SanitizeModelTransform(transform);
	if (sanitized.Position == m_ModelTransform.Position &&
		sanitized.Rotation == m_ModelTransform.Rotation &&
		sanitized.Scale == m_ModelTransform.Scale)
	{
		return;
	}

	m_ModelTransform = sanitized;
	UploadModelTransform();
	ResetAccumulation();
}

void ComputeRenderer::SetEnvironmentIntensity(float intensity)
{
	const float sanitized = RayScene::SanitizeEnvironmentIntensity(intensity);
	if (m_EnvironmentIntensity == sanitized)
		return;

	m_EnvironmentIntensity = sanitized;
	ResetAccumulation();
}

void ComputeRenderer::SetEnvironmentRotation(float rotationDegrees)
{
	const float wrapped = RayScene::WrapDegrees(rotationDegrees);
	if (m_EnvironmentRotation == wrapped)
		return;

	m_EnvironmentRotation = wrapped;
	ResetAccumulation();
}

void ComputeRenderer::SetCamera(const Camera& camera)
{
	m_Camera = RayScene::SanitizeCamera(camera);
	UpdateCameraBasis();
	ResetAccumulation();
}

// Exposure only scales the averaged colour on its way to the screen, so unlike a
// camera or scene change it leaves the meaning of the accumulated samples intact
// and must not reset them.
void ComputeRenderer::SetExposure(float exposure)
{
	m_Exposure = RayScene::SanitizeExposure(exposure);
}

// The bounce limit changes how far a sample can travel through reflection and
// refraction chains, so samples made with the old limit cannot stay in the same
// progressive average.
void ComputeRenderer::SetBounceCount(uint32_t bounceCount)
{
	const uint32_t sanitized = RayScene::SanitizeBounceCount(bounceCount);
	if (m_BounceCount == sanitized)
		return;

	m_BounceCount = sanitized;
	ResetAccumulation();
}

void ComputeRenderer::SetSahSplitEnabled(bool enabled)
{
	if (m_Settings.UseSahSplit == enabled)
		return;

	m_Settings.UseSahSplit = enabled;
	UploadSpheres();
	UploadModel();
}

void ComputeRenderer::SetStochasticLightsEnabled(bool enabled)
{
	if (m_Settings.UseStochasticLights == enabled)
		return;

	m_Settings.UseStochasticLights = enabled;
	ResetAccumulation();
}

void ComputeRenderer::SetRayQueryEnabled(bool enabled)
{
	const bool supportedValue = enabled && m_RayQuerySupported;
	if (m_Settings.UseRayQuery == supportedValue)
		return;

	m_Settings.UseRayQuery = supportedValue;
	ResetAccumulation();
}

// Turns the two angles the UI edits into the forward vector the shader needs.
// Zero yaw and zero pitch has to come out as -Z because that is the direction
// the camera looks in by default.
void ComputeRenderer::UpdateCameraBasis()
{
	const float yaw = glm::radians(m_Camera.Yaw);
	const float pitch = glm::radians(m_Camera.Pitch);

	glm::vec3 forward;
	forward.x = glm::cos(pitch) * glm::sin(yaw);
	forward.y = glm::sin(pitch);
	forward.z = -glm::cos(pitch) * glm::cos(yaw);
	m_CameraForward = glm::normalize(forward);
}

void ComputeRenderer::ResetAccumulation()
{
	m_FrameIndex = 0;
}

// ------------------------------------------------------------- images

void ComputeRenderer::Resize(uint32_t width, uint32_t height)
{
	if (width == 0 || height == 0 || (width == m_Width && height == m_Height))
		return;

	VkDevice device = Walnut::Application::GetDevice();
	check_vk_result(vkDeviceWaitIdle(device));

	vkDestroyImageView(device, m_OutputImageView, nullptr);
	vkDestroyImage(device, m_OutputImage, nullptr);
	vkFreeMemory(device, m_OutputImageMemory, nullptr);
	vkDestroyImageView(device, m_AccumulationImageView, nullptr);
	vkDestroyImage(device, m_AccumulationImage, nullptr);
	vkFreeMemory(device, m_AccumulationImageMemory, nullptr);

	m_Width = width;
	m_Height = height;
	CreateOutputImages();
	WriteStorageImageDescriptor(BINDING_OUTPUT_IMAGE, m_OutputImageView);
	WriteStorageImageDescriptor(BINDING_ACCUMULATION_IMAGE, m_AccumulationImageView);
	ResetAccumulation();
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
	allocationInfo.memoryTypeIndex = VulkanUtils::FindMemoryType(
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

	if (m_OutputSampler == VK_NULL_HANDLE)
	{
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
	}

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

	if (m_ImGuiDescriptorSet == VK_NULL_HANDLE)
	{
		m_ImGuiDescriptorSet = ImGui_ImplVulkan_AddTexture(
			m_OutputSampler,
			m_OutputImageView,
			VK_IMAGE_LAYOUT_GENERAL);
	}
	else
	{
		UpdateImGuiImageDescriptor();
	}
}

void ComputeRenderer::UpdateImGuiImageDescriptor()
{
	VkDescriptorImageInfo descriptorImage{};
	descriptorImage.sampler = m_OutputSampler;
	descriptorImage.imageView = m_OutputImageView;
	descriptorImage.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet descriptorWrite{};
	descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	descriptorWrite.dstSet = m_ImGuiDescriptorSet;
	descriptorWrite.dstBinding = 0;
	descriptorWrite.descriptorCount = 1;
	descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorWrite.pImageInfo = &descriptorImage;
	vkUpdateDescriptorSets(Walnut::Application::GetDevice(), 1, &descriptorWrite, 0, nullptr);
}

// ------------------------------------------------------------ descriptors

void ComputeRenderer::CreateComputeDescriptors()
{
	VkDevice device = Walnut::Application::GetDevice();

	const std::pair<uint32_t, VkDescriptorType> bindingTypes[] = {
		{ BINDING_OUTPUT_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
		{ BINDING_ACCUMULATION_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
		{ BINDING_SPHERES, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
		{ BINDING_SPHERE_BVH, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
		{ BINDING_LIGHTS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
		{ BINDING_TRIANGLES, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
		{ BINDING_TRIANGLE_BVH, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
		{ BINDING_MODEL_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER },
		{ BINDING_ENVIRONMENT_TEXTURE, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER },
		{ BINDING_ENVIRONMENT_DISTRIBUTION, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
		{ BINDING_MODEL_TRANSFORM, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER },
		{ BINDING_TRIANGLE_TLAS, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR },
	};

	std::vector<VkDescriptorSetLayoutBinding> bindings;
	std::vector<VkDescriptorPoolSize> poolSizes;
	for (const auto& [binding, type] : bindingTypes)
	{
		// The fallback shader has no TLAS, and the device may not even know the
		// descriptor type.
		if (type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR && !m_RayQuerySupported)
			continue;

		VkDescriptorSetLayoutBinding layoutBinding{};
		layoutBinding.binding = binding;
		layoutBinding.descriptorType = type;
		layoutBinding.descriptorCount = 1;
		layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		bindings.push_back(layoutBinding);

		const auto poolSize = std::find_if(
			poolSizes.begin(),
			poolSizes.end(),
			[type = type](const VkDescriptorPoolSize& size) { return size.type == type; });
		if (poolSize == poolSizes.end())
			poolSizes.push_back({ type, 1 });
		else
			poolSize->descriptorCount++;
	}

	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layoutInfo.pBindings = bindings.data();
	check_vk_result(vkCreateDescriptorSetLayout(
		device, &layoutInfo, nullptr, &m_ComputeDescriptorSetLayout));

	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.maxSets = 1;
	poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
	poolInfo.pPoolSizes = poolSizes.data();
	check_vk_result(vkCreateDescriptorPool(
		device, &poolInfo, nullptr, &m_ComputeDescriptorPool));

	VkDescriptorSetAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocateInfo.descriptorPool = m_ComputeDescriptorPool;
	allocateInfo.descriptorSetCount = 1;
	allocateInfo.pSetLayouts = &m_ComputeDescriptorSetLayout;
	check_vk_result(vkAllocateDescriptorSets(device, &allocateInfo, &m_ComputeDescriptorSet));

	WriteStorageImageDescriptor(BINDING_OUTPUT_IMAGE, m_OutputImageView);
	WriteStorageImageDescriptor(BINDING_ACCUMULATION_IMAGE, m_AccumulationImageView);
	WriteStorageBufferDescriptor(BINDING_SPHERES, m_SphereBuffer);
	WriteStorageBufferDescriptor(BINDING_SPHERE_BVH, m_SphereBvhBuffer);
	WriteStorageBufferDescriptor(BINDING_LIGHTS, m_LightBuffer);
	WriteStorageBufferDescriptor(BINDING_TRIANGLES, m_TriangleBuffer);
	WriteStorageBufferDescriptor(BINDING_TRIANGLE_BVH, m_TriangleBvhBuffer);
	WriteStorageBufferDescriptor(
		BINDING_ENVIRONMENT_DISTRIBUTION, m_EnvironmentDistributionBuffer);
	WriteStorageBufferDescriptor(BINDING_MODEL_TRANSFORM, m_ModelTransformBuffer);
	WriteSampledImageDescriptor(BINDING_MODEL_TEXTURE, *m_TextureImage);
	WriteSampledImageDescriptor(BINDING_ENVIRONMENT_TEXTURE, *m_EnvironmentImage);
	// The TLAS binding is written once the first model has been built.
}

void ComputeRenderer::WriteStorageImageDescriptor(uint32_t binding, VkImageView view)
{
	VkDescriptorImageInfo imageInfo{};
	imageInfo.imageView = view;
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = m_ComputeDescriptorSet;
	write.dstBinding = binding;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	write.pImageInfo = &imageInfo;
	vkUpdateDescriptorSets(Walnut::Application::GetDevice(), 1, &write, 0, nullptr);
}

void ComputeRenderer::WriteStorageBufferDescriptor(
	uint32_t binding,
	const VulkanUtils::Buffer& buffer)
{
	VkDescriptorBufferInfo bufferInfo{};
	bufferInfo.buffer = buffer.Handle;
	bufferInfo.range = VK_WHOLE_SIZE;

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = m_ComputeDescriptorSet;
	write.dstBinding = binding;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pBufferInfo = &bufferInfo;
	vkUpdateDescriptorSets(Walnut::Application::GetDevice(), 1, &write, 0, nullptr);
}

void ComputeRenderer::WriteSampledImageDescriptor(
	uint32_t binding,
	const Walnut::Image& image)
{
	if (m_ComputeDescriptorSet == VK_NULL_HANDLE)
		return;

	VkDescriptorImageInfo imageInfo{};
	imageInfo.sampler = image.GetSampler();
	imageInfo.imageView = image.GetImageView();
	imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = m_ComputeDescriptorSet;
	write.dstBinding = binding;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	write.pImageInfo = &imageInfo;
	vkUpdateDescriptorSets(Walnut::Application::GetDevice(), 1, &write, 0, nullptr);
}

void ComputeRenderer::WriteTlasDescriptor()
{
	const VkAccelerationStructureKHR tlas = m_AccelerationStructure.GetTlas();
	if (m_ComputeDescriptorSet == VK_NULL_HANDLE || tlas == VK_NULL_HANDLE)
		return;

	VkWriteDescriptorSetAccelerationStructureKHR accelerationInfo{};
	accelerationInfo.sType =
		VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	accelerationInfo.accelerationStructureCount = 1;
	accelerationInfo.pAccelerationStructures = &tlas;

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.pNext = &accelerationInfo;
	write.dstSet = m_ComputeDescriptorSet;
	write.dstBinding = BINDING_TRIANGLE_TLAS;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	vkUpdateDescriptorSets(Walnut::Application::GetDevice(), 1, &write, 0, nullptr);
}

// ------------------------------------------------------ pipeline and timing

void ComputeRenderer::CreateComputePipeline(const std::string& shaderPath)
{
	const std::vector<uint32_t> shaderCode = ReadShaderFile(shaderPath);
	if (shaderCode.empty())
	{
		throw std::runtime_error(
			"Compute shader could not be read: " + shaderPath +
			". Build the project so the SPIR-V is generated, and start the app "
			"with WalnutApp as its working directory.");
	}

	VkDevice device = Walnut::Application::GetDevice();
	VkShaderModuleCreateInfo shaderInfo{};
	shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shaderInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
	shaderInfo.pCode = shaderCode.data();
	VkShaderModule shaderModule = VK_NULL_HANDLE;
	check_vk_result(vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule));

	VkPushConstantRange pushConstantRange{};
	pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
	pushConstantRange.size = sizeof(GpuTypes::PushConstants);

	VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutInfo.setLayoutCount = 1;
	pipelineLayoutInfo.pSetLayouts = &m_ComputeDescriptorSetLayout;
	pipelineLayoutInfo.pushConstantRangeCount = 1;
	pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
	check_vk_result(vkCreatePipelineLayout(
		device, &pipelineLayoutInfo, nullptr, &m_ComputePipelineLayout));

	VkComputePipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	pipelineInfo.stage.module = shaderModule;
	pipelineInfo.stage.pName = "main";
	pipelineInfo.layout = m_ComputePipelineLayout;
	const VkResult result = vkCreateComputePipelines(
		device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_ComputePipeline);
	vkDestroyShaderModule(device, shaderModule, nullptr);
	check_vk_result(result);
}

void ComputeRenderer::CreateTimestampQueryPool()
{
	VkPhysicalDevice physicalDevice = Walnut::Application::GetPhysicalDevice();
	VkPhysicalDeviceProperties deviceProperties{};
	vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

	// A timestampPeriod of zero would make the tick to millisecond conversion
	// meaningless, so treat it the same way as missing hardware support.
	if (deviceProperties.limits.timestampComputeAndGraphics != VK_TRUE ||
		deviceProperties.limits.timestampPeriod <= 0.0f)
	{
		return;
	}

	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(
		physicalDevice, &queueFamilyCount, queueFamilies.data());

	const uint32_t queueFamilyIndex = Walnut::Application::GetQueueFamily();
	if (queueFamilyIndex >= queueFamilyCount)
		return;

	const uint32_t validBits = queueFamilies[queueFamilyIndex].timestampValidBits;
	if (validBits == 0)
		return;

	// Only the low validBits of every timestamp carry data; the rest must be
	// masked off before two timestamps can be subtracted.
	m_TimestampValidMask = validBits >= 64
		? ~static_cast<uint64_t>(0)
		: (static_cast<uint64_t>(1) << validBits) - 1;
	m_TimestampPeriodNs = deviceProperties.limits.timestampPeriod;

	VkQueryPoolCreateInfo queryPoolInfo{};
	queryPoolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
	queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
	queryPoolInfo.queryCount = TimestampQueryCount;
	check_vk_result(vkCreateQueryPool(
		Walnut::Application::GetDevice(), &queryPoolInfo, nullptr, &m_TimestampQueryPool));
}

void ComputeRenderer::ReadGpuComputeTime()
{
	if (m_TimestampQueryPool == VK_NULL_HANDLE)
		return;

	// FlushCommandBuffer already waited on a fence, so both queries have
	// finished and this read never blocks.
	std::array<uint64_t, TimestampQueryCount> timestamps{};
	const VkResult result = vkGetQueryPoolResults(
		Walnut::Application::GetDevice(),
		m_TimestampQueryPool,
		0,
		TimestampQueryCount,
		sizeof(timestamps),
		timestamps.data(),
		sizeof(uint64_t),
		VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
	if (result != VK_SUCCESS)
		return;

	const uint64_t begin = timestamps[0] & m_TimestampValidMask;
	const uint64_t end = timestamps[1] & m_TimestampValidMask;
	const uint64_t elapsed = (end - begin) & m_TimestampValidMask;
	const float elapsedMs =
		static_cast<float>(elapsed) * m_TimestampPeriodNs / 1000000.0f;
	m_GpuComputeTimeMs = SmoothTiming(m_GpuComputeTimeMs, elapsedMs);
}

// ----------------------------------------------------------------- render

void ComputeRenderer::Render()
{
	if (!IsInitialized())
		return;

	const auto cpuRenderBegin = std::chrono::steady_clock::now();

	m_FrameIndex++;
	uint32_t flags = 0;
	if (m_Settings.UseBvh)
		flags |= RENDER_FLAG_USE_BVH;
	if (m_Settings.UseStochasticLights)
		flags |= RENDER_FLAG_STOCHASTIC_LIGHTS;
	if (m_Settings.UseRayQuery && m_RayQuerySupported)
		flags |= RENDER_FLAG_RAY_QUERY;
	if (m_EnvironmentWidth > 0 && m_EnvironmentHeight > 0)
		flags |= RENDER_FLAG_ENVIRONMENT_MAP;

	GpuTypes::PushConstants pushConstants{};
	pushConstants.CameraPosition = glm::vec4(m_Camera.Position, 1.0f);
	pushConstants.CameraForward = glm::vec4(m_CameraForward, 0.0f);
	pushConstants.FrameIndex = m_FrameIndex;
	pushConstants.VerticalFov = m_Camera.VerticalFov;
	pushConstants.Exposure = m_Exposure;
	pushConstants.SphereCount = GetSphereCount();
	pushConstants.TriangleCount = GetTriangleCount();
	pushConstants.LightCount = GetLightCount();
	pushConstants.BounceCount = m_BounceCount;
	pushConstants.Flags = flags;
	pushConstants.EnvironmentIntensity = m_EnvironmentIntensity;
	pushConstants.EnvironmentRotation = glm::radians(m_EnvironmentRotation);
	pushConstants.EnvironmentWidth = m_EnvironmentWidth;
	pushConstants.EnvironmentHeight = m_EnvironmentHeight;

	VkCommandBuffer commandBuffer = Walnut::Application::GetCommandBuffer(true);

	// Queries hold results from the previous frame and must be reset before
	// they can be written again.
	if (m_TimestampQueryPool != VK_NULL_HANDLE)
		vkCmdResetQueryPool(commandBuffer, m_TimestampQueryPool, 0, TimestampQueryCount);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_ComputePipeline);
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
		sizeof(pushConstants),
		&pushConstants);

	// The two timestamps bracket only the dispatch, so their difference is the
	// time the GPU itself spent tracing rays.
	if (m_TimestampQueryPool != VK_NULL_HANDLE)
	{
		vkCmdWriteTimestamp(
			commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_TimestampQueryPool, 0);
	}

	vkCmdDispatch(
		commandBuffer,
		(m_Width + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
		(m_Height + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
		1);

	if (m_TimestampQueryPool != VK_NULL_HANDLE)
	{
		vkCmdWriteTimestamp(
			commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_TimestampQueryPool, 1);
	}

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

	// Waiting here keeps every scene buffer free to rewrite between frames
	// without extra synchronisation, at the price of CPU and GPU never overlapping.
	Walnut::Application::FlushCommandBuffer(commandBuffer);
	ReadGpuComputeTime();

	// This includes the fence wait inside FlushCommandBuffer, so it measures how
	// long the whole render call blocks the CPU rather than CPU-only work. On a
	// vsync limited swapchain the wait also absorbs the wait for the display; use
	// the GPU compute time to judge the shader.
	const std::chrono::duration<float, std::milli> cpuRenderDuration =
		std::chrono::steady_clock::now() - cpuRenderBegin;
	m_CpuRenderTimeMs = SmoothTiming(m_CpuRenderTimeMs, cpuRenderDuration.count());
}

void ComputeRenderer::Release()
{
	if (m_OutputImage == VK_NULL_HANDLE)
		return;

	check_vk_result(vkDeviceWaitIdle(Walnut::Application::GetDevice()));
	m_AccelerationStructure.Release();

	Walnut::Application::SubmitResourceFree(
		[pipeline = m_ComputePipeline,
		 pipelineLayout = m_ComputePipelineLayout,
		 descriptorPool = m_ComputeDescriptorPool,
		 descriptorSetLayout = m_ComputeDescriptorSetLayout,
		 sampler = m_OutputSampler,
		 outputImageView = m_OutputImageView,
		 outputImage = m_OutputImage,
		 outputImageMemory = m_OutputImageMemory,
		 accumulationImageView = m_AccumulationImageView,
		 accumulationImage = m_AccumulationImage,
		 accumulationImageMemory = m_AccumulationImageMemory,
		 timestampQueryPool = m_TimestampQueryPool,
		 buffers = std::array<VulkanUtils::Buffer, 7>{
			m_SphereBuffer,
			m_SphereBvhBuffer,
			m_LightBuffer,
			m_TriangleBuffer,
			m_TriangleBvhBuffer,
			m_EnvironmentDistributionBuffer,
			m_ModelTransformBuffer }]() mutable
		{
			VkDevice device = Walnut::Application::GetDevice();
			vkDestroyQueryPool(device, timestampQueryPool, nullptr);
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
			for (VulkanUtils::Buffer& buffer : buffers)
				VulkanUtils::DestroyBuffer(buffer);
		});

	m_OutputImage = VK_NULL_HANDLE;
	m_ComputePipeline = VK_NULL_HANDLE;
	m_TimestampQueryPool = VK_NULL_HANDLE;
}
