#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

// Plain scene data shared by the renderer, the scene file and the OBJ loader.
// Nothing in here touches Vulkan, so every module built on these types can be
// compiled and tested without a GPU.
namespace RayScene
{
	enum class MaterialType : uint32_t
	{
		Legacy = 0,
		Diffuse = 1,
		Metal = 2,
		Dielectric = 3
	};

	struct Sphere
	{
		glm::vec3 Center;
		float Radius;
		glm::vec3 Albedo;
		float Reflectivity;
		float Roughness;
		MaterialType Type = MaterialType::Legacy;
		float IndexOfRefraction = 1.5f;
	};

	struct SphereLight
	{
		glm::vec3 Position;
		glm::vec3 Color;
		float Radius;
		float Intensity;
	};

	struct Triangle
	{
		glm::vec3 Vertex0;
		glm::vec3 Vertex1;
		glm::vec3 Vertex2;
		glm::vec3 Albedo;
		float Reflectivity;
		float Roughness;
		MaterialType Type = MaterialType::Diffuse;
		float IndexOfRefraction = 1.5f;
		glm::vec3 Normal0{ 0.0f };
		glm::vec3 Normal1{ 0.0f };
		glm::vec3 Normal2{ 0.0f };
		uint32_t HasVertexNormals = 0;
		glm::vec2 TexCoord0{ 0.0f };
		glm::vec2 TexCoord1{ 0.0f };
		glm::vec2 TexCoord2{ 0.0f };
		uint32_t HasTexCoords = 0;
		uint32_t UsesImageTexture = 0;
	};

	struct ModelTransform
	{
		glm::vec3 Position{ 0.0f };
		glm::vec3 Rotation{ 0.0f };
		glm::vec3 Scale{ 1.0f };
	};

	// Yaw and pitch are stored in degrees instead of a forward vector because
	// they are what the UI edits and what a scene file can round trip; the
	// renderer derives the forward vector from them.
	struct Camera
	{
		glm::vec3 Position = { 0.0f, 0.0f, 3.0f };
		float Yaw = 0.0f;
		float Pitch = 0.0f;
		float VerticalFov = 45.0f;
	};

	// Every limit below is read by both the UI controls and the sanitizers a
	// scene file goes through, so a value that came out of a file is always one
	// the controls can still represent.
	constexpr uint32_t MaxSphereCount = 512;
	constexpr uint32_t MaxTriangleCount = 4096;
	// Every light costs one shadow ray per bounce, so this capacity is kept far
	// smaller than the sphere capacity.
	constexpr uint32_t MaxLightCount = 8;
	constexpr uint32_t MaxEnvironmentTexelCount = 262144;
	constexpr float MinExposure = 0.1f;
	constexpr float MaxExposure = 4.0f;
	constexpr uint32_t MinBounceCount = 1;
	constexpr uint32_t MaxBounceCount = 10;
	constexpr float MaxEnvironmentIntensity = 20.0f;
	constexpr float MinModelScale = 0.01f;
	constexpr float MaxModelScale = 100.0f;

	Sphere SanitizeSphere(const Sphere& sphere);
	SphereLight SanitizeSphereLight(const SphereLight& light);
	// Clamping the pitch away from straight up also keeps the forward vector from
	// lining up with the world up vector, which would collapse the camera basis.
	Camera SanitizeCamera(const Camera& camera);
	ModelTransform SanitizeModelTransform(const ModelTransform& transform);
	float SanitizeExposure(float exposure);
	uint32_t SanitizeBounceCount(uint32_t bounceCount);
	float SanitizeEnvironmentIntensity(float intensity);
	// Wraps any finite angle into [-180, 180), the range the rotation slider shows.
	float WrapDegrees(float degrees);

	bool IsFinite(const Sphere& sphere);
	bool IsFinite(const SphereLight& light);
	bool IsFinite(const Camera& camera);
	bool IsFinite(const ModelTransform& transform);

	// The single triangle shown when no OBJ model is loaded.
	std::vector<Triangle> DefaultTriangles();
	std::vector<Sphere> DefaultSpheres();
	// A warm key light on one side and a dimmer cool fill on the other, so the
	// scene shows two distinct shadow directions out of the box.
	std::vector<SphereLight> DefaultLights();
	glm::mat4 ComposeModelMatrix(const ModelTransform& transform);
}
