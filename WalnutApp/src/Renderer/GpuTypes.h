#pragma once

#include <cstdint>

#include <glm/glm.hpp>

// std430 mirrors of the structs RayTracing.comp declares. Each one packs its
// fields into vec4s so the C++ and GLSL layouts agree without any padding rules.
namespace GpuTypes
{
	struct alignas(16) PushConstants
	{
		glm::vec4 CameraPosition;
		glm::vec4 CameraForward;
		uint32_t FrameIndex;
		float VerticalFov;
		float Exposure;
		uint32_t SphereCount;
		uint32_t TriangleCount;
		uint32_t LightCount;
		uint32_t BounceCount;
		// RENDER_FLAG_* bits from ShaderInterface.h.
		uint32_t Flags;
		float EnvironmentIntensity;
		// Radians.
		float EnvironmentRotation;
		uint32_t EnvironmentWidth;
		uint32_t EnvironmentHeight;
	};

	// Vulkan guarantees 128 bytes of push constants, which leaves room to grow.
	static_assert(sizeof(PushConstants) == 80);

	struct alignas(16) Sphere
	{
		glm::vec4 CenterRadius;
		glm::vec4 AlbedoReflectivity;
		// x stores roughness, y the material type and z the index of refraction.
		glm::vec4 RoughnessMaterial;
	};

	static_assert(sizeof(Sphere) == 48);

	struct alignas(16) Triangle
	{
		glm::vec4 Vertex0;
		glm::vec4 Vertex1;
		glm::vec4 Vertex2;
		glm::vec4 AlbedoReflectivity;
		glm::vec4 RoughnessMaterial;
		// Normal0.w is one when all three OBJ corner normals are available.
		glm::vec4 Normal0;
		glm::vec4 Normal1;
		glm::vec4 Normal2;
		glm::vec4 TexCoord01;
		// TexCoord2.z is one when an image texture is sampled and w is one when
		// all three OBJ corners have UV coordinates.
		glm::vec4 TexCoord2;
	};

	static_assert(sizeof(Triangle) == 160);

	// Position and intensity share one vec4 because std430 would pad a lone vec3
	// out to 16 bytes anyway.
	struct alignas(16) SphereLight
	{
		glm::vec4 PositionIntensity;
		glm::vec4 Color;
		// x stores the radius, z the selection CDF and w its probability.
		glm::vec4 RadiusSampling;
	};

	static_assert(sizeof(SphereLight) == 48);

	struct alignas(16) ModelTransform
	{
		glm::mat4 WorldToObject;
	};

	static_assert(sizeof(ModelTransform) == 64);
}
