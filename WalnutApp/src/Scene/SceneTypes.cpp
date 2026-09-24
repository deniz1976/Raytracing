#include "SceneTypes.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace RayScene
{
	namespace
	{
		float FiniteOr(float value, float fallback)
		{
			return std::isfinite(value) ? value : fallback;
		}

		bool IsFinite(const glm::vec3& value)
		{
			return std::isfinite(value.x) &&
				std::isfinite(value.y) &&
				std::isfinite(value.z);
		}

		glm::vec3 SanitizeColor(const glm::vec3& color, float fallback)
		{
			return glm::clamp(
				glm::vec3(
					FiniteOr(color.r, fallback),
					FiniteOr(color.g, fallback),
					FiniteOr(color.b, fallback)),
				glm::vec3(0.0f),
				glm::vec3(1.0f));
		}
	}

	Sphere SanitizeSphere(const Sphere& sphere)
	{
		Sphere result = sphere;
		result.Radius = std::clamp(FiniteOr(result.Radius, 0.5f), 0.05f, 200.0f);
		result.Albedo = SanitizeColor(result.Albedo, 0.8f);
		result.Reflectivity =
			std::clamp(FiniteOr(result.Reflectivity, 0.0f), 0.0f, 1.0f);
		result.Roughness =
			std::clamp(FiniteOr(result.Roughness, 0.5f), 0.0f, 1.0f);
		result.IndexOfRefraction =
			std::clamp(FiniteOr(result.IndexOfRefraction, 1.5f), 1.0f, 2.5f);
		if (result.Type != MaterialType::Legacy &&
			result.Type != MaterialType::Diffuse &&
			result.Type != MaterialType::Metal &&
			result.Type != MaterialType::Dielectric)
		{
			result.Type = MaterialType::Legacy;
		}
		return result;
	}

	SphereLight SanitizeSphereLight(const SphereLight& light)
	{
		SphereLight result = light;
		result.Color = SanitizeColor(result.Color, 1.0f);
		result.Radius = std::clamp(FiniteOr(result.Radius, 0.5f), 0.05f, 20.0f);
		result.Intensity =
			std::clamp(FiniteOr(result.Intensity, 1.0f), 0.0f, 100.0f);
		return result;
	}

	Camera SanitizeCamera(const Camera& camera)
	{
		Camera result = camera;
		result.Pitch = std::clamp(FiniteOr(result.Pitch, 0.0f), -89.0f, 89.0f);
		result.VerticalFov =
			std::clamp(FiniteOr(result.VerticalFov, 45.0f), 20.0f, 90.0f);
		return result;
	}

	ModelTransform SanitizeModelTransform(const ModelTransform& transform)
	{
		ModelTransform result = transform;
		if (!IsFinite(result.Position))
			result.Position = glm::vec3(0.0f);
		if (!IsFinite(result.Rotation))
			result.Rotation = glm::vec3(0.0f);
		if (!IsFinite(result.Scale))
			result.Scale = glm::vec3(1.0f);
		// Scale stays positive so the transform never mirrors a triangle and
		// flips its winding.
		result.Scale = glm::clamp(
			result.Scale,
			glm::vec3(MinModelScale),
			glm::vec3(MaxModelScale));
		return result;
	}

	float SanitizeExposure(float exposure)
	{
		return std::clamp(FiniteOr(exposure, 1.0f), MinExposure, MaxExposure);
	}

	uint32_t SanitizeBounceCount(uint32_t bounceCount)
	{
		return std::clamp(bounceCount, MinBounceCount, MaxBounceCount);
	}

	float SanitizeEnvironmentIntensity(float intensity)
	{
		return std::clamp(
			FiniteOr(intensity, 1.0f),
			0.0f,
			MaxEnvironmentIntensity);
	}

	float WrapDegrees(float degrees)
	{
		if (!std::isfinite(degrees))
			return 0.0f;

		float wrapped = std::fmod(degrees + 180.0f, 360.0f);
		if (wrapped < 0.0f)
			wrapped += 360.0f;
		wrapped -= 180.0f;
		// fmod of a value a hair below a multiple of 360 can round up to 360.
		return wrapped >= 180.0f ? -180.0f : wrapped;
	}

	bool IsFinite(const Sphere& sphere)
	{
		return IsFinite(sphere.Center) &&
			std::isfinite(sphere.Radius) &&
			IsFinite(sphere.Albedo) &&
			std::isfinite(sphere.Reflectivity) &&
			std::isfinite(sphere.Roughness) &&
			std::isfinite(sphere.IndexOfRefraction);
	}

	bool IsFinite(const SphereLight& light)
	{
		return IsFinite(light.Position) &&
			IsFinite(light.Color) &&
			std::isfinite(light.Radius) &&
			std::isfinite(light.Intensity);
	}

	bool IsFinite(const Camera& camera)
	{
		return IsFinite(camera.Position) &&
			std::isfinite(camera.Yaw) &&
			std::isfinite(camera.Pitch) &&
			std::isfinite(camera.VerticalFov);
	}

	bool IsFinite(const ModelTransform& transform)
	{
		return IsFinite(transform.Position) &&
			IsFinite(transform.Rotation) &&
			IsFinite(transform.Scale);
	}

	std::vector<Triangle> DefaultTriangles()
	{
		return {
			Triangle{
				{ -1.25f, 0.1f, -3.0f },
				{ 1.25f, 0.1f, -3.0f },
				{ 0.0f, 2.2f, -3.0f },
				{ 0.8f, 0.65f, 0.15f },
				0.0f,
				0.5f,
				MaterialType::Diffuse,
				1.5f }
		};
	}

	std::vector<Sphere> DefaultSpheres()
	{
		return {
			Sphere{ { 0.0f, 0.0f, 0.0f }, 1.0f, { 0.85f, 0.18f, 0.12f },
				0.15f, 0.65f, MaterialType::Diffuse, 1.5f },
			Sphere{ { -2.1f, 0.0f, -1.0f }, 1.0f, { 0.12f, 0.35f, 0.85f },
				0.9f, 0.05f, MaterialType::Metal, 1.5f },
			Sphere{ { 2.1f, 0.0f, -1.0f }, 1.0f, { 0.92f, 1.0f, 0.95f },
				1.0f, 0.0f, MaterialType::Dielectric, 1.5f },
			Sphere{ { 0.0f, -101.0f, 0.0f }, 100.0f, { 0.55f, 0.55f, 0.55f },
				0.15f, 0.55f, MaterialType::Diffuse, 1.5f }
		};
	}

	std::vector<SphereLight> DefaultLights()
	{
		return {
			SphereLight{ { -2.5f, 5.0f, 2.0f }, { 1.0f, 0.95f, 0.85f }, 1.5f, 24.0f },
			SphereLight{ { 3.5f, 4.0f, -2.0f }, { 0.45f, 0.6f, 1.0f }, 1.0f, 12.0f }
		};
	}

	glm::mat4 ComposeModelMatrix(const ModelTransform& transform)
	{
		glm::mat4 matrix(1.0f);
		matrix = glm::translate(matrix, transform.Position);
		matrix = glm::rotate(
			matrix, glm::radians(transform.Rotation.z), { 0.0f, 0.0f, 1.0f });
		matrix = glm::rotate(
			matrix, glm::radians(transform.Rotation.y), { 0.0f, 1.0f, 0.0f });
		matrix = glm::rotate(
			matrix, glm::radians(transform.Rotation.x), { 1.0f, 0.0f, 0.0f });
		return glm::scale(matrix, transform.Scale);
	}
}
