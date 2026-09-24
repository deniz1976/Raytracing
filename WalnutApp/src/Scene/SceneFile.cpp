#include "SceneFile.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

// Format history. Every version only appends blocks to the end of the previous
// one, so older fields never move and every older file still loads:
//   1  spheres and a single rectangular light
//   2  light count
//   3  camera
//   4  exposure
//   5  per sphere material types (older files keep the Legacy model)
//   6  per sphere index of refraction
//   7  exact sphere light radii (the legacy size pair stays in LIGHT lines)
//   8  OBJ model path and transform
//   9  environment map path, intensity and rotation
//   10 bounce count and render settings; an empty model path is allowed
namespace RayScene
{
	namespace
	{
		// Reads the next token and checks that it is the expected label.
		bool ExpectLabel(std::istream& stream, const char* expected)
		{
			std::string label;
			return (stream >> label) && label == expected;
		}

		// Reads "LABEL value" and requires the value to be finite.
		bool ReadFloat(std::istream& stream, const char* label, float& value)
		{
			return ExpectLabel(stream, label) &&
				(stream >> value) &&
				std::isfinite(value);
		}

		// Reads "LABEL count" and requires the count to match an earlier block.
		bool ReadMatchingCount(
			std::istream& stream,
			const char* label,
			uint64_t expectedCount)
		{
			uint64_t count = 0;
			return ExpectLabel(stream, label) &&
				(stream >> count) &&
				count == expectedCount;
		}

		bool ReadFlag(std::istream& stream, bool& value)
		{
			uint32_t flag = 0;
			if (!(stream >> flag) || flag > 1)
				return false;
			value = flag != 0;
			return true;
		}

		bool Fail(std::string& errorMessage, const char* message)
		{
			errorMessage = message;
			return false;
		}
	}

	bool SaveSceneFile(
		const std::string& path,
		const SceneDescription& scene,
		std::string& errorMessage)
	{
		errorMessage.clear();
		const std::filesystem::path scenePath(path);
		if (scenePath.has_parent_path())
		{
			std::error_code directoryError;
			std::filesystem::create_directories(
				scenePath.parent_path(),
				directoryError);
			if (directoryError)
				return Fail(errorMessage, "Scene directory could not be created.");
		}

		std::ofstream file(scenePath, std::ios::trunc);
		if (!file)
			return Fail(errorMessage, "Scene file could not be opened for writing.");

		file << std::setprecision(std::numeric_limits<float>::max_digits10);
		file << "WALNUT_RAY_SCENE " << CurrentSceneFileVersion << '\n';
		file << "SPHERE_COUNT " << scene.Spheres.size() << '\n';
		for (const Sphere& sphere : scene.Spheres)
		{
			file << "SPHERE "
				<< sphere.Center.x << ' '
				<< sphere.Center.y << ' '
				<< sphere.Center.z << ' '
				<< sphere.Radius << ' '
				<< sphere.Albedo.r << ' '
				<< sphere.Albedo.g << ' '
				<< sphere.Albedo.b << ' '
				<< sphere.Reflectivity << ' '
				<< sphere.Roughness << '\n';
		}
		file << "LIGHT_COUNT " << scene.Lights.size() << '\n';
		for (const SphereLight& light : scene.Lights)
		{
			file << "LIGHT "
				<< light.Position.x << ' '
				<< light.Position.y << ' '
				<< light.Position.z << ' '
				<< light.Color.r << ' '
				<< light.Color.g << ' '
				<< light.Color.b << ' '
				<< light.Radius * 2.0f << ' '
				<< light.Radius * 2.0f << ' '
				<< light.Intensity << '\n';
		}
		const Camera& camera = scene.SceneCamera;
		file << "CAMERA "
			<< camera.Position.x << ' '
			<< camera.Position.y << ' '
			<< camera.Position.z << ' '
			<< camera.Yaw << ' '
			<< camera.Pitch << ' '
			<< camera.VerticalFov << '\n';
		file << "EXPOSURE " << scene.Exposure << '\n';
		file << "MATERIAL_COUNT " << scene.Spheres.size() << '\n';
		for (const Sphere& sphere : scene.Spheres)
			file << "MATERIAL " << static_cast<uint32_t>(sphere.Type) << '\n';
		file << "IOR_COUNT " << scene.Spheres.size() << '\n';
		for (const Sphere& sphere : scene.Spheres)
			file << "IOR " << sphere.IndexOfRefraction << '\n';
		file << "LIGHT_RADIUS_COUNT " << scene.Lights.size() << '\n';
		for (const SphereLight& light : scene.Lights)
			file << "LIGHT_RADIUS " << light.Radius << '\n';
		// std::quoted preserves spaces without inventing a separate escaping format.
		file << "MODEL_PATH " << std::quoted(scene.ModelPath) << '\n';
		const ModelTransform& transform = scene.Transform;
		file << "MODEL_TRANSFORM "
			<< transform.Position.x << ' '
			<< transform.Position.y << ' '
			<< transform.Position.z << ' '
			<< transform.Rotation.x << ' '
			<< transform.Rotation.y << ' '
			<< transform.Rotation.z << ' '
			<< transform.Scale.x << ' '
			<< transform.Scale.y << ' '
			<< transform.Scale.z << '\n';
		file << "ENVIRONMENT_PATH " << std::quoted(scene.EnvironmentPath) << '\n';
		file << "ENVIRONMENT_INTENSITY " << scene.EnvironmentIntensity << '\n';
		file << "ENVIRONMENT_ROTATION " << scene.EnvironmentRotation << '\n';
		file << "BOUNCE_COUNT " << scene.BounceCount << '\n';
		file << "RENDER_SETTINGS "
			<< (scene.Settings.UseBvh ? 1 : 0) << ' '
			<< (scene.Settings.UseSahSplit ? 1 : 0) << ' '
			<< (scene.Settings.UseStochasticLights ? 1 : 0) << ' '
			<< (scene.Settings.UseRayQuery ? 1 : 0) << '\n';
		file.flush();

		if (!file)
			return Fail(errorMessage, "Scene file could not be written completely.");
		return true;
	}

	bool LoadSceneFile(
		const std::string& path,
		SceneDescription& scene,
		std::string& errorMessage)
	{
		errorMessage.clear();
		std::ifstream file(path);
		if (!file)
			return Fail(errorMessage, "Scene file could not be opened.");

		uint32_t version = 0;
		if (!ExpectLabel(file, "WALNUT_RAY_SCENE") ||
			!(file >> version) ||
			version < 1 || version > CurrentSceneFileVersion)
		{
			return Fail(errorMessage, "Scene header or version is invalid.");
		}

		SceneDescription loaded = scene;

		uint64_t sphereCount = 0;
		if (!ExpectLabel(file, "SPHERE_COUNT") || !(file >> sphereCount))
			return Fail(errorMessage, "Sphere count is missing or invalid.");
		if (sphereCount > MaxSphereCount)
			return Fail(errorMessage, "Scene exceeds the sphere capacity.");

		loaded.Spheres.clear();
		for (uint64_t index = 0; index < sphereCount; index++)
		{
			Sphere sphere{};
			if (!ExpectLabel(file, "SPHERE") ||
				!(file
					>> sphere.Center.x
					>> sphere.Center.y
					>> sphere.Center.z
					>> sphere.Radius
					>> sphere.Albedo.r
					>> sphere.Albedo.g
					>> sphere.Albedo.b
					>> sphere.Reflectivity
					>> sphere.Roughness))
			{
				return Fail(errorMessage, "Sphere data is missing or invalid.");
			}
			if (!IsFinite(sphere))
				return Fail(errorMessage, "Sphere data contains a non-finite number.");
			loaded.Spheres.push_back(sphere);
		}

		// Version 1 stored exactly one light and no count line.
		uint64_t lightCount = 1;
		if (version >= 2)
		{
			if (!ExpectLabel(file, "LIGHT_COUNT") || !(file >> lightCount))
				return Fail(errorMessage, "Light count is missing or invalid.");
			if (lightCount > MaxLightCount)
				return Fail(errorMessage, "Scene exceeds the light capacity.");
		}

		loaded.Lights.clear();
		for (uint64_t index = 0; index < lightCount; index++)
		{
			SphereLight light{};
			glm::vec2 legacySize{};
			if (!ExpectLabel(file, "LIGHT") ||
				!(file
					>> light.Position.x
					>> light.Position.y
					>> light.Position.z
					>> light.Color.r
					>> light.Color.g
					>> light.Color.b
					>> legacySize.x
					>> legacySize.y
					>> light.Intensity))
			{
				return Fail(errorMessage, "Light data is missing or invalid.");
			}
			// Before version 7 lights were rectangles; half their average side is
			// a stable stand-in radius.
			light.Radius = 0.25f * (legacySize.x + legacySize.y);
			if (!IsFinite(light))
				return Fail(errorMessage, "Light data contains a non-finite number.");
			loaded.Lights.push_back(light);
		}

		if (version >= 3)
		{
			Camera& camera = loaded.SceneCamera;
			if (!ExpectLabel(file, "CAMERA") ||
				!(file
					>> camera.Position.x
					>> camera.Position.y
					>> camera.Position.z
					>> camera.Yaw
					>> camera.Pitch
					>> camera.VerticalFov))
			{
				return Fail(errorMessage, "Camera data is missing or invalid.");
			}
			if (!IsFinite(camera))
				return Fail(errorMessage, "Camera data contains a non-finite number.");
		}

		if (version >= 4 && !ReadFloat(file, "EXPOSURE", loaded.Exposure))
			return Fail(errorMessage, "Exposure is missing or invalid.");

		// Files before version 5 keep the Legacy default, which preserves the
		// hybrid shading they were saved with instead of guessing a material.
		if (version >= 5)
		{
			if (!ReadMatchingCount(file, "MATERIAL_COUNT", sphereCount))
				return Fail(errorMessage, "Material count is missing or does not match the spheres.");
			for (Sphere& sphere : loaded.Spheres)
			{
				uint32_t materialType = 0;
				if (!ExpectLabel(file, "MATERIAL") ||
					!(file >> materialType) ||
					materialType > static_cast<uint32_t>(MaterialType::Dielectric))
				{
					return Fail(errorMessage, "Material type is missing or invalid.");
				}
				sphere.Type = static_cast<MaterialType>(materialType);
			}
		}

		if (version >= 6)
		{
			if (!ReadMatchingCount(file, "IOR_COUNT", sphereCount))
				return Fail(errorMessage, "IOR count is missing or does not match the spheres.");
			for (Sphere& sphere : loaded.Spheres)
			{
				if (!ReadFloat(file, "IOR", sphere.IndexOfRefraction))
					return Fail(errorMessage, "Index of refraction is missing or invalid.");
			}
		}

		if (version >= 7)
		{
			if (!ReadMatchingCount(file, "LIGHT_RADIUS_COUNT", lightCount))
				return Fail(errorMessage, "Light radius count is missing or does not match the lights.");
			for (SphereLight& light : loaded.Lights)
			{
				if (!ReadFloat(file, "LIGHT_RADIUS", light.Radius))
					return Fail(errorMessage, "Light radius is missing or invalid.");
			}
		}

		if (version >= 8)
		{
			// Versions 8 and 9 always named a model; from version 10 an empty
			// path stands for the built-in triangle.
			if (!ExpectLabel(file, "MODEL_PATH") ||
				!(file >> std::quoted(loaded.ModelPath)) ||
				(version < 10 && loaded.ModelPath.empty()))
			{
				return Fail(errorMessage, "Model path is missing or invalid.");
			}
			ModelTransform& transform = loaded.Transform;
			if (!ExpectLabel(file, "MODEL_TRANSFORM") ||
				!(file
					>> transform.Position.x
					>> transform.Position.y
					>> transform.Position.z
					>> transform.Rotation.x
					>> transform.Rotation.y
					>> transform.Rotation.z
					>> transform.Scale.x
					>> transform.Scale.y
					>> transform.Scale.z))
			{
				return Fail(errorMessage, "Model transform is missing or invalid.");
			}
			if (!IsFinite(transform))
				return Fail(errorMessage, "Model transform contains a non-finite number.");
		}

		if (version >= 9)
		{
			if (!ExpectLabel(file, "ENVIRONMENT_PATH") ||
				!(file >> std::quoted(loaded.EnvironmentPath)))
			{
				return Fail(errorMessage, "Environment path is missing or invalid.");
			}
			if (!ReadFloat(file, "ENVIRONMENT_INTENSITY", loaded.EnvironmentIntensity))
				return Fail(errorMessage, "Environment intensity is missing or invalid.");
			if (!ReadFloat(file, "ENVIRONMENT_ROTATION", loaded.EnvironmentRotation))
				return Fail(errorMessage, "Environment rotation is missing or invalid.");
		}

		if (version >= 10)
		{
			if (!ExpectLabel(file, "BOUNCE_COUNT") || !(file >> loaded.BounceCount))
				return Fail(errorMessage, "Bounce count is missing or invalid.");
			RenderSettings& settings = loaded.Settings;
			if (!ExpectLabel(file, "RENDER_SETTINGS") ||
				!ReadFlag(file, settings.UseBvh) ||
				!ReadFlag(file, settings.UseSahSplit) ||
				!ReadFlag(file, settings.UseStochasticLights) ||
				!ReadFlag(file, settings.UseRayQuery))
			{
				return Fail(errorMessage, "Render settings are missing or invalid.");
			}
		}

		std::string unexpectedData;
		if (file >> unexpectedData)
			return Fail(errorMessage, "Scene file contains unexpected trailing data.");

		for (Sphere& sphere : loaded.Spheres)
			sphere = SanitizeSphere(sphere);
		for (SphereLight& light : loaded.Lights)
			light = SanitizeSphereLight(light);
		loaded.SceneCamera = SanitizeCamera(loaded.SceneCamera);
		loaded.Exposure = SanitizeExposure(loaded.Exposure);
		loaded.BounceCount = SanitizeBounceCount(loaded.BounceCount);
		loaded.Transform = SanitizeModelTransform(loaded.Transform);
		loaded.EnvironmentIntensity =
			SanitizeEnvironmentIntensity(loaded.EnvironmentIntensity);
		loaded.EnvironmentRotation = WrapDegrees(loaded.EnvironmentRotation);

		scene = std::move(loaded);
		return true;
	}
}
