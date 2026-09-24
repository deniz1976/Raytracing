#pragma once

#include "SceneTypes.h"

#include <string>
#include <vector>

namespace RayScene
{
	struct RenderSettings
	{
		bool UseBvh = true;
		bool UseSahSplit = true;
		bool UseStochasticLights = false;
		bool UseRayQuery = false;
	};

	// Everything a scene file stores. Paths are kept as written; resolving and
	// loading the files they name is the renderer's job.
	struct SceneDescription
	{
		std::vector<Sphere> Spheres;
		std::vector<SphereLight> Lights;
		Camera SceneCamera;
		float Exposure = 1.0f;
		uint32_t BounceCount = 3;
		// Empty means no OBJ is loaded and the built-in triangle is shown.
		std::string ModelPath;
		ModelTransform Transform;
		// Empty means the procedural sky.
		std::string EnvironmentPath;
		float EnvironmentIntensity = 1.0f;
		float EnvironmentRotation = 0.0f;
		RenderSettings Settings;
	};

	constexpr uint32_t CurrentSceneFileVersion = 10;

	bool SaveSceneFile(
		const std::string& path,
		const SceneDescription& scene,
		std::string& errorMessage);

	// Reads a scene written by any format version. Fields that an older version
	// does not store keep whatever the caller put into scene beforehand, so
	// passing in the current scene makes old files leave those settings alone.
	// Nothing is written into scene unless the whole file parses.
	bool LoadSceneFile(
		const std::string& path,
		SceneDescription& scene,
		std::string& errorMessage);
}
