#pragma once

#include "SceneTypes.h"

#include <filesystem>
#include <string>
#include <vector>

namespace RayScene
{
	struct ObjModel
	{
		// Triangles in the model's own coordinate space; placement in the world
		// is applied later as a transform instead of being baked in here.
		std::vector<Triangle> Triangles;
		// Empty when no triangle samples an image texture.
		std::filesystem::path DiffuseTexturePath;
	};

	// Parses an OBJ file and the MTL libraries it references. Supported face
	// corners are v, v/vt, v//vn and v/vt/vn; polygons are fan triangulated. On
	// failure the model is left untouched and errorMessage names the problem.
	bool LoadObjModel(
		const std::string& path,
		ObjModel& model,
		std::string& errorMessage);
}
