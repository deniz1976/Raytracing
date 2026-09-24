#include "ObjLoader.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace RayScene
{
	namespace
	{
		struct Material
		{
			glm::vec3 Albedo{ 1.0f };
			std::filesystem::path TexturePath;
		};

		using MaterialLibrary = std::unordered_map<std::string, Material>;

		// Index -1 means the corner does not reference this attribute.
		struct FaceCorner
		{
			int64_t Position = -1;
			int64_t TexCoord = -1;
			int64_t Normal = -1;
		};

		std::string LineError(const char* what, uint32_t lineNumber)
		{
			return std::string(what) + " at OBJ line " +
				std::to_string(lineNumber) + ".";
		}

		bool IsFiniteVector(const glm::vec3& value)
		{
			return std::isfinite(value.x) &&
				std::isfinite(value.y) &&
				std::isfinite(value.z);
		}

		// Turns one OBJ index into a zero based one. Positive OBJ indices start at
		// one; negative ones are relative to the end of the list, with -1 naming
		// the newest element. Zero, garbage and out of range values all fail.
		bool ResolveObjIndex(
			const std::string& text,
			size_t elementCount,
			int64_t& resolvedIndex)
		{
			int64_t objIndex = 0;
			try
			{
				size_t parsedCharacters = 0;
				objIndex = std::stoll(text, &parsedCharacters);
				if (parsedCharacters != text.size())
					return false;
			}
			catch (const std::exception&)
			{
				return false;
			}

			if (objIndex == 0)
				return false;

			resolvedIndex = objIndex < 0
				? static_cast<int64_t>(elementCount) + objIndex
				: objIndex - 1;
			return resolvedIndex >= 0 &&
				resolvedIndex < static_cast<int64_t>(elementCount);
		}

		// Splits "v", "v/vt", "v//vn" or "v/vt/vn" and resolves every part that is
		// present against the element counts read so far.
		bool ParseFaceCorner(
			const std::string& reference,
			size_t positionCount,
			size_t texCoordCount,
			size_t normalCount,
			FaceCorner& corner,
			const char*& error)
		{
			const size_t firstSlash = reference.find('/');
			const size_t secondSlash = firstSlash == std::string::npos
				? std::string::npos
				: reference.find('/', firstSlash + 1);

			if (!ResolveObjIndex(
				reference.substr(0, firstSlash), positionCount, corner.Position))
			{
				error = "Invalid or out of range face index";
				return false;
			}

			if (firstSlash != std::string::npos)
			{
				const size_t texCoordLength = secondSlash == std::string::npos
					? std::string::npos
					: secondSlash - firstSlash - 1;
				const std::string texCoordText =
					reference.substr(firstSlash + 1, texCoordLength);
				if (!texCoordText.empty() &&
					!ResolveObjIndex(texCoordText, texCoordCount, corner.TexCoord))
				{
					error = "Invalid or out of range texture index";
					return false;
				}
			}

			if (secondSlash != std::string::npos)
			{
				const std::string normalText = reference.substr(secondSlash + 1);
				if (!normalText.empty() &&
					!ResolveObjIndex(normalText, normalCount, corner.Normal))
				{
					error = "Invalid or out of range normal index";
					return false;
				}
			}

			return true;
		}

		// Only newmtl, Kd and map_Kd are understood; every other statement is
		// ignored so richer MTL files still load with their diffuse colours.
		bool LoadMaterialLibrary(
			const std::filesystem::path& materialPath,
			MaterialLibrary& materials,
			std::string& errorMessage)
		{
			std::ifstream file(materialPath);
			if (!file)
			{
				errorMessage = "MTL file could not be opened: " +
					materialPath.string();
				return false;
			}

			std::string line;
			std::string materialName;
			while (std::getline(file, line))
			{
				std::istringstream stream(line);
				std::string command;
				stream >> command;
				if (command == "newmtl")
				{
					stream >> materialName;
					materials[materialName] = Material{};
				}
				else if (command == "Kd" && !materialName.empty())
				{
					glm::vec3 albedo{};
					if (!(stream >> albedo.r >> albedo.g >> albedo.b) ||
						!IsFiniteVector(albedo))
					{
						errorMessage = "MTL diffuse color is invalid.";
						return false;
					}
					materials[materialName].Albedo =
						glm::clamp(albedo, glm::vec3(0.0f), glm::vec3(1.0f));
				}
				else if (command == "map_Kd" && !materialName.empty())
				{
					std::string textureName;
					std::getline(stream >> std::ws, textureName);
					if (!textureName.empty() && textureName.back() == '\r')
						textureName.pop_back();
					if (textureName.empty())
					{
						errorMessage = "MTL diffuse texture path is empty.";
						return false;
					}
					materials[materialName].TexturePath =
						materialPath.parent_path() / textureName;
				}
			}

			if (!file.eof())
			{
				errorMessage = "MTL file could not be read completely.";
				return false;
			}
			return true;
		}
	}

	bool LoadObjModel(
		const std::string& path,
		ObjModel& model,
		std::string& errorMessage)
	{
		errorMessage.clear();
		std::ifstream file(path);
		if (!file)
		{
			errorMessage = "OBJ file could not be opened.";
			return false;
		}

		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> normals;
		std::vector<glm::vec2> texCoords;
		MaterialLibrary materials;
		Material currentMaterial;
		currentMaterial.Albedo = { 0.8f, 0.65f, 0.15f };
		ObjModel loaded;
		std::string line;
		uint32_t lineNumber = 0;

		while (std::getline(file, line))
		{
			lineNumber++;
			std::istringstream lineStream(line);
			std::string command;
			lineStream >> command;
			if (command.empty() || command[0] == '#')
				continue;

			if (command == "v" || command == "vn")
			{
				glm::vec3 value{};
				const bool isNormal = command == "vn";
				if (!(lineStream >> value.x >> value.y >> value.z) ||
					!IsFiniteVector(value) ||
					(isNormal && glm::dot(value, value) <= 1e-12f))
				{
					errorMessage = LineError(
						isNormal ? "Invalid normal" : "Invalid vertex", lineNumber);
					return false;
				}
				if (isNormal)
					normals.push_back(glm::normalize(value));
				else
					positions.push_back(value);
				continue;
			}
			if (command == "vt")
			{
				glm::vec2 texCoord{};
				if (!(lineStream >> texCoord.x >> texCoord.y) ||
					!std::isfinite(texCoord.x) ||
					!std::isfinite(texCoord.y))
				{
					errorMessage = LineError("Invalid texture coordinate", lineNumber);
					return false;
				}
				texCoords.push_back(texCoord);
				continue;
			}
			if (command == "mtllib")
			{
				std::string libraryName;
				if (!(lineStream >> libraryName))
				{
					errorMessage = LineError("Material library is missing", lineNumber);
					return false;
				}
				if (!LoadMaterialLibrary(
					std::filesystem::path(path).parent_path() / libraryName,
					materials,
					errorMessage))
				{
					return false;
				}
				continue;
			}
			if (command == "usemtl")
			{
				std::string materialName;
				if (!(lineStream >> materialName))
				{
					errorMessage = LineError("Material name is missing", lineNumber);
					return false;
				}
				const auto material = materials.find(materialName);
				if (material == materials.end())
				{
					errorMessage = LineError(
						"OBJ references an unknown material", lineNumber);
					return false;
				}
				currentMaterial = material->second;
				continue;
			}
			if (command != "f")
				continue;

			std::vector<FaceCorner> corners;
			std::string reference;
			while (lineStream >> reference)
			{
				FaceCorner corner;
				const char* cornerError = nullptr;
				if (!ParseFaceCorner(
					reference,
					positions.size(),
					texCoords.size(),
					normals.size(),
					corner,
					cornerError))
				{
					errorMessage = LineError(cornerError, lineNumber);
					return false;
				}
				corners.push_back(corner);
			}

			if (corners.size() < 3)
			{
				errorMessage = LineError(
					"Face has fewer than three vertices", lineNumber);
				return false;
			}

			// A triangle fan turns (0, 1, 2, 3) into (0, 1, 2) and (0, 2, 3), which
			// is exact for triangles, quads and convex polygons.
			for (size_t fan = 1; fan + 1 < corners.size(); fan++)
			{
				if (loaded.Triangles.size() >= MaxTriangleCount)
				{
					errorMessage = "OBJ exceeds the triangle capacity.";
					return false;
				}

				const FaceCorner& a = corners[0];
				const FaceCorner& b = corners[fan];
				const FaceCorner& c = corners[fan + 1];
				Triangle triangle{
					positions[a.Position],
					positions[b.Position],
					positions[c.Position],
					currentMaterial.Albedo,
					0.0f,
					0.5f,
					MaterialType::Diffuse,
					1.5f
				};
				triangle.HasVertexNormals =
					a.Normal >= 0 && b.Normal >= 0 && c.Normal >= 0;
				if (triangle.HasVertexNormals)
				{
					triangle.Normal0 = normals[a.Normal];
					triangle.Normal1 = normals[b.Normal];
					triangle.Normal2 = normals[c.Normal];
				}
				triangle.HasTexCoords =
					a.TexCoord >= 0 && b.TexCoord >= 0 && c.TexCoord >= 0;
				if (triangle.HasTexCoords)
				{
					triangle.TexCoord0 = texCoords[a.TexCoord];
					triangle.TexCoord1 = texCoords[b.TexCoord];
					triangle.TexCoord2 = texCoords[c.TexCoord];
				}
				triangle.UsesImageTexture =
					triangle.HasTexCoords && !currentMaterial.TexturePath.empty();
				if (triangle.UsesImageTexture)
				{
					if (!loaded.DiffuseTexturePath.empty() &&
						loaded.DiffuseTexturePath != currentMaterial.TexturePath)
					{
						errorMessage =
							"OBJ uses more than one diffuse texture; this renderer supports one.";
						return false;
					}
					loaded.DiffuseTexturePath = currentMaterial.TexturePath;
				}
				loaded.Triangles.push_back(triangle);
			}
		}

		if (!file.eof())
		{
			errorMessage = "OBJ file could not be read completely.";
			return false;
		}
		if (positions.empty())
		{
			errorMessage = "OBJ contains no vertices.";
			return false;
		}
		if (loaded.Triangles.empty())
		{
			errorMessage = "OBJ contains no faces.";
			return false;
		}

		model = std::move(loaded);
		return true;
	}
}
