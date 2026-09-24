// Unit tests for the GPU-independent scene modules: OBJ parsing, scene files,
// BVH construction and the environment sampling distribution. They need no
// Vulkan device, so they run anywhere the sources compile.

#include "Scene/Bvh.h"
#include "Scene/EnvironmentDistribution.h"
#include "Scene/ObjLoader.h"
#include "Scene/SceneFile.h"
#include "Shaders/ShaderInterface.h"

#include <algorithm>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace RayScene;

namespace
{
	int s_Failures = 0;
	int s_Checks = 0;

	void Check(bool condition, const char* expression, const char* file, int line)
	{
		s_Checks++;
		if (condition)
			return;
		s_Failures++;
		std::printf("  FAILED %s:%d: %s\n", file, line, expression);
	}

#define CHECK(expression) Check((expression), #expression, __FILE__, __LINE__)

	std::filesystem::path TempDirectory()
	{
		const std::filesystem::path directory =
			std::filesystem::temp_directory_path() / "raytracing_scene_tests";
		std::filesystem::create_directories(directory);
		return directory;
	}

	std::string WriteFile(const std::string& name, const std::string& contents)
	{
		const std::filesystem::path path = TempDirectory() / name;
		std::ofstream(path) << contents;
		return path.string();
	}

	bool NearlyEqual(float a, float b, float tolerance = 1e-5f)
	{
		return std::fabs(a - b) <= tolerance;
	}

	// ------------------------------------------------------------------ OBJ

	void TestObjQuadIsTriangulated()
	{
		const std::string path = WriteFile("quad.obj",
			"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n");
		ObjModel model;
		std::string error;
		CHECK(LoadObjModel(path, model, error));
		CHECK(model.Triangles.size() == 2);
		CHECK(model.Triangles[1].Vertex1 == glm::vec3(1, 1, 0));
		CHECK(model.Triangles[1].Vertex2 == glm::vec3(0, 1, 0));
	}

	void TestObjAllCornerForms()
	{
		const std::string path = WriteFile("corners.obj",
			"v 0 0 0\nv 1 0 0\nv 0 1 0\n"
			"vt 0 0\nvt 1 0\nvt 0 1\n"
			"vn 0 0 2\n"
			"f 1/1/1 2/2/1 3/3/1\n"
			"f 1//1 2//1 3//1\n"
			"f 1/1 2/2 3/3\n"
			"f -3 -2 -1\n");
		ObjModel model;
		std::string error;
		CHECK(LoadObjModel(path, model, error));
		CHECK(model.Triangles.size() == 4);
		CHECK(model.Triangles[0].HasVertexNormals && model.Triangles[0].HasTexCoords);
		CHECK(model.Triangles[0].Normal0 == glm::vec3(0, 0, 1));
		CHECK(model.Triangles[1].HasVertexNormals && !model.Triangles[1].HasTexCoords);
		CHECK(!model.Triangles[2].HasVertexNormals && model.Triangles[2].HasTexCoords);
		CHECK(model.Triangles[3].Vertex0 == glm::vec3(0, 0, 0));
		CHECK(model.Triangles[3].Vertex2 == glm::vec3(0, 1, 0));
	}

	void TestObjRejectsBadIndices()
	{
		const char* badFaces[] = {
			"f 1 2 4\n", "f 0 1 2\n", "f 1 2 x\n", "f 1/5 2/1 3/1\n",
			"f 1//9 2//1 3//1\n", "f 1 2\n", "f -4 1 2\n" };
		for (const char* face : badFaces)
		{
			const std::string path = WriteFile("bad.obj",
				std::string("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\n") + face);
			ObjModel model;
			model.Triangles.resize(7);
			std::string error;
			CHECK(!LoadObjModel(path, model, error));
			CHECK(!error.empty());
			// A failed load must not touch the output.
			CHECK(model.Triangles.size() == 7);
		}
	}

	void TestObjMaterials()
	{
		WriteFile("materials.mtl",
			"newmtl red\nKd 1 0 0\n"
			"newmtl textured\nKd 0.5 0.5 0.5\nmap_Kd my texture.png\r\n");
		const std::string path = WriteFile("materials.obj",
			"mtllib materials.mtl\n"
			"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\n"
			"usemtl red\nf 1 2 3\n"
			"usemtl textured\nf 1/1 2/1 3/1\n");
		ObjModel model;
		std::string error;
		CHECK(LoadObjModel(path, model, error));
		CHECK(model.Triangles.size() == 2);
		CHECK(model.Triangles[0].Albedo == glm::vec3(1, 0, 0));
		CHECK(!model.Triangles[0].UsesImageTexture);
		CHECK(model.Triangles[1].UsesImageTexture);
		CHECK(model.DiffuseTexturePath.filename() == "my texture.png");

		const std::string unknown = WriteFile("unknown.obj",
			"mtllib materials.mtl\nusemtl missing\n");
		CHECK(!LoadObjModel(unknown, model, error));
	}

	// ----------------------------------------------------------- scene file

	SceneDescription MakeScene()
	{
		SceneDescription scene;
		scene.Spheres = DefaultSpheres();
		scene.Lights = DefaultLights();
		scene.SceneCamera.Position = { 1.0f, 2.0f, 3.0f };
		scene.SceneCamera.Yaw = 33.0f;
		scene.Exposure = 1.7f;
		scene.BounceCount = 6;
		scene.ModelPath = "assets/models/My Model.obj";
		scene.Transform.Scale = { 2.0f, 0.5f, 1.0f };
		scene.EnvironmentPath = "assets/environment/Studio.hdr";
		scene.EnvironmentIntensity = 3.5f;
		scene.EnvironmentRotation = -90.0f;
		scene.Settings.UseSahSplit = false;
		scene.Settings.UseStochasticLights = true;
		scene.Settings.UseRayQuery = true;
		return scene;
	}

	void TestSceneRoundTrip()
	{
		const std::string path = (TempDirectory() / "round.scene").string();
		const SceneDescription saved = MakeScene();
		std::string error;
		CHECK(SaveSceneFile(path, saved, error));

		SceneDescription loaded;
		CHECK(LoadSceneFile(path, loaded, error));
		CHECK(loaded.Spheres.size() == saved.Spheres.size());
		CHECK(loaded.Spheres[2].Type == MaterialType::Dielectric);
		CHECK(loaded.Lights.size() == saved.Lights.size());
		CHECK(loaded.Lights[0].Radius == saved.Lights[0].Radius);
		CHECK(loaded.SceneCamera.Position == saved.SceneCamera.Position);
		CHECK(loaded.Exposure == saved.Exposure);
		CHECK(loaded.BounceCount == 6);
		CHECK(loaded.ModelPath == saved.ModelPath);
		CHECK(loaded.Transform.Scale == saved.Transform.Scale);
		// The environment path used to be dropped on save.
		CHECK(loaded.EnvironmentPath == saved.EnvironmentPath);
		CHECK(loaded.EnvironmentIntensity == 3.5f);
		CHECK(loaded.EnvironmentRotation == -90.0f);
		CHECK(!loaded.Settings.UseSahSplit);
		CHECK(loaded.Settings.UseStochasticLights);
		CHECK(loaded.Settings.UseRayQuery);
	}

	void TestSceneWithoutModelRoundTrips()
	{
		const std::string path = (TempDirectory() / "nomodel.scene").string();
		SceneDescription saved = MakeScene();
		saved.ModelPath.clear();
		saved.EnvironmentPath.clear();
		std::string error;
		CHECK(SaveSceneFile(path, saved, error));
		SceneDescription loaded = MakeScene();
		CHECK(LoadSceneFile(path, loaded, error));
		CHECK(loaded.ModelPath.empty());
		CHECK(loaded.EnvironmentPath.empty());
	}

	void TestOldVersionKeepsCurrentSettings()
	{
		const std::string path = WriteFile("v1.scene",
			"WALNUT_RAY_SCENE 1\nSPHERE_COUNT 1\n"
			"SPHERE 0 0 0 1 0.5 0.5 0.5 0.2 0.3\n"
			"LIGHT 0 5 0 1 1 1 2 2 10\n");
		SceneDescription scene = MakeScene();
		std::string error;
		CHECK(LoadSceneFile(path, scene, error));
		CHECK(scene.Spheres.size() == 1);
		CHECK(scene.Spheres[0].Type == MaterialType::Legacy);
		CHECK(scene.Lights.size() == 1);
		CHECK(NearlyEqual(scene.Lights[0].Radius, 1.0f));
		// Nothing after the lights exists in version 1.
		CHECK(scene.SceneCamera.Yaw == 33.0f);
		CHECK(scene.ModelPath == "assets/models/My Model.obj");
		CHECK(scene.BounceCount == 6);
	}

	void TestSceneLoadIsAllOrNothing()
	{
		const std::string path = WriteFile("broken.scene",
			"WALNUT_RAY_SCENE 4\nSPHERE_COUNT 0\nLIGHT_COUNT 0\n"
			"CAMERA 9 9 9 0 0 45\nEXPOSURE nan\n");
		SceneDescription scene = MakeScene();
		std::string error;
		CHECK(!LoadSceneFile(path, scene, error));
		CHECK(!error.empty());
		CHECK(scene.Spheres.size() == DefaultSpheres().size());
		CHECK(scene.SceneCamera.Position == glm::vec3(1, 2, 3));

		const std::string trailing = WriteFile("trailing.scene",
			"WALNUT_RAY_SCENE 2\nSPHERE_COUNT 0\nLIGHT_COUNT 0\nEXTRA\n");
		CHECK(!LoadSceneFile(trailing, scene, error));
	}

	void TestSceneValuesAreSanitized()
	{
		const std::string path = (TempDirectory() / "clamp.scene").string();
		SceneDescription saved = MakeScene();
		saved.EnvironmentIntensity = 80.0f;
		saved.EnvironmentRotation = 540.0f;
		saved.BounceCount = 99;
		saved.Exposure = 50.0f;
		std::string error;
		CHECK(SaveSceneFile(path, saved, error));
		SceneDescription loaded;
		CHECK(LoadSceneFile(path, loaded, error));
		CHECK(loaded.EnvironmentIntensity == MaxEnvironmentIntensity);
		CHECK(loaded.EnvironmentRotation == -180.0f);
		CHECK(loaded.BounceCount == MaxBounceCount);
		CHECK(loaded.Exposure == MaxExposure);
	}

	void TestWrapDegrees()
	{
		CHECK(WrapDegrees(0.0f) == 0.0f);
		CHECK(WrapDegrees(190.0f) == -170.0f);
		CHECK(WrapDegrees(-190.0f) == 170.0f);
		CHECK(WrapDegrees(180.0f) == -180.0f);
		CHECK(WrapDegrees(720.0f + 45.0f) == 45.0f);
		CHECK(WrapDegrees(NAN) == 0.0f);
	}

	// ------------------------------------------------------------------ BVH

	bool BoxContains(const BvhNode& node, const BvhBounds& bounds)
	{
		return glm::all(glm::lessThanEqual(glm::vec3(node.BoundsMin), bounds.Min)) &&
			glm::all(glm::greaterThanEqual(glm::vec3(node.BoundsMax), bounds.Max));
	}

	// Walks the finished tree and checks the invariants the shader relies on.
	void ValidateTree(
		const std::vector<BvhPrimitive>& primitives,
		const BvhBuildResult& tree,
		const BvhBuildSettings& settings)
	{
		std::vector<uint32_t> seen(primitives.size(), 0);
		uint32_t maxDepth = 0;
		std::function<void(uint32_t, uint32_t)> visit =
			[&](uint32_t nodeIndex, uint32_t depth)
		{
			maxDepth = std::max(maxDepth, depth);
			const BvhNode& node = tree.Nodes[nodeIndex];
			if (node.Links.z > 0)
			{
				for (uint32_t offset = 0; offset < node.Links.z; offset++)
				{
					const uint32_t primitive =
						tree.PrimitiveOrder[node.Links.x + offset];
					seen[primitive]++;
					CHECK(BoxContains(node, primitives[primitive].Bounds));
				}
				return;
			}
			CHECK(node.Links.x < tree.Nodes.size() && node.Links.y < tree.Nodes.size());
			visit(node.Links.x, depth + 1);
			visit(node.Links.y, depth + 1);
		};
		visit(0, 1);

		bool everyPrimitiveOnce = true;
		for (uint32_t count : seen)
			everyPrimitiveOnce &= count == 1;
		CHECK(everyPrimitiveOnce);
		CHECK(maxDepth == tree.Depth);
		CHECK(tree.Depth <= settings.MaxDepth);
		CHECK(tree.Depth < BVH_STACK_SIZE);
		CHECK(tree.Nodes.size() <= 2 * primitives.size());
	}

	std::vector<BvhPrimitive> RandomSpheres(uint32_t count, uint32_t seed, bool clumped)
	{
		std::mt19937 random(seed);
		std::uniform_real_distribution<float> position(-50.0f, 50.0f);
		std::uniform_real_distribution<float> radius(0.1f, 2.0f);
		std::vector<BvhPrimitive> primitives;
		for (uint32_t index = 0; index < count; index++)
		{
			glm::vec3 center(position(random), position(random), position(random));
			if (clumped && index % 4 != 0)
				center *= 0.01f;
			const float r = radius(random);
			BvhPrimitive primitive;
			primitive.Bounds.Grow(center - glm::vec3(r));
			primitive.Bounds.Grow(center + glm::vec3(r));
			primitive.Centroid = center;
			primitives.push_back(primitive);
		}
		return primitives;
	}

	void TestBvhInvariants()
	{
		for (bool useSah : { true, false })
		{
			for (bool clumped : { false, true })
			{
				BvhBuildSettings settings;
				settings.UseSah = useSah;
				const auto primitives = RandomSpheres(512, 7, clumped);
				ValidateTree(primitives, BuildBvh(primitives, settings), settings);
			}
		}

		// Coincident centroids leave no plane to split on.
		std::vector<BvhPrimitive> stacked(40);
		for (BvhPrimitive& primitive : stacked)
		{
			primitive.Bounds.Grow(glm::vec3(-1.0f));
			primitive.Bounds.Grow(glm::vec3(1.0f));
			primitive.Centroid = glm::vec3(0.0f);
		}
		BvhBuildSettings settings;
		ValidateTree(stacked, BuildBvh(stacked, settings), settings);

		BvhBuildResult empty = BuildBvh({}, settings);
		CHECK(empty.Nodes.empty());
		CHECK(empty.PrimitiveOrder.empty());
	}

	void TestSahBeatsMedianOnClumpedScene()
	{
		const auto primitives = RandomSpheres(512, 11, true);
		BvhBuildSettings sah;
		BvhBuildSettings median;
		median.UseSah = false;
		CHECK(BuildBvh(primitives, sah).Cost < BuildBvh(primitives, median).Cost);
	}

	// -------------------------------------------------- environment sampling

	void TestEnvironmentSolidAnglesCoverSphere()
	{
		const uint32_t width = 64;
		const uint32_t height = 32;
		double total = 0.0;
		for (uint32_t row = 0; row < height; row++)
			total += EnvironmentTexelSolidAngle(row, width, height) * width;
		CHECK(std::fabs(total - 4.0 * 3.14159265358979) < 1e-4);
	}

	void TestEnvironmentDistribution()
	{
		const uint32_t width = 8;
		const uint32_t height = 4;
		std::vector<float> pixels(width * height * 4, 0.0f);
		// One bright texel in an equator row, the rest dim.
		for (size_t index = 0; index < width * height; index++)
			pixels[index * 4 + 1] = 0.001f;
		pixels[(1 * width + 3) * 4 + 1] = 1000.0f;

		std::vector<glm::vec4> entries;
		CHECK(BuildEnvironmentDistribution(pixels.data(), width, height, entries));
		CHECK(entries.size() == width * height);
		CHECK(entries.back().x == 1.0f);
		float previous = 0.0f;
		float probabilitySum = 0.0f;
		bool monotonic = true;
		bool dimTexelsReachable = true;
		for (const glm::vec4& entry : entries)
		{
			monotonic &= entry.x >= previous;
			previous = entry.x;
			probabilitySum += entry.y;
			dimTexelsReachable &= entry.y > 0.0f;
		}
		CHECK(monotonic);
		CHECK(dimTexelsReachable);
		CHECK(NearlyEqual(probabilitySum, 1.0f, 1e-4f));
		CHECK(entries[1 * width + 3].y > 0.99f);

		std::vector<float> black(width * height * 4, 0.0f);
		CHECK(!BuildEnvironmentDistribution(black.data(), width, height, entries));
	}
}

int main()
{
	const std::pair<const char*, void (*)()> tests[] = {
		{ "ObjQuadIsTriangulated", TestObjQuadIsTriangulated },
		{ "ObjAllCornerForms", TestObjAllCornerForms },
		{ "ObjRejectsBadIndices", TestObjRejectsBadIndices },
		{ "ObjMaterials", TestObjMaterials },
		{ "SceneRoundTrip", TestSceneRoundTrip },
		{ "SceneWithoutModelRoundTrips", TestSceneWithoutModelRoundTrips },
		{ "OldVersionKeepsCurrentSettings", TestOldVersionKeepsCurrentSettings },
		{ "SceneLoadIsAllOrNothing", TestSceneLoadIsAllOrNothing },
		{ "SceneValuesAreSanitized", TestSceneValuesAreSanitized },
		{ "WrapDegrees", TestWrapDegrees },
		{ "BvhInvariants", TestBvhInvariants },
		{ "SahBeatsMedianOnClumpedScene", TestSahBeatsMedianOnClumpedScene },
		{ "EnvironmentSolidAnglesCoverSphere", TestEnvironmentSolidAnglesCoverSphere },
		{ "EnvironmentDistribution", TestEnvironmentDistribution },
	};

	for (const auto& [name, test] : tests)
	{
		const int failuresBefore = s_Failures;
		test();
		std::printf("%s %s\n", s_Failures == failuresBefore ? "[pass]" : "[FAIL]", name);
	}

	std::printf("\n%d checks, %d failed\n", s_Checks, s_Failures);
	return s_Failures == 0 ? 0 : 1;
}
