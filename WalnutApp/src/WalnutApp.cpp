#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Walnut/Input/Input.h"

#include "ComputeRenderer.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <array>
#include <string>

namespace
{
	constexpr const char* SceneFilePath =
		"assets/scenes/CurrentScene.local.scene";
}

class ExampleLayer : public Walnut::Layer
{
public:
	virtual void OnAttach() override
	{
		m_Renderer.Init("assets/shaders/RayTracing.comp.spv", 1600, 900);
		std::string errorMessage;
		if (m_Renderer.LoadObj(m_ObjPath.data(), errorMessage))
		{
			ComputeRenderer::ModelTransform previewTransform;
			previewTransform.Position = { 0.0f, 0.0f, 1.5f };
			previewTransform.Rotation = { -10.0f, 25.0f, 0.0f };
			previewTransform.Scale = { 0.6f, 0.6f, 0.6f };
			m_Renderer.SetModelTransform(previewTransform);
			m_SceneStatus = "OBJ loaded: " +
				std::to_string(m_Renderer.GetTriangleCount()) +
				" triangles.";
		}
		else
		{
			m_SceneStatus = "OBJ load failed: " + errorMessage;
		}
		if (!m_Renderer.LoadEnvironmentMap(
			m_EnvironmentPath.data(), errorMessage))
			m_SceneStatus = "Initial HDR load failed: " + errorMessage;
	}

	virtual void OnDetach() override
	{
		SetMouseLookEnabled(false);
	}

	virtual void OnUpdate(float timeStep) override
	{
		if (!m_MouseLookEnabled)
			return;

		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::Escape))
		{
			SetMouseLookEnabled(false);
			return;
		}

		bool cameraChanged = false;
		ComputeRenderer::Camera camera = m_Renderer.GetCamera();
		// The renderer already derives this from the current yaw and pitch, so
		// the movement directions never disagree with the traced view.
		const glm::vec3 forward = m_Renderer.GetCameraForward();
		const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
		const glm::vec3 right = glm::normalize(glm::cross(
			forward,
			worldUp));

		glm::vec3 movement(0.0f);
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::W))
			movement += forward;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::S))
			movement -= forward;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::D))
			movement += right;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::A))
			movement -= right;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::E))
			movement += worldUp;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::Q))
			movement -= worldUp;

		if (glm::dot(movement, movement) > 0.0f)
		{
			camera.Position +=
				glm::normalize(movement) *
				m_CameraMoveSpeed *
				timeStep;
			cameraChanged = true;
		}

		const glm::vec2 mousePosition = Walnut::Input::GetMousePosition();
		if (!m_HasMousePosition)
		{
			m_LastMousePosition = mousePosition;
			m_HasMousePosition = true;
		}
		else
		{
			const glm::vec2 mouseDelta = mousePosition - m_LastMousePosition;
			m_LastMousePosition = mousePosition;

			if (glm::dot(mouseDelta, mouseDelta) > 0.0f)
			{
				camera.Yaw += mouseDelta.x * m_MouseSensitivity;
				camera.Pitch -= mouseDelta.y * m_MouseSensitivity;
				cameraChanged = true;
			}
		}

		// SetCamera clamps the pitch, so the look direction can never flip over
		// the top even though the mouse delta is unbounded.
		if (cameraChanged)
			m_Renderer.SetCamera(camera);
	}

	virtual void OnUIRender() override
	{
		ImGui::Begin("Camera and Render Controls");

		// Read back every frame instead of mirroring the values here, so a scene
		// load shows up in these controls without any extra bookkeeping.
		ComputeRenderer::Camera camera = m_Renderer.GetCamera();
		bool cameraChanged = false;
		cameraChanged |= ImGui::DragFloat3(
			"Position",
			&camera.Position.x,
			0.05f);
		cameraChanged |= ImGui::DragFloat(
			"Yaw",
			&camera.Yaw,
			0.5f);
		cameraChanged |= ImGui::SliderFloat(
			"Pitch",
			&camera.Pitch,
			-89.0f,
			89.0f);
		cameraChanged |= ImGui::SliderFloat(
			"Vertical FOV",
			&camera.VerticalFov,
			20.0f,
			90.0f);

		// Read back from the renderer for the same reason the camera is, so a
		// loaded scene moves this slider without any extra bookkeeping.
		float exposure = m_Renderer.GetExposure();
		if (ImGui::SliderFloat(
			"Exposure",
			&exposure,
			ComputeRenderer::MinExposure,
			ComputeRenderer::MaxExposure))
		{
			m_Renderer.SetExposure(exposure);
		}
		ImGui::TextWrapped(
			"Exposure scales linear HDR light before ACES tone mapping converts "
			"it to the display range.");

		int bounceCount = static_cast<int>(m_Renderer.GetBounceCount());
		if (ImGui::SliderInt(
			"Max Bounces",
			&bounceCount,
			static_cast<int>(ComputeRenderer::MinBounceCount),
			static_cast<int>(ComputeRenderer::MaxBounceCount)))
		{
			m_Renderer.SetBounceCount(static_cast<uint32_t>(bounceCount));
		}
		ImGui::TextWrapped(
			"More bounces allow longer reflection and refraction paths, but cost "
			"more GPU work per sample.");

		if (ImGui::Button("Reset Camera"))
		{
			camera = ComputeRenderer::Camera{};
			cameraChanged = true;
		}

		ImGui::SliderFloat(
			"Move Speed",
			&m_CameraMoveSpeed,
			0.5f,
			15.0f);
		if (ImGui::Button(
			m_MouseLookEnabled
				? "Disable Mouse Look"
				: "Enable Mouse Look"))
		{
			SetMouseLookEnabled(!m_MouseLookEnabled);
		}
		ImGui::TextWrapped(
			"WASD: Move | Q/E: Down/Up | Mouse: Look | Esc: Release");

		if (cameraChanged)
			m_Renderer.SetCamera(camera);

		ImGui::End();

		ImGui::Begin("Scene Controls");
		if (ImGui::Button("Save Scene"))
		{
			std::string errorMessage;
			if (m_Renderer.SaveScene(SceneFilePath, errorMessage))
				m_SceneStatus = "Scene saved.";
			else
				m_SceneStatus = "Save failed: " + errorMessage;
		}
		ImGui::SameLine();
		if (ImGui::Button("Load Scene"))
		{
			std::string errorMessage;
			if (m_Renderer.LoadScene(SceneFilePath, errorMessage))
			{
				m_SceneStatus = "Scene loaded.";
				m_ObjPath.fill('\0');
				const std::string& loadedModelPath = m_Renderer.GetModelPath();
				std::copy_n(
					loadedModelPath.data(),
					std::min(loadedModelPath.size(), m_ObjPath.size() - 1),
					m_ObjPath.data());
				const uint32_t loadedSphereCount = m_Renderer.GetSphereCount();
				if (loadedSphereCount == 0)
					m_SelectedSphereIndex = 0;
				else if (m_SelectedSphereIndex >=
					static_cast<int>(loadedSphereCount))
				{
					m_SelectedSphereIndex =
						static_cast<int>(loadedSphereCount) - 1;
				}

				const uint32_t loadedLightCount = m_Renderer.GetLightCount();
				if (loadedLightCount == 0)
					m_SelectedLightIndex = 0;
				else if (m_SelectedLightIndex >=
					static_cast<int>(loadedLightCount))
				{
					m_SelectedLightIndex =
						static_cast<int>(loadedLightCount) - 1;
				}
			}
			else
			{
				m_SceneStatus = "Load failed: " + errorMessage;
			}
		}

		if (!m_SceneStatus.empty())
			ImGui::TextWrapped("%s", m_SceneStatus.c_str());

		ImGui::Separator();
		ImGui::Text("Environment");
		ImGui::SetNextItemWidth(300.0f);
		ImGui::InputText(
			"HDR Path",
			m_EnvironmentPath.data(),
			m_EnvironmentPath.size());
		if (ImGui::Button("Load HDR Environment"))
		{
			std::string errorMessage;
			if (m_Renderer.LoadEnvironmentMap(
				m_EnvironmentPath.data(), errorMessage))
				m_SceneStatus = "HDR environment loaded.";
			else
				m_SceneStatus = "HDR load failed: " + errorMessage;
		}
		ImGui::SameLine();
		if (ImGui::Button("Use Procedural Sky"))
		{
			m_Renderer.ClearEnvironmentMap();
			m_SceneStatus = "Procedural sky enabled.";
		}
		float environmentIntensity = m_Renderer.GetEnvironmentIntensity();
		if (ImGui::SliderFloat(
			"Environment Intensity", &environmentIntensity, 0.0f, 20.0f))
			m_Renderer.SetEnvironmentIntensity(environmentIntensity);
		float environmentRotation = m_Renderer.GetEnvironmentRotation();
		if (ImGui::SliderFloat(
			"Environment Rotation", &environmentRotation, -180.0f, 180.0f))
			m_Renderer.SetEnvironmentRotation(environmentRotation);
		ImGui::TextWrapped(
			"HDR pixels become the background and indirect environment light. "
			"Rotation turns the panorama around the vertical axis.");

		uint32_t sphereCount = m_Renderer.GetSphereCount();
		ImGui::Text(
			"Triangles: %u / %u",
			m_Renderer.GetTriangleCount(),
			ComputeRenderer::MaxTriangleCount);
		ImGui::SetNextItemWidth(300.0f);
		ImGui::InputText(
			"OBJ Path",
			m_ObjPath.data(),
			m_ObjPath.size());
		if (ImGui::Button("Load OBJ"))
		{
			std::string errorMessage;
			if (m_Renderer.LoadObj(m_ObjPath.data(), errorMessage))
			{
				m_SceneStatus = "OBJ loaded: " +
					std::to_string(m_Renderer.GetTriangleCount()) +
					" triangles.";
			}
			else
			{
				m_SceneStatus = "OBJ load failed: " + errorMessage;
			}
		}
		ImGui::TextWrapped(
			"OBJ faces replace the current triangle model. Polygon faces are "
			"triangulated and use diffuse MTL colors. Vertex normals enable "
			"smooth shading, while UVs and map_Kd provide an image texture.");

		ComputeRenderer::ModelTransform modelTransform =
			m_Renderer.GetModelTransform();
		bool modelTransformChanged = false;
		modelTransformChanged |= ImGui::DragFloat3(
			"Model Position",
			&modelTransform.Position.x,
			0.05f);
		modelTransformChanged |= ImGui::DragFloat3(
			"Model Rotation",
			&modelTransform.Rotation.x,
			0.5f);
		modelTransformChanged |= ImGui::DragFloat3(
			"Model Scale",
			&modelTransform.Scale.x,
			0.01f,
			0.01f,
			100.0f);
		if (ImGui::Button("Reset Model Transform"))
		{
			modelTransform = {};
			modelTransformChanged = true;
		}
		if (modelTransformChanged)
			m_Renderer.SetModelTransform(modelTransform);
		ImGui::TextWrapped(
			"Rotation uses degrees. Scale is applied per axis and remains "
			"positive to preserve triangle orientation.");
		ImGui::Text(
			"Spheres: %u / %u",
			sphereCount,
			ComputeRenderer::MaxSphereCount);

		if (sphereCount < ComputeRenderer::MaxSphereCount &&
			ImGui::Button("Add Sphere"))
		{
			if (m_Renderer.AddSphere())
				m_SelectedSphereIndex =
					static_cast<int>(m_Renderer.GetSphereCount()) - 1;
		}

		sphereCount = m_Renderer.GetSphereCount();
		if (sphereCount > 0)
		{
			ImGui::SameLine();
			if (ImGui::Button("Remove Selected"))
			{
				m_Renderer.RemoveSphere(
					static_cast<uint32_t>(m_SelectedSphereIndex));
				sphereCount = m_Renderer.GetSphereCount();
				if (sphereCount == 0)
					m_SelectedSphereIndex = 0;
				else if (m_SelectedSphereIndex >= static_cast<int>(sphereCount))
					m_SelectedSphereIndex = static_cast<int>(sphereCount) - 1;
			}
		}

		if (sphereCount == 0)
		{
			ImGui::Text("No spheres in the scene.");
		}
		else
		{
			ImGui::SliderInt(
				"Sphere Index",
				&m_SelectedSphereIndex,
				0,
				static_cast<int>(sphereCount) - 1);

			ComputeRenderer::Sphere sphere = m_Renderer.GetSphere(
				static_cast<uint32_t>(m_SelectedSphereIndex));
			bool sphereChanged = false;
			int materialType = static_cast<int>(sphere.Type);
			if (ImGui::Combo(
				"Material",
				&materialType,
				"Legacy\0Diffuse\0Metal\0Dielectric\0"))
			{
				sphere.Type =
					static_cast<ComputeRenderer::MaterialType>(materialType);
				sphereChanged = true;
			}
			sphereChanged |= ImGui::DragFloat3(
				"Center",
				&sphere.Center.x,
				0.05f);
			sphereChanged |= ImGui::DragFloat(
				"Radius",
				&sphere.Radius,
				0.05f,
				0.05f,
				200.0f);
			const char* colorLabel =
				sphere.Type == ComputeRenderer::MaterialType::Dielectric
					? "Transmittance"
					: "Albedo";
			sphereChanged |= ImGui::ColorEdit3(
				colorLabel,
				&sphere.Albedo.x);

			if (sphere.Type == ComputeRenderer::MaterialType::Legacy)
			{
				ImGui::TextWrapped(
					"Legacy preserves the original diffuse and reflection blend.");
				sphereChanged |= ImGui::SliderFloat(
					"Reflectivity",
					&sphere.Reflectivity,
					0.0f,
					1.0f);
				sphereChanged |= ImGui::SliderFloat(
					"Roughness",
					&sphere.Roughness,
					0.0f,
					1.0f);
			}
			else if (sphere.Type == ComputeRenderer::MaterialType::Diffuse)
			{
				ImGui::TextWrapped(
					"Diffuse receives direct light and does not create a reflection ray.");
			}
			else if (sphere.Type == ComputeRenderer::MaterialType::Metal)
			{
				ImGui::TextWrapped(
					"Metal uses a GGX microfacet reflection model. Albedo is the "
					"normal-incidence reflection colour, roughness spreads the "
					"microfacet normals, and strength scales the reflected energy.");
				sphereChanged |= ImGui::SliderFloat(
					"Reflection Strength",
					&sphere.Reflectivity,
					0.0f,
					1.0f);
				sphereChanged |= ImGui::SliderFloat(
					"Roughness",
					&sphere.Roughness,
					0.0f,
					1.0f);
			}
			else
			{
				ImGui::TextWrapped(
					"Dielectric refracts through the surface and reflects more at "
					"grazing angles. Transmittance is the fraction of each colour "
					"that remains after one world unit inside the glass.");
				sphereChanged |= ImGui::SliderFloat(
					"Index of Refraction",
					&sphere.IndexOfRefraction,
					1.0f,
					2.5f);
			}

			if (sphereChanged)
			{
				m_Renderer.SetSphere(
					static_cast<uint32_t>(m_SelectedSphereIndex),
					sphere);
			}
		}

		ImGui::End();

		// Keep this window name stable because ImGui uses it to restore the saved
		// docking layout. The controls edit sphere lights even though the window
		// keeps its original, shorter title.
		ImGui::Begin("Light Controls");

		uint32_t lightCount = m_Renderer.GetLightCount();
		ImGui::Text(
			"Sphere Lights: %u / %u",
			lightCount,
			ComputeRenderer::MaxLightCount);

		if (lightCount < ComputeRenderer::MaxLightCount &&
			ImGui::Button("Add Light"))
		{
			if (m_Renderer.AddLight())
				m_SelectedLightIndex =
					static_cast<int>(m_Renderer.GetLightCount()) - 1;
		}

		lightCount = m_Renderer.GetLightCount();
		if (lightCount > 0)
		{
			ImGui::SameLine();
			if (ImGui::Button("Remove Selected Light"))
			{
				m_Renderer.RemoveLight(
					static_cast<uint32_t>(m_SelectedLightIndex));
				lightCount = m_Renderer.GetLightCount();
				if (lightCount == 0)
					m_SelectedLightIndex = 0;
				else if (m_SelectedLightIndex >= static_cast<int>(lightCount))
					m_SelectedLightIndex = static_cast<int>(lightCount) - 1;
			}
		}

		ImGui::Separator();
		ImGui::TextWrapped(
			"Each light is a visible glowing sphere. Radius controls its visible "
			"size and the softness of its shadows; intensity controls brightness.");

		if (lightCount == 0)
		{
			ImGui::TextWrapped(
				"No sphere lights in the scene. Diffuse surfaces can still receive "
				"light from the sky through indirect paths.");
		}
		else
		{
			ImGui::SliderInt(
				"Light Index",
				&m_SelectedLightIndex,
				0,
				static_cast<int>(lightCount) - 1);

			ComputeRenderer::SphereLight light = m_Renderer.GetLight(
				static_cast<uint32_t>(m_SelectedLightIndex));
			bool lightChanged = false;
			lightChanged |= ImGui::DragFloat3(
				"Position",
				&light.Position.x,
				0.05f);
			lightChanged |= ImGui::ColorEdit3(
				"Color",
				&light.Color.x);
			lightChanged |= ImGui::DragFloat(
				"Radius",
				&light.Radius,
				0.05f,
				0.05f,
				20.0f);
			lightChanged |= ImGui::DragFloat(
				"Intensity",
				&light.Intensity,
				0.1f,
				0.0f,
				100.0f);

			if (lightChanged)
			{
				m_Renderer.SetLight(
					static_cast<uint32_t>(m_SelectedLightIndex),
					light);
			}
		}

		ImGui::End();

		ImGui::Begin("Compute Shader Output");

		const ImVec2 viewportSize = ImGui::GetContentRegionAvail();
		if (viewportSize.x >= 16.0f && viewportSize.y >= 16.0f)
		{
			const uint32_t viewportWidth =
				static_cast<uint32_t>(viewportSize.x);
			const uint32_t viewportHeight =
				static_cast<uint32_t>(viewportSize.y);
			m_Renderer.Resize(viewportWidth, viewportHeight);
			m_Renderer.Render();

			// The shader already writes row 0 as the top of the scene, so the
			// default UVs show it upright. Flipping them here as well would
			// mirror the image a second time.
			ImGui::Image(
				(ImTextureID)m_Renderer.GetImageDescriptorSet(),
				ImVec2(
					static_cast<float>(m_Renderer.GetWidth()),
					static_cast<float>(m_Renderer.GetHeight())));
		}
		else
		{
			ImGui::Text("Viewport is too small to render.");
		}
		ImGui::End();

		// Drawn after the render call so the numbers describe the frame that
		// was just traced instead of the previous one.
		ImGui::Begin("Performance");
		const float framesPerSecond = ImGui::GetIO().Framerate;
		ImGui::Text("FPS: %.1f", framesPerSecond);
		ImGui::Text(
			"Frame time: %.3f ms",
			framesPerSecond > 0.0f ? 1000.0f / framesPerSecond : 0.0f);
		ImGui::Text(
			"CPU render time: %.3f ms",
			m_Renderer.GetCpuRenderTimeMs());

		if (m_Renderer.AreGpuTimestampsSupported())
		{
			ImGui::Text(
				"GPU compute time: %.3f ms",
				m_Renderer.GetGpuComputeTimeMs());
		}
		else
		{
			ImGui::Text("GPU compute time: timestamps not supported");
		}

		ImGui::Separator();
		bool bvhEnabled = m_Renderer.IsBvhEnabled();
		// The tree only changes how the spheres are searched, never the image,
		// so toggling it keeps the accumulated samples and the two modes can be
		// compared without waiting for the picture to converge again.
		if (ImGui::Checkbox("Use BVH", &bvhEnabled))
			m_Renderer.SetBvhEnabled(bvhEnabled);

		bool sahSplitEnabled = m_Renderer.IsSahSplitEnabled();
		// Same reasoning as the BVH toggle: a different split rearranges the tree
		// but cannot change which sphere a ray meets first, so the samples stand.
		if (ImGui::Checkbox("SAH split (off: median split)", &sahSplitEnabled))
			m_Renderer.SetSahSplitEnabled(sahSplitEnabled);

		ImGui::Text(
			"BVH nodes: %u (depth %u)",
			m_Renderer.GetBvhNodeCount(),
			m_Renderer.GetBvhDepth());
		ImGui::Text(
			"Triangle BVH nodes: %u (depth %u)",
			m_Renderer.GetTriangleBvhNodeCount(),
			m_Renderer.GetTriangleBvhDepth());
		ImGui::Text(
			"Estimated sphere tests per ray: %.2f",
			m_Renderer.GetBvhCost());
		ImGui::Text(
			"BVH build time: %.3f ms",
			m_Renderer.GetBvhBuildTimeMs());

		bool stochasticLights = m_Renderer.AreStochasticLightsEnabled();
		// Unlike the BVH toggle this one changes what a single sample is worth,
		// so the renderer restarts the accumulation instead of keeping it.
		if (ImGui::Checkbox("Sample one light per hit", &stochasticLights))
			m_Renderer.SetStochasticLightsEnabled(stochasticLights);
		ImGui::Text(
			"Shadow rays per hit: %u",
			stochasticLights
				? (m_Renderer.GetLightCount() > 0 ? 1u : 0u)
				: m_Renderer.GetLightCount());

		ImGui::Separator();
		ImGui::Text(
			"Resolution: %u x %u",
			m_Renderer.GetWidth(),
			m_Renderer.GetHeight());
		ImGui::Text(
			"Soft shadows: %u samples per pixel",
			m_Renderer.GetFrameIndex());
		ImGui::TextWrapped(
			"GPU compute time is the dispatch alone, measured with timestamp "
			"queries. CPU render time also covers command buffer setup, "
			"submission and the fence wait. Because compute and present share "
			"one queue and the swapchain is vsync limited, that wait absorbs "
			"the wait for the display, so CPU render time stays near the frame "
			"time even when the dispatch is much shorter.");
		ImGui::TextWrapped(
			"Turn the BVH off to compare both trees against testing every sphere "
			"and triangle. The triangle tree uses a balanced median split and up "
			"to four triangles per leaf. "
			"The split heuristic decides where each range of spheres is cut in "
			"two: the surface area heuristic cuts where the two child boxes are "
			"cheapest to trace, while the median split cuts where the sphere "
			"count is even. Estimated sphere tests per ray is what the heuristic "
			"itself predicts for the finished tree, so it compares two trees over "
			"the same scene immediately, without waiting for a timing average to "
			"settle. Compare it against the sphere count, which is what testing "
			"every sphere would cost. The two splits agree on an evenly spread "
			"scene and part company on a clumped one, because only the heuristic "
			"can see that a box packed tightly around a cluster is one a ray "
			"usually misses. Neither changes the image: within a tree, the near "
			"child is always visited first, so a surface found early shrinks the "
			"distance the box test prunes against and the far box behind it is "
			"rejected without ever being opened.");
		ImGui::TextWrapped(
			"Sampling one light "
			"per hit keeps the shadow ray cost flat as lights are added and "
			"converges to the same image, but each sample is noisier, so more "
			"frames are needed. The light is picked in proportion to its "
			"brightness and colour, so uneven scenes lose far less to that noise "
			"than even ones: with one light dominating it needs about twice the "
			"frames for the same result while each frame is three times cheaper, "
			"but with eight equal lights it needs about fifty times the frames. "
			"It also helps while the camera moves, and it is what keeps the cost "
			"flat as a scene gains many more lights.");
		ImGui::End();
	}

private:
	void SetMouseLookEnabled(bool enabled)
	{
		if (m_MouseLookEnabled == enabled)
			return;

		m_MouseLookEnabled = enabled;
		m_HasMousePosition = false;
		ImGuiIO& io = ImGui::GetIO();
		if (enabled)
			io.ConfigFlags |= ImGuiConfigFlags_NoMouse;
		else
			io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;
		Walnut::Input::SetCursorMode(
			enabled
				? Walnut::CursorMode::Locked
				: Walnut::CursorMode::Normal);
	}

private:
	std::array<char, 260> m_ObjPath{ "assets/models/Cube.obj" };
	std::array<char, 260> m_EnvironmentPath{ "assets/environment/Studio.hdr" };
	ComputeRenderer m_Renderer;
	float m_CameraMoveSpeed = 3.0f;
	float m_MouseSensitivity = 0.1f;
	glm::vec2 m_LastMousePosition = { 0.0f, 0.0f };
	bool m_HasMousePosition = false;
	bool m_MouseLookEnabled = false;
	int m_SelectedSphereIndex = 0;
	int m_SelectedLightIndex = 0;
	std::string m_SceneStatus;
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "GPU Ray Tracing";
	spec.Width = 1600;
	spec.Height = 900;

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<ExampleLayer>();
	app->SetMenubarCallback([app]()
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Exit"))
			{
				app->Close();
			}
			ImGui::EndMenu();
		}
	});
	return app;
}
