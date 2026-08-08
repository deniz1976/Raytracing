#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Walnut/Input/Input.h"

#include "ComputeRenderer.h"

#include <glm/gtc/constants.hpp>

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
		UpdateCamera();
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
		const glm::vec3 forward = CalculateCameraForward();
		const glm::vec3 right = glm::normalize(glm::cross(
			forward,
			glm::vec3(0.0f, 1.0f, 0.0f)));

		glm::vec3 movement(0.0f);
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::W))
			movement += forward;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::S))
			movement -= forward;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::D))
			movement += right;
		if (Walnut::Input::IsKeyDown(Walnut::KeyCode::A))
			movement -= right;

		if (glm::dot(movement, movement) > 0.0f)
		{
			m_CameraPosition +=
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
				m_CameraYaw += mouseDelta.x * m_MouseSensitivity;
				m_CameraPitch -= mouseDelta.y * m_MouseSensitivity;
				m_CameraPitch = glm::clamp(m_CameraPitch, -89.0f, 89.0f);
				cameraChanged = true;
			}
		}

		if (cameraChanged)
			UpdateCamera();
	}

	virtual void OnUIRender() override
	{
		ImGui::Begin("Camera and Render Controls");

		bool cameraChanged = false;
		cameraChanged |= ImGui::DragFloat3(
			"Position",
			&m_CameraPosition.x,
			0.05f);
		cameraChanged |= ImGui::DragFloat(
			"Yaw",
			&m_CameraYaw,
			0.5f);
		cameraChanged |= ImGui::SliderFloat(
			"Pitch",
			&m_CameraPitch,
			-89.0f,
			89.0f);
		cameraChanged |= ImGui::SliderFloat(
			"Vertical FOV",
			&m_VerticalFov,
			20.0f,
			90.0f);

		if (ImGui::SliderFloat(
			"Exposure",
			&m_Exposure,
			0.1f,
			4.0f))
		{
			m_Renderer.SetExposure(m_Exposure);
		}

		if (ImGui::Button("Reset Camera"))
		{
			m_CameraPosition = { 0.0f, 0.0f, 3.0f };
			m_CameraYaw = 0.0f;
			m_CameraPitch = 0.0f;
			m_VerticalFov = 45.0f;
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
		ImGui::TextWrapped("WASD: Move | Mouse: Look | Esc: Release");

		if (cameraChanged)
			UpdateCamera();

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
				const uint32_t loadedSphereCount = m_Renderer.GetSphereCount();
				if (loadedSphereCount == 0)
					m_SelectedSphereIndex = 0;
				else if (m_SelectedSphereIndex >=
					static_cast<int>(loadedSphereCount))
				{
					m_SelectedSphereIndex =
						static_cast<int>(loadedSphereCount) - 1;
				}
			}
			else
			{
				m_SceneStatus = "Load failed: " + errorMessage;
			}
		}

		if (!m_SceneStatus.empty())
			ImGui::TextWrapped("%s", m_SceneStatus.c_str());

		uint32_t sphereCount = m_Renderer.GetSphereCount();
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
			sphereChanged |= ImGui::ColorEdit3(
				"Albedo",
				&sphere.Albedo.x);
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

			if (sphereChanged)
			{
				m_Renderer.SetSphere(
					static_cast<uint32_t>(m_SelectedSphereIndex),
					sphere);
			}
		}

		ImGui::End();

		ImGui::Begin("Light Controls");
		ComputeRenderer::AreaLight light = m_Renderer.GetAreaLight();
		bool lightChanged = false;
		lightChanged |= ImGui::DragFloat3(
			"Position",
			&light.Position.x,
			0.05f);
		lightChanged |= ImGui::ColorEdit3(
			"Color",
			&light.Color.x);
		lightChanged |= ImGui::DragFloat2(
			"Size",
			&light.Size.x,
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
			m_Renderer.SetAreaLight(light);

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

			ImGui::Image(
				(ImTextureID)m_Renderer.GetImageDescriptorSet(),
				ImVec2(
					static_cast<float>(m_Renderer.GetWidth()),
					static_cast<float>(m_Renderer.GetHeight())),
				ImVec2(0.0f, 1.0f),
				ImVec2(1.0f, 0.0f));
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
		ImGui::Text(
			"BVH nodes: %u (depth %u)",
			m_Renderer.GetBvhNodeCount(),
			m_Renderer.GetBvhDepth());

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
			"time even when the dispatch is much shorter. Turn the BVH off to "
			"compare the tree against testing every sphere.");
		ImGui::End();
	}

private:
	glm::vec3 CalculateCameraForward() const
	{
		const float yaw = glm::radians(m_CameraYaw);
		const float pitch = glm::radians(m_CameraPitch);

		glm::vec3 forward;
		forward.x = glm::cos(pitch) * glm::sin(yaw);
		forward.y = glm::sin(pitch);
		forward.z = -glm::cos(pitch) * glm::cos(yaw);
		return glm::normalize(forward);
	}

	void UpdateCamera()
	{
		m_Renderer.SetCamera(
			m_CameraPosition,
			CalculateCameraForward(),
			m_VerticalFov);
	}

	void SetMouseLookEnabled(bool enabled)
	{
		if (m_MouseLookEnabled == enabled)
			return;

		m_MouseLookEnabled = enabled;
		m_HasMousePosition = false;
		Walnut::Input::SetCursorMode(
			enabled
				? Walnut::CursorMode::Locked
				: Walnut::CursorMode::Normal);
	}

private:
	ComputeRenderer m_Renderer;
	glm::vec3 m_CameraPosition = { 0.0f, 0.0f, 3.0f };
	float m_CameraYaw = 0.0f;
	float m_CameraPitch = 0.0f;
	float m_VerticalFov = 45.0f;
	float m_Exposure = 1.0f;
	float m_CameraMoveSpeed = 3.0f;
	float m_MouseSensitivity = 0.1f;
	glm::vec2 m_LastMousePosition = { 0.0f, 0.0f };
	bool m_HasMousePosition = false;
	bool m_MouseLookEnabled = false;
	int m_SelectedSphereIndex = 0;
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
