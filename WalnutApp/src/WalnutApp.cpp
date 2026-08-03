#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "ComputeRenderer.h"

#include <glm/gtc/constants.hpp>

class ExampleLayer : public Walnut::Layer
{
public:
	virtual void OnAttach() override
	{
		m_Renderer.Init("assets/shaders/RayTracing.comp.spv", 1600, 900);
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

		if (cameraChanged)
			UpdateCamera();

		ImGui::End();

		ImGui::Begin("Scene Controls");
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

		m_Renderer.Render();

		ImGui::Begin("Compute Shader Output");
		ImGui::Text(
			"Soft shadows: %u samples per pixel",
			m_Renderer.GetFrameIndex());
		ImGui::Image(
			(ImTextureID)m_Renderer.GetImageDescriptorSet(),
			ImVec2((float)m_Renderer.GetWidth(), (float)m_Renderer.GetHeight()),
			ImVec2(0.0f, 1.0f),
			ImVec2(1.0f, 0.0f));
		ImGui::End();
	}

private:
	void UpdateCamera()
	{
		const float yaw = glm::radians(m_CameraYaw);
		const float pitch = glm::radians(m_CameraPitch);

		glm::vec3 forward;
		forward.x = glm::cos(pitch) * glm::sin(yaw);
		forward.y = glm::sin(pitch);
		forward.z = -glm::cos(pitch) * glm::cos(yaw);

		m_Renderer.SetCamera(
			m_CameraPosition,
			glm::normalize(forward),
			m_VerticalFov);
	}

private:
	ComputeRenderer m_Renderer;
	glm::vec3 m_CameraPosition = { 0.0f, 0.0f, 3.0f };
	float m_CameraYaw = 0.0f;
	float m_CameraPitch = 0.0f;
	float m_VerticalFov = 45.0f;
	float m_Exposure = 1.0f;
	int m_SelectedSphereIndex = 0;
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
