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
