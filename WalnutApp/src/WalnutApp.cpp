#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "ComputeRenderer.h"

class ExampleLayer : public Walnut::Layer
{
public:
	virtual void OnAttach() override
	{
		m_Renderer.Init("assets/shaders/RayTracing.comp.spv", 1600, 900);
	}

	virtual void OnUIRender() override
	{
		m_Renderer.Render();

		ImGui::Begin("Compute Shader Output");
		ImGui::Text("Reflections with iterative ray bounces");
		ImGui::Image(
			(ImTextureID)m_Renderer.GetImageDescriptorSet(),
			ImVec2((float)m_Renderer.GetWidth(), (float)m_Renderer.GetHeight()),
			ImVec2(0.0f, 1.0f),
			ImVec2(1.0f, 0.0f));
		ImGui::End();
	}

private:
	ComputeRenderer m_Renderer;
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
