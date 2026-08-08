# GPU Ray Tracer

A real-time ray tracer that runs entirely in a Vulkan compute shader, with a Dear ImGui
interface built on the [Walnut](https://github.com/TheCherno/Walnut) application framework.
The compute shader writes into an image that is displayed and resized live inside an ImGui
viewport.

## Features

- Sphere scene stored in a shader storage buffer (up to 64 spheres)
- Area light with stochastic soft shadows
- Rough reflections with up to 3 ray bounces
- Progressive sample accumulation that resets when the scene or camera changes
- Reinhard tone mapping with an exposure control
- Free-look camera (WASD + mouse) and ImGui panels for camera, spheres and light
- Render resolution follows the viewport size
- Performance panel with FPS, CPU render time and GPU compute time from Vulkan timestamp queries
- Scene save/load to `WalnutApp/assets/scenes/CurrentScene.local.scene`

## Requirements

- [Visual Studio 2022](https://visualstudio.com)
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) (`glslc` is used to compile the shader)

## Build and Run

```
git clone --recursive <repo-url>
scripts/Setup.bat
```

Open `WalnutApp.sln` and run the `WalnutApp` project. `src/Shaders/RayTracing.comp` is
compiled to SPIR-V as a pre-build step, so editing the shader only requires a rebuild.

## Controls

| Input | Action |
| --- | --- |
| Enable Mouse Look | Capture the cursor for free-look |
| W / A / S / D | Move the camera |
| Mouse | Look around |
| Esc | Release the cursor |

Camera position, field of view, sphere materials and light parameters can also be edited
directly from the ImGui panels.

## Layout

- `WalnutApp/src/ComputeRenderer.cpp` - Vulkan resources, pipeline, scene buffer, scene file I/O
- `WalnutApp/src/Shaders/RayTracing.comp` - the ray tracing compute shader
- `WalnutApp/src/WalnutApp.cpp` - ImGui layer, camera input and UI panels
- `Walnut/` - application framework (window, Vulkan setup, ImGui integration)

## Third Party

[Dear ImGui](https://github.com/ocornut/imgui), [GLFW](https://github.com/glfw/glfw),
[GLM](https://github.com/g-truc/glm), [stb_image](https://github.com/nothings/stb).
Walnut is by Studio Cherno and is MIT licensed; see `LICENSE.txt`.
