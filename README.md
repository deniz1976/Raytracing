# Vulkan Compute Path Tracer

A learning-focused GPU path tracer built on the
[Walnut](https://github.com/StudioCherno/Walnut) application framework. Rays are
generated and shaded in a Vulkan compute shader. Triangle traversal can be switched
between a custom shader BVH and `VK_KHR_ray_query` hardware traversal.

The default render starts at 1600 x 900 and then follows the live ImGui viewport.

## Features

- Progressive path tracing with sub-pixel anti-aliasing
- Editable spheres with diffuse, GGX metal and absorbing dielectric materials
- Visible spherical area lights, soft shadows and multiple importance sampling
- Configurable bounce depth and Russian roulette path termination
- OBJ triangle models with transforms, triangle BVH acceleration and smooth normals
- MTL diffuse colors, UV coordinates and `map_Kd` image textures
- Radiance HDR environment maps with intensity and horizontal rotation controls
- Luminance-weighted HDR environment importance sampling with BSDF MIS
- ACES filmic tone mapping and standard linear-to-sRGB output conversion
- Scene save/load with backward-compatible format versions
- Free-look camera and live scene, light, render and performance panels
- CPU timings, Vulkan timestamp-based GPU timings and BVH statistics
- Live custom triangle BVH versus RTX ray-query traversal comparison

## Requirements

- Windows 10 or 11
- A GPU and driver with Vulkan 1.2, `VK_KHR_ray_query` and
  `VK_KHR_acceleration_structure` support
- Visual Studio 2022 or newer with **Desktop development with C++**
- [Vulkan SDK](https://vulkan.lunarg.com/sdk/home/windows) with the
  `VULKAN_SDK` environment variable set
- Git submodules cloned (`--recursive`)

The project is developed and validated on an NVIDIA RTX 3050. Ray query is a standard
Vulkan extension rather than an NVIDIA-only API, but compatible ray-tracing hardware
and drivers are required by this experimental version.

## Setup, Build and Run

Clone the repository with its submodules, then generate the Visual Studio solution:

```bat
git clone --recursive <repo-url>
cd Walnut
scripts\Setup.bat
```

For the shortest Debug workflow, run:

```bat
scripts\BuildAndRun.bat
```

The script locates MSBuild through Visual Studio Installer, builds `Debug|x64`, and
starts the executable with `WalnutApp` as its working directory so relative model,
shader and environment paths resolve correctly.

Alternatively, open `WalnutApp.sln`, select `WalnutApp` as the startup project and
run it from Visual Studio. The compute shader is compiled to SPIR-V before each app
build.

## Controls

| Input | Action |
| --- | --- |
| Enable Mouse Look | Capture the cursor without activating ImGui widgets |
| W / A / S / D | Move forward, left, backward and right |
| Q / E | Move down and up |
| Mouse | Look around while mouse look is enabled |
| Esc | Release the cursor |

The panels can add/remove spheres and spherical lights, edit materials, change camera
and render settings, load OBJ and HDR files, transform a model, save/load the scene,
and compare custom BVH, hardware ray query or light sampling strategies. The ray-query
toggle accelerates triangles; analytic spheres continue to use the custom sphere BVH.

## Asset Notes

- OBJ polygons are triangulated. Supported face corners are `v`, `v/vt`, `v//vn`
  and `v/vt/vn`.
- MTL support currently covers `newmtl`, `Kd` and `map_Kd`.
- One diffuse image texture is supported per loaded OBJ model.
- Environment maps must be equirectangular Radiance `.hdr` files.
- `WalnutApp/assets/environment/Studio.hdr` and the cube assets are small test
  assets included with the repository.
- The scene editor saves to
  `WalnutApp/assets/scenes/CurrentScene.local.scene`; local scene files are ignored
  by Git.

## Troubleshooting

- **`VULKAN_SDK is not set`**: install the Vulkan SDK, restart the terminal and run
  setup again.
- **Shader file cannot be opened**: start the app with `WalnutApp` as its working
  directory; `BuildAndRun.bat` does this automatically.
- **`glslc.exe` is missing**: confirm `%VULKAN_SDK%\Bin\glslc.exe` exists.
- **Validation-layer startup failure**: install the SDK runtime/layers or remove an
  externally forced `VK_INSTANCE_LAYERS` value for a normal run.
- **OBJ/HDR load failure**: the UI reports the rejected path and keeps the previous
  valid model or environment active.
- **Noisy image**: stop moving the camera and allow progressive samples to accumulate.
  More bounces, many lights and small bright HDR regions need more GPU work/samples.

## Project Layout

- `WalnutApp/src/ComputeRenderer.cpp` - Vulkan resources, descriptors, buffers,
  BVHs, asset loading, scene I/O and dispatch
- `WalnutApp/src/Shaders/RayTracing.comp` - intersections, materials, lighting,
  path sampling, accumulation and display transform
- `WalnutApp/src/WalnutApp.cpp` - ImGui panels and camera input
- `WalnutApp/assets/` - tracked example models, textures and environment
- `Walnut/` - Walnut window, Vulkan and ImGui framework code

## Current Scope

This is an educational renderer, not a production DCC renderer. Notable deliberate
limits include one loaded OBJ model, one diffuse model texture, no texture mipmaps,
no environment texture mipmaps and no `VK_KHR_ray_tracing_pipeline` shader stages.
The sphere and triangle BVHs are built on the CPU; the triangle path can instead build
a Vulkan BLAS/TLAS and traverse it from the same compute shader with ray queries.

## Third Party

[Dear ImGui](https://github.com/ocornut/imgui),
[GLFW](https://github.com/glfw/glfw), [GLM](https://github.com/g-truc/glm), and
[stb_image](https://github.com/nothings/stb). Walnut is by Studio Cherno and is MIT
licensed; see `LICENSE.txt`.
