// Shared by the C++ renderer and RayTracing.comp. Only preprocessor definitions
// live here, because this header is compiled both as C++ and as GLSL; keeping the
// numbers in one place means a binding can never be renumbered on one side only.
#ifndef RAYTRACING_SHADER_INTERFACE_H
#define RAYTRACING_SHADER_INTERFACE_H

#define WORKGROUP_SIZE 8

#define BINDING_OUTPUT_IMAGE 0
#define BINDING_ACCUMULATION_IMAGE 1
#define BINDING_SPHERES 2
#define BINDING_SPHERE_BVH 3
#define BINDING_LIGHTS 4
#define BINDING_TRIANGLES 5
#define BINDING_TRIANGLE_BVH 6
#define BINDING_MODEL_TEXTURE 7
#define BINDING_ENVIRONMENT_TEXTURE 8
#define BINDING_ENVIRONMENT_DISTRIBUTION 9
#define BINDING_MODEL_TRANSFORM 10
// Only present in the ray query shader variant, so it is numbered last and the
// fallback layout is simply the same list without it.
#define BINDING_TRIANGLE_TLAS 11

// The traversal stack holds one pending sibling per tree level, so every BVH
// the CPU builds must stay shallower than this.
#define BVH_STACK_SIZE 32

// Bits of the Flags push constant.
#define RENDER_FLAG_USE_BVH 1u
#define RENDER_FLAG_STOCHASTIC_LIGHTS 2u
#define RENDER_FLAG_RAY_QUERY 4u
#define RENDER_FLAG_ENVIRONMENT_MAP 8u

#define MATERIAL_LEGACY 0u
#define MATERIAL_DIFFUSE 1u
#define MATERIAL_METAL 2u
#define MATERIAL_DIELECTRIC 3u

#endif
