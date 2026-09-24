#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace RayScene
{
	// The solid angle one texel of an equirectangular map covers. Row y spans the
	// polar band [pi*y/H, pi*(y+1)/H], whose area is the difference of the two
	// cosines; the product-of-sines form of that difference avoids cancellation
	// for the thin rows near the poles. The shader uses the same formula, and it
	// also samples uniformly by solid angle inside a texel, so the density it
	// samples with is exactly the density it divides by.
	float EnvironmentTexelSolidAngle(uint32_t row, uint32_t width, uint32_t height);

	// Builds the luminance x solid angle distribution the shader importance
	// samples. Entry.x is the running CDF and Entry.y the probability of that
	// texel. Returns false when no texel carries positive luminance.
	bool BuildEnvironmentDistribution(
		const float* rgbaPixels,
		uint32_t width,
		uint32_t height,
		std::vector<glm::vec4>& entries);
}
