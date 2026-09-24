#include "EnvironmentDistribution.h"

#include <cmath>

namespace RayScene
{
	namespace
	{
		constexpr double Pi = 3.14159265358979323846;
	}

	float EnvironmentTexelSolidAngle(uint32_t row, uint32_t width, uint32_t height)
	{
		const double halfRowAngle = Pi / (2.0 * height);
		const double rowCenterAngle = Pi * (2.0 * row + 1.0) / (2.0 * height);
		const double cosineDifference =
			2.0 * std::sin(rowCenterAngle) * std::sin(halfRowAngle);
		return static_cast<float>(2.0 * Pi / width * cosineDifference);
	}

	bool BuildEnvironmentDistribution(
		const float* rgbaPixels,
		uint32_t width,
		uint32_t height,
		std::vector<glm::vec4>& entries)
	{
		const size_t texelCount = static_cast<size_t>(width) * height;
		if (!rgbaPixels || texelCount == 0)
			return false;

		// The running sum is kept in double: in float, a quarter million small
		// increments stop registering once the total grows, which would give dim
		// texels a probability of exactly zero.
		std::vector<double> weights(texelCount);
		double totalWeight = 0.0;
		for (uint32_t y = 0; y < height; y++)
		{
			const double solidAngle = EnvironmentTexelSolidAngle(y, width, height);
			for (uint32_t x = 0; x < width; x++)
			{
				const size_t index = static_cast<size_t>(y) * width + x;
				const float* pixel = rgbaPixels + index * 4;
				double luminance =
					0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
				if (!std::isfinite(luminance) || luminance < 0.0)
					luminance = 0.0;
				weights[index] = luminance * solidAngle;
				totalWeight += weights[index];
			}
		}

		if (!(totalWeight > 0.0) || !std::isfinite(totalWeight))
			return false;

		entries.assign(texelCount, glm::vec4(0.0f));
		double cumulative = 0.0;
		for (size_t index = 0; index < texelCount; index++)
		{
			cumulative += weights[index];
			entries[index].x = static_cast<float>(cumulative / totalWeight);
			entries[index].y = static_cast<float>(weights[index] / totalWeight);
		}
		// Rounding can leave the last bound a hair below one, which would let a
		// random number land past every texel.
		entries.back().x = 1.0f;
		return true;
	}
}
