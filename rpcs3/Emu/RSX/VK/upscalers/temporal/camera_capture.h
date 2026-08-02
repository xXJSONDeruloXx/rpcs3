#pragma once

#include "util/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace vk::temporal_camera
{
	using matrix = std::array<float, 16>;

	inline void load_matrix(const std::array<u32[4], 512>& constants, u32 slot, matrix& result)
	{
		for (u32 row = 0; row < 4; ++row)
		{
			for (u32 column = 0; column < 4; ++column)
			{
				float value;
				const u32 raw = constants[slot + row][column];
				std::memcpy(&value, &raw, sizeof(value));
				result[row * 4 + column] = value;
			}
		}
	}

	inline bool is_orthonormal_view(const matrix& value)
	{
		for (const float element : value)
		{
			if (!std::isfinite(element))
			{
				return false;
			}
		}

		if (std::abs(value[12]) > 1e-3f || std::abs(value[13]) > 1e-3f ||
			std::abs(value[14]) > 1e-3f || std::abs(value[15] - 1.f) > 1e-3f)
		{
			return false;
		}

		for (u32 row = 0; row < 3; ++row)
		{
			const float length = value[row * 4] * value[row * 4] +
				value[row * 4 + 1] * value[row * 4 + 1] +
				value[row * 4 + 2] * value[row * 4 + 2];
			if (std::abs(length - 1.f) > 0.02f)
			{
				return false;
			}
		}

		for (u32 a = 0; a < 3; ++a)
		{
			for (u32 b = a + 1; b < 3; ++b)
			{
				const float dot = value[a * 4] * value[b * 4] +
					value[a * 4 + 1] * value[b * 4 + 1] +
					value[a * 4 + 2] * value[b * 4 + 2];
				if (std::abs(dot) > 0.02f)
				{
					return false;
				}
			}
		}

		return true;
	}

	inline bool is_perspective_projection(const matrix& value)
	{
		return value[0] > 0.05f && std::abs(value[1]) < 1e-4f && std::abs(value[2]) < 1e-4f && std::abs(value[3]) < 1e-4f &&
			std::abs(value[4]) < 1e-4f && value[5] > 0.05f && std::abs(value[6]) < 1e-4f && std::abs(value[7]) < 1e-4f &&
			std::abs(value[8]) < 1e-4f && std::abs(value[9]) < 1e-4f &&
			std::abs(value[12]) < 1e-4f && std::abs(value[13]) < 1e-4f &&
			std::abs(std::abs(value[14]) - 1.f) < 0.01f && std::abs(value[15]) < 1e-3f;
	}

	inline bool is_square_projection(const matrix& value)
	{
		return value[5] > 0.05f && std::abs(std::abs(value[0] / value[5]) - 1.f) < 0.2f;
	}

	inline bool product_matches(const matrix& projection, const matrix& view, const matrix& view_projection)
	{
		for (u32 row = 0; row < 4; ++row)
		{
			for (u32 column = 0; column < 4; ++column)
			{
				float expected = 0.f;
				for (u32 k = 0; k < 4; ++k)
				{
					expected += projection[row * 4 + k] * view[k * 4 + column];
				}

				const float tolerance = std::max(0.02f, std::abs(expected) * 0.01f);
				if (std::abs(view_projection[row * 4 + column] - expected) > tolerance)
				{
					return false;
				}
			}
		}

		return true;
	}

	// Find the same contiguous View/Proj/ViewProj tuple used by the temporal
	// motion pass. Keeping this structural validation in one header lets the
	// draw path gate jitter to camera-like geometry without accepting UI,
	// shadow-map, or arbitrary constant blocks.
	inline bool capture(const std::array<u32[4], 512>& constants, float expected_aspect, matrix& result)
	{
		float best_aspect_error = std::numeric_limits<float>::max();
		bool found = false;

		for (u32 slot = 0; slot + 11 < constants.size(); ++slot)
		{
			matrix view{};
			matrix projection{};
			matrix view_projection{};
			load_matrix(constants, slot, view);
			load_matrix(constants, slot + 4, projection);
			load_matrix(constants, slot + 8, view_projection);

			if (!is_orthonormal_view(view) || !is_perspective_projection(projection) ||
				is_square_projection(projection) || !product_matches(projection, view, view_projection))
			{
				continue;
			}

			const float projection_aspect = std::abs(projection[0] / projection[5]);
			if (!std::isfinite(projection_aspect) || projection_aspect <= 0.f)
			{
				continue;
			}

			const float aspect_error = std::min(std::abs(projection_aspect - expected_aspect),
				std::abs(1.f / projection_aspect - expected_aspect));
			if (!found || aspect_error < best_aspect_error)
			{
				found = true;
				best_aspect_error = aspect_error;
				result = view_projection;
			}
		}

		return found;
	}

	inline bool has_camera_constants(const std::array<u32[4], 512>& constants, float expected_aspect = 1.f)
	{
		matrix unused{};
		return capture(constants, expected_aspect, unused);
	}
}
