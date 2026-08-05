R"(
#version 450

// A component-wise 3x3 median removes isolated false vectors while preserving
// coherent camera/object motion. The raw field remains available for metadata
// and diagnostics; Streamline receives the filtered R16G16 result.

layout(set = 0, binding = 0) uniform sampler2D InputMotion;
layout(set = 0, binding = 1, rg16f) uniform writeonly image2D OutputMotion;

layout(push_constant) uniform PushConstants
{
	vec4 InputSize;
} params;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

float median9(inout float values[9])
{
	for (int i = 0; i < 5; ++i)
	{
		int minimum = i;
		for (int j = i + 1; j < 9; ++j)
		{
			if (values[j] < values[minimum])
			{
				minimum = j;
			}
		}
		float value = values[i];
		values[i] = values[minimum];
		values[minimum] = value;
	}
	return values[4];
}

void main()
{
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 input_size = ivec2(params.InputSize.xy);
	if (pixel.x >= input_size.x || pixel.y >= input_size.y)
	{
		return;
	}

	float x[9];
	float y[9];
	int index = 0;
	for (int oy = -1; oy <= 1; ++oy)
	{
		for (int ox = -1; ox <= 1; ++ox)
		{
			vec2 value = texelFetch(InputMotion, clamp(pixel + ivec2(ox, oy), ivec2(0), input_size - ivec2(1)), 0).rg;
			x[index] = value.x;
			y[index] = value.y;
			++index;
		}
	}

	imageStore(OutputMotion, pixel, vec4(median9(x), median9(y), 0.0, 0.0));
}
)"
