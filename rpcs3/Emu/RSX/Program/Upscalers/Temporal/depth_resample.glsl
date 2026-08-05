R"(
#version 450

// Normalize a guest depth attachment to the color input grid. RPCS3 can see
// depth allocations whose dimensions lag the presented color during dynamic
// resolution changes; a nearest texel fetch preserves the guest's depth
// convention and avoids handing Streamline an unrelated auxiliary surface.

layout(set = 0, binding = 0) uniform sampler2D InputDepth;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D OutputDepth;

layout(push_constant) uniform PushConstants
{
	vec4 InputOutputSize; // source width/height, destination width/height
} params;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main()
{
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 output_size = ivec2(params.InputOutputSize.zw);
	if (pixel.x >= output_size.x || pixel.y >= output_size.y)
	{
		return;
	}

	ivec2 input_size = ivec2(params.InputOutputSize.xy);
	// A depth allocation can lag or lead the active color extent during a
	// dynamic-resolution step. The guest's active rectangle is top-left
	// anchored: when the allocation is larger, do not sample the stale margin;
	// when it is smaller, stretch the complete available depth into the color
	// grid. This mirrors Beast's active-subrect policy without inventing a
	// camera viewport that RSX never exposed to the host.
	vec2 active_size = min(vec2(input_size), vec2(output_size));
	vec2 source_position = (vec2(pixel) + vec2(0.5)) * active_size / vec2(output_size);
	ivec2 source_pixel = clamp(ivec2(source_position), ivec2(0), input_size - ivec2(1));
	imageStore(OutputDepth, pixel, vec4(texelFetch(InputDepth, source_pixel, 0).r, 0.0, 0.0, 1.0));
}
)"
