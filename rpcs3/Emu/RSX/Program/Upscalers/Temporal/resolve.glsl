R"(
#version 450

layout(set = 0, binding = 0) uniform sampler2D CurrentTexture;
layout(set = 0, binding = 1) uniform sampler2D PreviousTexture;
layout(set = 0, binding = 2) uniform sampler2D MotionTexture;
layout(set = 0, binding = 3) uniform sampler2D DepthTexture;
layout(set = 0, binding = 4, rgba8) uniform writeonly image2D OutputTexture;

layout(push_constant) uniform PushConstants
{
	vec4 InputOutputSize; // input width/height, output width/height
	vec4 Flags;           // reset, has-depth, unused, unused
} params;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main()
{
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	vec2 output_size = params.InputOutputSize.zw;
	if (pixel.x >= int(output_size.x) || pixel.y >= int(output_size.y))
	{
		return;
	}

	vec2 uv = (vec2(pixel) + vec2(0.5)) / max(output_size, vec2(1.0));
	vec4 current = texture(CurrentTexture, uv);
	vec4 motion = texture(MotionTexture, uv);
	vec2 history_uv = clamp(uv + motion.xy / max(params.InputOutputSize.xy, vec2(1.0)), vec2(0.0001), vec2(0.9999));
	vec4 history = texture(PreviousTexture, history_uv);

	float history_weight = params.Flags.x > 0.5 ? 0.0 : clamp(motion.z, 0.0, 1.0) * 0.85;
	if (params.Flags.y > 0.5)
	{
		float current_depth = texture(DepthTexture, uv).r;
		float history_depth = texture(DepthTexture, history_uv).r;
		float depth_delta = abs(current_depth - history_depth);
		// The emulator has no guaranteed depth convention, so use depth only as
		// a disocclusion rejection signal rather than trying to linearize it.
		history_weight *= 1.0 - smoothstep(0.015, 0.15, depth_delta);
	}

	vec4 result = mix(current, history, history_weight);
	imageStore(OutputTexture, pixel, vec4(result.rgb, current.a));
}
)"
