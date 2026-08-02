R"(
#version 450

// RPCS3's emulated RSX does not expose a native velocity attachment. When a
// structurally validated guest camera pair and depth buffer are available,
// this pass reprojects static-world motion; otherwise it builds a conservative
// field from consecutive resolved color frames. The separate resource can be
// consumed by a vendor temporal upscaler or the local fallback.

layout(set = 0, binding = 0) uniform sampler2D CurrentTexture;
layout(set = 0, binding = 1) uniform sampler2D PreviousTexture;
layout(set = 0, binding = 2) uniform sampler2D DepthTexture;
layout(set = 0, binding = 3, rgba16f) uniform writeonly image2D MotionTexture;

layout(push_constant) uniform PushConstants
{
	vec4 InputOutputSize; // input width/height, output width/height
	vec4 Flags;           // reset, has-depth, camera-pair, depth-inverted
	vec4 CameraClipToPrevious[4]; // row-major clip-space transform, when camera-pair is set
} params;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

float luminance(vec3 value)
{
	return dot(value, vec3(0.2126, 0.7152, 0.0722));
}

float sample_luma(sampler2D texture_sampler, vec2 uv)
{
	return luminance(texture(texture_sampler, clamp(uv, vec2(0.0001), vec2(0.9999))).rgb);
}

void main()
{
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 input_size = ivec2(params.InputOutputSize.xy);
	if (pixel.x >= input_size.x || pixel.y >= input_size.y)
	{
		return;
	}

	vec2 texel = 1.0 / max(params.InputOutputSize.xy, vec2(1.0));
	vec2 uv = (vec2(pixel) + vec2(0.5)) * texel;

	float best_error = 1e20;
	vec2 best_offset = vec2(0.0);
	float confidence = 0.0;
	bool camera_motion_valid = params.Flags.z > 0.5 && params.Flags.y > 0.5;
	if (camera_motion_valid)
	{
		float depth = texture(DepthTexture, uv).r;
		camera_motion_valid = depth > 0.0 && depth < 1.0;
		if (!camera_motion_valid)
		{
			// Cleared/far-plane depth has no stable world position. Let the
			// color estimator handle sky and disocclusion pixels instead.
			confidence = 0.0;
		}
	}
	if (camera_motion_valid)
	{
		float depth = texture(DepthTexture, uv).r;
		float ndc_z = params.Flags.w > 0.5 ? depth * 2.0 - 1.0 : depth;
		vec4 current_clip = vec4(uv * 2.0 - 1.0, ndc_z, 1.0);
		vec4 previous_clip = vec4(
			dot(CameraClipToPrevious[0], current_clip),
			dot(CameraClipToPrevious[1], current_clip),
			dot(CameraClipToPrevious[2], current_clip),
			dot(CameraClipToPrevious[3], current_clip));

		if (abs(previous_clip.w) > 1e-5)
		{
			vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
			if (all(greaterThanEqual(previous_uv, vec2(0.0))) && all(lessThanEqual(previous_uv, vec2(1.0))))
			{
				// DLSS and the local resolve use render-resolution pixels pointing
				// toward the source location in the previous frame.
				best_offset = (previous_uv - uv) * params.InputOutputSize.xy;
				confidence = 0.9;
			}
			else
			{
				camera_motion_valid = false;
			}
		}
		else
		{
			camera_motion_valid = false;
		}
	}

	// A small patch search is considerably less brittle than a raw color
	// difference on emulator output: it tolerates post-process noise and gives
	// us a usable field for both DLSS and the local temporal resolve. It is the
	// fallback when no structurally validated camera pair is available.
	if (!camera_motion_valid)
	{
		for (int oy = -2; oy <= 2; ++oy)
		{
			for (int ox = -2; ox <= 2; ++ox)
			{
				vec2 offset = vec2(ox, oy);
				float error = 0.0;
				for (int py = -1; py <= 1; ++py)
				{
					for (int px = -1; px <= 1; ++px)
					{
						vec2 sample_uv = uv + (offset + vec2(px, py)) * texel;
						float previous_luma = sample_luma(PreviousTexture, sample_uv);
						float current_patch = sample_luma(CurrentTexture, uv + vec2(px, py) * texel);
						error += abs(current_patch - previous_luma);
					}
				}

				if (error < best_error)
				{
					best_error = error;
					best_offset = offset;
				}
			}
		}

		confidence = 1.0 - clamp(best_error * 4.0, 0.0, 1.0);
	}

	best_offset = clamp(best_offset, vec2(-32.0), vec2(32.0));

	if (params.Flags.y < 0.5)
	{
		// No depth is still a supported operating mode, but marking this field
		// as lower confidence makes the history resolve less aggressive.
		confidence *= 0.75;
	}

	float depth = params.Flags.y > 0.5 ? texture(DepthTexture, uv).r : 0.0;
	imageStore(MotionTexture, pixel, vec4(best_offset, confidence, depth));
}
)"
