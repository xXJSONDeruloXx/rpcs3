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
layout(set = 0, binding = 3, rg16f) uniform writeonly image2D MotionTexture;
layout(set = 0, binding = 4, rgba16f) uniform writeonly image2D MotionMetadataTexture;
layout(std430, set = 0, binding = 5) buffer SceneChange
{
	uint ChangedCount;
	uint MotionCount;
	uint Reserved[7];
} scene_change;
layout(set = 0, binding = 6) uniform sampler2D PreviousMotionMetadata;
layout(set = 0, binding = 7, r8) uniform writeonly image2D BiasTexture;

layout(push_constant) uniform PushConstants
{
	vec4 InputOutputSize; // input width/height, output width/height
	vec4 Flags;           // reset, has-depth, camera-pair, depth-inverted
	vec4 JitterDelta;     // current jitter minus previous jitter, in render pixels
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

float patch_error(sampler2D current_texture, sampler2D previous_texture, vec2 current_uv,
	vec2 previous_uv, vec2 texel)
{
	float error = 0.0;
	for (int py = -1; py <= 1; ++py)
	{
		for (int px = -1; px <= 1; ++px)
		{
			vec2 patch_offset = vec2(px, py) * texel;
			error += abs(sample_luma(current_texture, current_uv + patch_offset) -
				sample_luma(previous_texture, previous_uv + patch_offset));
		}
	}
	return error;
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
	vec2 predicted_previous_uv = uv;
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
			dot(params.CameraClipToPrevious[0], current_clip),
			dot(params.CameraClipToPrevious[1], current_clip),
			dot(params.CameraClipToPrevious[2], current_clip),
			dot(params.CameraClipToPrevious[3], current_clip));

		if (abs(previous_clip.w) > 1e-5)
		{
			predicted_previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
			if (all(greaterThanEqual(predicted_previous_uv, vec2(0.0))) && all(lessThanEqual(predicted_previous_uv, vec2(1.0))))
			{
				// DLSS and the local resolve use render-resolution pixels pointing
				// toward the source location in the previous frame.
				best_offset = (predicted_previous_uv - uv) * params.InputOutputSize.xy;
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

	if (camera_motion_valid)
	{
		// Camera reprojection is exact for static world pixels. Search a small
		// residual around that prediction as a cheap dynamic-object path: the
		// guest has no native velocity attachment, so an animated object must be
		// allowed to disagree with the camera field instead of inheriting it.
		const float camera_error = patch_error(CurrentTexture, PreviousTexture, uv, predicted_previous_uv, texel);
		float residual_error = 1e20;
		vec2 residual = vec2(0.0);
		for (int oy = -2; oy <= 2; ++oy)
		{
			for (int ox = -2; ox <= 2; ++ox)
			{
				vec2 candidate = vec2(ox, oy);
				float error = patch_error(CurrentTexture, PreviousTexture, uv,
					predicted_previous_uv + candidate * texel, texel);
				if (error < residual_error)
				{
					residual_error = error;
					residual = candidate;
				}
			}
		}

		best_error = residual_error;
		if (residual_error + 0.002 < camera_error)
		{
			best_offset += residual;
			confidence = 0.25 + 0.65 * (1.0 - clamp(residual_error * 4.0, 0.0, 1.0));
		}
		else
		{
			confidence = 0.9;
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
		// Color flow observes the apparent shift caused by camera jitter. Add the
		// opposite of that apparent motion so the field handed to DLSS is clean.
		best_offset += params.JitterDelta.xy;
	}

	best_offset = clamp(best_offset, vec2(-32.0), vec2(32.0));

	if (params.Flags.y < 0.5)
	{
		// No depth is still a supported operating mode, but marking this field
		// as lower confidence makes the history resolve less aggressive.
		confidence *= 0.75;
	}

	// Keep a cheap one-frame-late scene-cut meter alongside the field. A large
	// local temporal change that the patch model cannot explain is a useful cut
	// signal; confident vectors over one pixel identify ordinary camera/object
	// motion and keep the CPU-side hysteresis from resetting during gameplay.
	float temporal_change = patch_error(CurrentTexture, PreviousTexture, uv, uv, texel) / 9.0;
	if (temporal_change > 0.08 && confidence < 0.4)
	{
		atomicAdd(scene_change.ChangedCount, 1u);
	}
	if (confidence > 0.5 && dot(best_offset, best_offset) > 1.0)
	{
		atomicAdd(scene_change.MotionCount, 1u);
	}

	float motion_bias = 0.0;
	if (params.JitterDelta.z > 0.5 && params.Flags.y > 0.5 && params.Flags.x < 0.5)
	{
		// A camera move changes the current depth and the previous depth at the
		// same world point. Compare the reprojected previous-frame depth instead
		// of comparing the two raw samples directly; otherwise every translation
		// would look like an object-disocclusion event.
		if (camera_motion_valid)
		{
			float current_depth = texture(DepthTexture, uv).r;
			float current_ndc_depth = params.Flags.w > 0.5 ? current_depth * 2.0 - 1.0 : current_depth;
			vec4 current_clip = vec4(uv * 2.0 - 1.0, current_ndc_depth, 1.0);
			vec4 previous_clip = vec4(
				dot(params.CameraClipToPrevious[0], current_clip),
				dot(params.CameraClipToPrevious[1], current_clip),
				dot(params.CameraClipToPrevious[2], current_clip),
				dot(params.CameraClipToPrevious[3], current_clip));
			float predicted_depth = 0.0;
			if (abs(previous_clip.w) > 1e-5)
			{
				float previous_ndc_depth = previous_clip.z / previous_clip.w;
				predicted_depth = params.Flags.w > 0.5 ? previous_ndc_depth * 0.5 + 0.5 : previous_ndc_depth;
			}

			vec2 previous_uv = clamp(predicted_previous_uv, vec2(0.0001), vec2(0.9999));
			vec2 texel = 1.0 / max(params.InputOutputSize.xy, vec2(1.0));
			float best_relative_depth_error = 1e20;
			for (int tap = 0; tap < 5; ++tap)
			{
				vec2 offset = tap == 0 ? vec2(0.0) :
					tap == 1 ? vec2(texel.x, 0.0) :
					tap == 2 ? vec2(-texel.x, 0.0) :
					tap == 3 ? vec2(0.0, texel.y) :
					vec2(0.0, -texel.y);
				float previous_depth = texture(PreviousMotionMetadata,
					clamp(previous_uv + offset, vec2(0.0001), vec2(0.9999))).a;
				float relative_error = abs(predicted_depth - previous_depth) /
					max(max(abs(predicted_depth), abs(previous_depth)), 0.01);
				best_relative_depth_error = min(best_relative_depth_error, relative_error);
			}

			motion_bias = predicted_depth <= 0.0 || predicted_depth >= 1.0 ||
				best_relative_depth_error > 0.15 ? 1.0 : 0.0;
		}
		else
		{
			// With no camera pair there is no defensible depth prediction. Keep
			// the optional hint conservative and let the color-flow confidence
			// determine history usage.
			motion_bias = 0.0;
		}
	}

	float depth = params.Flags.y > 0.5 ? texture(DepthTexture, uv).r : 0.0;
	imageStore(MotionTexture, pixel, vec4(best_offset, 0.0, 0.0));
	imageStore(MotionMetadataTexture, pixel, vec4(best_offset, confidence, depth));
	imageStore(BiasTexture, pixel, vec4(motion_bias, 0.0, 0.0, 1.0));
}
)"
