#pragma once

#include "../vkutils/commands.h"
#include "../vkutils/image.h"

#include <array>

namespace vk
{
	namespace upscaling_flags_
	{
		enum upscaling_flags
		{
			UPSCALE_DEFAULT_VIEW = (1 << 0),
			UPSCALE_LEFT_VIEW    = (1 << 0),
			UPSCALE_RIGHT_VIEW   = (1 << 1),
			UPSCALE_AND_COMMIT   = (1 << 2)
		};
	}

	using namespace upscaling_flags_;

	// Additional per-frame inputs used by temporal upscalers. The normal
	// output-scaling interface intentionally remains color-only so the GL
	// backend and the existing nearest/bilinear/FSR1 paths are unaffected.
	struct temporal_frame_inputs
	{
		vk::viewable_image* depth = nullptr;
		u32 present_width = 0;
		u32 present_height = 0;
		u32 present_buffer_count = 0;
		VkFormat present_format = VK_FORMAT_UNDEFINED;
		bool reset_history = false;
		float jitter_x = 0.f;
		float jitter_y = 0.f;
		// Optional structural camera capture. When present, the temporal pass
		// derives static-world motion by reprojection; otherwise it retains the
		// color patch-search estimator.
		std::array<float, 16> camera_view_projection{};
		bool has_camera_view_projection = false;
	};

	struct upscaler
	{
		virtual ~upscaler() {}

		virtual vk::viewable_image* scale_output(
			const vk::command_buffer& cmd,          // CB
			vk::viewable_image* src,                // Source input
			VkImage present_surface,                // Present target. May be VK_NULL_HANDLE for some passes
			VkImageLayout present_surface_layout,   // Present surface layout, or VK_IMAGE_LAYOUT_UNDEFINED if no present target is provided
			const VkImageBlit& request,             // Scaling request information
			rsx::flags32_t mode                     // Mode
		) = 0;

		// Temporal implementations override this entry point. The default is
		// deliberately a color-only fallback so selecting a temporal mode cannot
		// make non-Vulkan or legacy upscalers dereference missing resources.
		virtual vk::viewable_image* scale_output_temporal(
			const vk::command_buffer& cmd,
			vk::viewable_image* src,
			VkImage present_surface,
			VkImageLayout present_surface_layout,
			const VkImageBlit& request,
			rsx::flags32_t mode,
			const temporal_frame_inputs& /*inputs*/
		)
		{
			return scale_output(cmd, src, present_surface, present_surface_layout, request, mode);
		}
	};
}
