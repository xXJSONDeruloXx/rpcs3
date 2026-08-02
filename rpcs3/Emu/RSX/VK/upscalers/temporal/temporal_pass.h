#pragma once

#include "../../vkutils/barriers.h"
#include "../../vkutils/sampler.h"
#include "../../VKCompute.h"

#include "streamline_dlss.h"

#include "../upscaling.h"

namespace vk
{
	namespace temporal
	{
		struct motion_pass final : compute_task
		{
			std::unique_ptr<vk::sampler> m_sampler;
			const vk::image_view* m_current_image = nullptr;
			const vk::image_view* m_previous_image = nullptr;
			const vk::image_view* m_depth_image = nullptr;
			const vk::image_view* m_motion_image = nullptr;
			std::array<float, 24> m_constants{};

			motion_pass();

			std::vector<glsl::program_input> get_inputs() override;
			void bind_resources(const vk::command_buffer&) override;
			void run(const vk::command_buffer& cmd,
				vk::viewable_image* current,
				vk::viewable_image* previous,
				vk::viewable_image* depth,
				vk::viewable_image* motion,
				const size2u& input_size,
				const size2u& output_size,
				const std::array<float, 16>* clip_to_previous,
				bool reset);
		};

		struct resolve_pass final : compute_task
		{
			std::unique_ptr<vk::sampler> m_sampler;
			const vk::image_view* m_current_image = nullptr;
			const vk::image_view* m_previous_image = nullptr;
			const vk::image_view* m_motion_image = nullptr;
			const vk::image_view* m_depth_image = nullptr;
			const vk::image_view* m_output_image = nullptr;
			std::array<float, 8> m_constants{};

			resolve_pass();

			std::vector<glsl::program_input> get_inputs() override;
			void bind_resources(const vk::command_buffer&) override;
			void run(const vk::command_buffer& cmd,
				vk::viewable_image* current,
				vk::viewable_image* previous,
				vk::viewable_image* motion,
				vk::viewable_image* depth,
				vk::viewable_image* output,
				const size2u& input_size,
				const size2u& output_size,
				bool reset);
		};
	}

	// A vendor-neutral temporal front end. With Streamline present this is the
	// place where DLSS-SR evaluates; without it the same inputs go through the
	// local resolve pass. Keeping both paths behind one interface is important:
	// the emulator can be tested on AMD/Intel and the vendor path never receives
	// guessed resources that the fallback did not also validate.
	class dlss_upscale_pass final : public upscaler
	{
		std::unique_ptr<vk::viewable_image> m_output;
		std::unique_ptr<vk::viewable_image> m_previous_color;
		std::unique_ptr<vk::viewable_image> m_motion;
		VkFormat m_output_format = VK_FORMAT_UNDEFINED;
		VkFormat m_input_format = VK_FORMAT_UNDEFINED;
		bool m_has_history = false;
		std::array<float, 16> m_previous_camera_view_projection{};
		bool m_has_previous_camera_view_projection = false;
		bool m_streamline_attempted = false;

		void dispose_images();
		bool initialize_images(const vk::viewable_image* src, const size2u& input_size, const size2u& output_size);
		void copy_current_to_history(const vk::command_buffer& cmd, vk::viewable_image* src, const size2u& input_size);
		static bool supports_format(const vk::render_device& device, VkFormat format, VkFormatFeatureFlags features);

	public:
		vk::viewable_image* scale_output(
			const vk::command_buffer& cmd,
			vk::viewable_image* src,
			VkImage present_surface,
			VkImageLayout present_surface_layout,
			const VkImageBlit& request,
			rsx::flags32_t mode
		) override;

		vk::viewable_image* scale_output_temporal(
			const vk::command_buffer& cmd,
			vk::viewable_image* src,
			VkImage present_surface,
			VkImageLayout present_surface_layout,
			const VkImageBlit& request,
			rsx::flags32_t mode,
			const temporal_frame_inputs& inputs
		) override;

		vk::viewable_image* motion_image() const { return m_motion.get(); }
		vk::viewable_image* output_image() const { return m_output.get(); }
		bool has_history() const { return m_has_history; }
	};
}
