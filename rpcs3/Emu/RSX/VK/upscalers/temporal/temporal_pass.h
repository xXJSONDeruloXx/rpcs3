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
		struct depth_resample_pass final : compute_task
		{
			std::unique_ptr<vk::sampler> m_sampler;
			const vk::image_view* m_input_image = nullptr;
			const vk::image_view* m_output_image = nullptr;
			std::array<float, 4> m_constants{};

			depth_resample_pass();

			std::vector<glsl::program_input> get_inputs() override;
			void bind_resources(const vk::command_buffer&) override;
			void run(const vk::command_buffer& cmd,
				vk::viewable_image* input,
				vk::viewable_image* output,
				const size2u& input_size,
				const size2u& output_size);
		};

		struct motion_pass final : compute_task
		{
			std::unique_ptr<vk::sampler> m_sampler;
			const vk::image_view* m_current_image = nullptr;
			const vk::image_view* m_previous_image = nullptr;
			const vk::image_view* m_depth_image = nullptr;
			const vk::image_view* m_motion_image = nullptr;
			const vk::image_view* m_motion_meta_image = nullptr;
			const vk::image_view* m_previous_motion_image = nullptr;
			const vk::image_view* m_motion_bias_image = nullptr;
			const vk::image_view* m_object_motion_image = nullptr;
			const vk::buffer* m_scene_change_buffer = nullptr;
			std::array<float, 32> m_constants{};

			motion_pass();

			std::vector<glsl::program_input> get_inputs() override;
			void bind_resources(const vk::command_buffer&) override;
			void run(const vk::command_buffer& cmd,
				vk::viewable_image* current,
				vk::viewable_image* previous,
				vk::viewable_image* depth,
				vk::viewable_image* motion,
				vk::viewable_image* motion_meta,
				vk::viewable_image* previous_motion,
				vk::viewable_image* motion_bias,
				vk::viewable_image* object_motion,
				const size2u& input_size,
				const size2u& output_size,
				const std::array<float, 16>* clip_to_previous,
				float jitter_delta_x,
				float jitter_delta_y,
				bool dynamic_mask,
				bool far_rotation,
				u32 edge_mode,
				float max_motion,
				const vk::buffer* scene_change_buffer,
				bool generate_motion_bias,
				bool object_motion_valid,
				bool reset);
		};

		struct motion_filter_pass final : compute_task
		{
			std::unique_ptr<vk::sampler> m_sampler;
			const vk::image_view* m_input_image = nullptr;
			const vk::image_view* m_output_image = nullptr;
			std::array<float, 4> m_constants{};

			motion_filter_pass();

			std::vector<glsl::program_input> get_inputs() override;
			void bind_resources(const vk::command_buffer&) override;
			void run(const vk::command_buffer& cmd, vk::viewable_image* input,
				vk::viewable_image* output, const size2u& input_size);
		};

		struct resolve_pass final : compute_task
		{
			std::unique_ptr<vk::sampler> m_sampler;
			const vk::image_view* m_current_image = nullptr;
			const vk::image_view* m_previous_image = nullptr;
			const vk::image_view* m_motion_image = nullptr;
			const vk::image_view* m_previous_motion_image = nullptr;
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
				vk::viewable_image* previous_motion,
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
		std::unique_ptr<vk::viewable_image> m_depth_resampled;
		std::unique_ptr<vk::viewable_image> m_dummy_depth;
		std::unique_ptr<vk::viewable_image> m_motion;
		std::unique_ptr<vk::viewable_image> m_motion_meta;
		std::unique_ptr<vk::viewable_image> m_motion_filtered;
		std::unique_ptr<vk::viewable_image> m_motion_bias;
		std::unique_ptr<vk::viewable_image> m_previous_motion;
		std::unique_ptr<vk::viewable_image> m_native_output;
		VkFormat m_output_format = VK_FORMAT_UNDEFINED;
		VkFormat m_input_format = VK_FORMAT_UNDEFINED;
		bool m_has_history = false;
		std::array<float, 16> m_previous_camera_view_projection{};
		bool m_has_previous_camera_view_projection = false;
		float m_previous_jitter_x = 0.f;
		float m_previous_jitter_y = 0.f;
		bool m_has_previous_jitter = false;
		bool m_streamline_attempted = false;
		bool m_configuration_logged = false;
		bool m_last_native_evaluated = false;
		VkImage m_midframe_source = VK_NULL_HANDLE;
		VkImage m_midframe_destination = VK_NULL_HANDLE;
		u64 m_midframe_frame = 0;
		bool m_midframe_native_history_valid = false;

		static constexpr u32 scene_change_counter_count = 9;
		struct scene_change_buffer_slot
		{
			std::unique_ptr<vk::buffer> buffer;
			u64 frame_tag = 0;
			u32 width = 0;
			u32 height = 0;
			bool consumed = true;
		};
		std::vector<scene_change_buffer_slot> m_scene_change_buffers;
		vk::buffer* m_active_scene_change_buffer = nullptr;
		bool m_scene_was_changing = false;
		bool m_scene_motion_meter_warm = false;
		u32 m_scene_frames_since_reset = 0;

		void dispose_images();
		bool read_scene_change_counters(const size2u& input_size, u32& changed, u32& motion);
		vk::buffer* prepare_scene_change_buffer(const size2u& input_size);
		bool initialize_images(const vk::viewable_image* src, const size2u& input_size, const size2u& output_size);
		void copy_current_to_history(const vk::command_buffer& cmd, vk::viewable_image* src, const size2u& input_size);
		void copy_motion_to_history(const vk::command_buffer& cmd);
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

		// Beast's optional mid-frame hook has a separate history/viewport from
		// the final present path. The caller supplies the real destination RT;
		// this method returns true only when a native Streamline evaluation wrote
		// it and the guest fullscreen draw can be skipped.
		bool run_mid_frame(
			const vk::command_buffer& cmd,
			vk::viewable_image* src,
			vk::viewable_image* dst,
			vk::viewable_image* depth,
			const temporal_frame_inputs& inputs);

		vk::viewable_image* motion_image() const { return m_motion_filtered.get(); }
		vk::viewable_image* output_image() const { return m_output.get(); }
		bool has_history() const { return m_has_history; }
	};
}
