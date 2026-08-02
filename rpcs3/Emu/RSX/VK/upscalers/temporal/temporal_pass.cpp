#include "stdafx.h"

#include "temporal_pass.h"

#include "../../VKHelpers.h"
#include "../../VKRenderTargets.h"
#include "../../VKResourceManager.h"

#include "util/asm.hpp"

#include <algorithm>
#include <cmath>

namespace vk
{
	namespace
	{
		constexpr u32 temporal_workgroup_size = 16;

		vk::sampler* make_temporal_sampler(std::unique_ptr<vk::sampler>& sampler)
		{
			if (!sampler)
			{
				const auto pdev = vk::get_current_renderer();
				sampler = std::make_unique<vk::sampler>(*pdev,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_FALSE, 0.f, 1.f, 0.f, 0.f,
					VK_FILTER_LINEAR, VK_FILTER_LINEAR,
					VK_SAMPLER_MIPMAP_MODE_NEAREST,
					VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
			}

			return sampler.get();
		}

		void push_temporal_constants(const vk::command_buffer& cmd, vk::glsl::program* program,
			std::array<float, 8>& constants, const size2u& input_size, const size2u& output_size, bool reset, bool has_depth)
		{
			constants[0] = static_cast<float>(input_size.width);
			constants[1] = static_cast<float>(input_size.height);
			constants[2] = static_cast<float>(output_size.width);
			constants[3] = static_cast<float>(output_size.height);
			constants[4] = reset ? 1.f : 0.f;
			constants[5] = has_depth ? 1.f : 0.f;
			constants[6] = 0.f;
			constants[7] = 0.f;

			vkCmdPushConstants(cmd, program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
				static_cast<u32>(constants.size() * sizeof(float)), constants.data());
		}

		void push_motion_constants(const vk::command_buffer& cmd, vk::glsl::program* program,
			std::array<float, 24>& constants, const size2u& input_size, const size2u& output_size,
			bool reset, bool has_depth, const std::array<float, 16>* clip_to_previous)
		{
			constants[0] = static_cast<float>(input_size.width);
			constants[1] = static_cast<float>(input_size.height);
			constants[2] = static_cast<float>(output_size.width);
			constants[3] = static_cast<float>(output_size.height);
			constants[4] = reset ? 1.f : 0.f;
			constants[5] = has_depth ? 1.f : 0.f;
			constants[6] = clip_to_previous ? 1.f : 0.f;
			constants[7] = 0.f; // RPCS3's Vulkan depth attachment is in [0, 1].

			if (clip_to_previous)
			{
				std::copy(clip_to_previous->begin(), clip_to_previous->end(), constants.begin() + 8);
			}
			else
			{
				std::fill(constants.begin() + 8, constants.end(), 0.f);
			}

			vkCmdPushConstants(cmd, program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
				static_cast<u32>(constants.size() * sizeof(float)), constants.data());
		}

		bool invert_matrix(const std::array<float, 16>& input, std::array<float, 16>& output)
		{
			float augmented[4][8]{};
			for (u32 row = 0; row < 4; ++row)
			{
				for (u32 column = 0; column < 4; ++column)
				{
					augmented[row][column] = input[row * 4 + column];
				}
				augmented[row][row + 4] = 1.f;
			}

			for (u32 column = 0; column < 4; ++column)
			{
				u32 pivot = column;
				for (u32 row = column + 1; row < 4; ++row)
				{
					if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column]))
					{
						pivot = row;
					}
				}

				if (std::abs(augmented[pivot][column]) < 1e-6f)
				{
					return false;
				}

				if (pivot != column)
				{
					for (u32 value = 0; value < 8; ++value)
					{
						std::swap(augmented[pivot][value], augmented[column][value]);
					}
				}

				const float scale = augmented[column][column];
				for (u32 value = 0; value < 8; ++value)
				{
					augmented[column][value] /= scale;
				}

				for (u32 row = 0; row < 4; ++row)
				{
					if (row == column)
					{
						continue;
					}

					const float factor = augmented[row][column];
					for (u32 value = 0; value < 8; ++value)
					{
						augmented[row][value] -= factor * augmented[column][value];
					}
				}
			}

			for (u32 row = 0; row < 4; ++row)
			{
				for (u32 column = 0; column < 4; ++column)
				{
					output[row * 4 + column] = augmented[row][column + 4];
				}
			}
			return true;
		}

		bool make_clip_to_previous(const std::array<float, 16>& current,
			const std::array<float, 16>& previous, std::array<float, 16>& result)
		{
			std::array<float, 16> inverse_current{};
			if (!invert_matrix(current, inverse_current))
			{
				return false;
			}

			for (u32 row = 0; row < 4; ++row)
			{
				for (u32 column = 0; column < 4; ++column)
				{
					float value = 0.f;
					for (u32 k = 0; k < 4; ++k)
					{
						value += previous[row * 4 + k] * inverse_current[k * 4 + column];
					}
					result[row * 4 + column] = value;
				}
			}
			return true;
		}

		void insert_temporal_write_read_barrier(const vk::command_buffer& cmd, const vk::viewable_image& image,
			VkAccessFlags dst_access = VK_ACCESS_SHADER_READ_BIT)
		{
			vk::insert_image_memory_barrier(cmd,
				image.value,
				image.current_layout,
				image.current_layout,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_SHADER_WRITE_BIT,
				dst_access,
				{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
		}
	}

	namespace temporal
	{
		motion_pass::motion_pass()
		{
			m_src =
			#include "Emu/RSX/Program/Upscalers/Temporal/motion.glsl"
			;

			ssbo_count = 0;
			use_push_constants = true;
			push_constants_size = static_cast<u32>(m_constants.size() * sizeof(float));
			create();
		}

		std::vector<glsl::program_input> motion_pass::get_inputs()
		{
			std::vector<vk::glsl::program_input> inputs =
			{
				glsl::program_input::make(::glsl::glsl_compute_program, "CurrentTexture", vk::glsl::input_type_texture, 0, 0),
				glsl::program_input::make(::glsl::glsl_compute_program, "PreviousTexture", vk::glsl::input_type_texture, 0, 1),
				glsl::program_input::make(::glsl::glsl_compute_program, "DepthTexture", vk::glsl::input_type_texture, 0, 2),
				glsl::program_input::make(::glsl::glsl_compute_program, "MotionTexture", vk::glsl::input_type_storage_texture, 0, 3),
			};

			auto result = compute_task::get_inputs();
			result.insert(result.end(), inputs.begin(), inputs.end());
			return result;
		}

		void motion_pass::bind_resources(const vk::command_buffer& /*cmd*/)
		{
			make_temporal_sampler(m_sampler);
			m_program->bind_uniform({ *m_current_image, *m_sampler }, 0, 0);
			m_program->bind_uniform({ *m_previous_image, *m_sampler }, 0, 1);
			m_program->bind_uniform({ *m_depth_image, *m_sampler }, 0, 2);
			m_program->bind_uniform({ *m_motion_image }, 0, 3);
		}

		void motion_pass::run(const vk::command_buffer& cmd,
			vk::viewable_image* current,
			vk::viewable_image* previous,
			vk::viewable_image* depth,
			vk::viewable_image* motion,
			const size2u& input_size,
			const size2u& output_size,
			const std::array<float, 16>* clip_to_previous,
			bool reset)
		{
			const auto remap = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
			m_current_image = current->get_view(remap);
			m_previous_image = previous->get_view(remap);
			m_depth_image = depth
				? depth->get_view(remap, VK_IMAGE_ASPECT_DEPTH_BIT)
				: m_current_image;
			m_motion_image = motion->get_view(remap);

			if (!m_program)
			{
				load_program(cmd);
			}

			push_motion_constants(cmd, m_program.get(), m_constants, input_size, output_size, reset, depth != nullptr, clip_to_previous);
			compute_task::run(cmd, utils::aligned_div(input_size.width, temporal_workgroup_size),
				utils::aligned_div(input_size.height, temporal_workgroup_size), 1);
		}

		resolve_pass::resolve_pass()
		{
			m_src =
			#include "Emu/RSX/Program/Upscalers/Temporal/resolve.glsl"
			;

			ssbo_count = 0;
			use_push_constants = true;
			push_constants_size = static_cast<u32>(m_constants.size() * sizeof(float));
			create();
		}

		std::vector<glsl::program_input> resolve_pass::get_inputs()
		{
			std::vector<vk::glsl::program_input> inputs =
			{
				glsl::program_input::make(::glsl::glsl_compute_program, "CurrentTexture", vk::glsl::input_type_texture, 0, 0),
				glsl::program_input::make(::glsl::glsl_compute_program, "PreviousTexture", vk::glsl::input_type_texture, 0, 1),
				glsl::program_input::make(::glsl::glsl_compute_program, "MotionTexture", vk::glsl::input_type_texture, 0, 2),
				glsl::program_input::make(::glsl::glsl_compute_program, "DepthTexture", vk::glsl::input_type_texture, 0, 3),
				glsl::program_input::make(::glsl::glsl_compute_program, "OutputTexture", vk::glsl::input_type_storage_texture, 0, 4),
			};

			auto result = compute_task::get_inputs();
			result.insert(result.end(), inputs.begin(), inputs.end());
			return result;
		}

		void resolve_pass::bind_resources(const vk::command_buffer& /*cmd*/)
		{
			make_temporal_sampler(m_sampler);
			m_program->bind_uniform({ *m_current_image, *m_sampler }, 0, 0);
			m_program->bind_uniform({ *m_previous_image, *m_sampler }, 0, 1);
			m_program->bind_uniform({ *m_motion_image, *m_sampler }, 0, 2);
			m_program->bind_uniform({ *m_depth_image, *m_sampler }, 0, 3);
			m_program->bind_uniform({ *m_output_image }, 0, 4);
		}

		void resolve_pass::run(const vk::command_buffer& cmd,
			vk::viewable_image* current,
			vk::viewable_image* previous,
			vk::viewable_image* motion,
			vk::viewable_image* depth,
			vk::viewable_image* output,
			const size2u& input_size,
			const size2u& output_size,
			bool reset)
		{
			const auto remap = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
			m_current_image = current->get_view(remap);
			m_previous_image = previous->get_view(remap);
			m_motion_image = motion->get_view(remap);
			m_depth_image = depth
				? depth->get_view(remap, VK_IMAGE_ASPECT_DEPTH_BIT)
				: m_current_image;
			m_output_image = output->get_view(remap);

			if (!m_program)
			{
				load_program(cmd);
			}

			push_temporal_constants(cmd, m_program.get(), m_constants, input_size, output_size, reset, depth != nullptr);
			compute_task::run(cmd, utils::aligned_div(output_size.width, temporal_workgroup_size),
				utils::aligned_div(output_size.height, temporal_workgroup_size), 1);
		}
	}

	void dlss_upscale_pass::dispose_images()
	{
		auto dispose = [](auto& image)
		{
			if (image && image->value)
			{
				vk::get_resource_manager()->dispose(image);
			}
			else
			{
				image.reset();
			}
		};

		dispose(m_output);
		dispose(m_previous_color);
		dispose(m_motion);
		m_output_format = VK_FORMAT_UNDEFINED;
		m_input_format = VK_FORMAT_UNDEFINED;
		m_has_history = false;
		m_previous_camera_view_projection = {};
		m_has_previous_camera_view_projection = false;
	}

	bool dlss_upscale_pass::supports_format(const vk::render_device& device, VkFormat format, VkFormatFeatureFlags features)
	{
		return format != VK_FORMAT_UNDEFINED &&
			(device.get_format_properties(format).optimalTilingFeatures & features) == features;
	}

	bool dlss_upscale_pass::initialize_images(const vk::viewable_image* src, const size2u& input_size, const size2u& output_size)
	{
		dispose_images();

		if (!src || !input_size.width || !input_size.height || !output_size.width || !output_size.height)
		{
			return false;
		}

		const auto pdev = vk::get_current_renderer();
		const VkFormatFeatureFlags sampled = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
		const VkFormatFeatureFlags output_features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | sampled | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
		const VkFormatFeatureFlags motion_features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | sampled;

		if (!supports_format(*pdev, src->format(), sampled))
		{
			rsx_log.error("DLSS temporal path: source format 0x%x is not sampleable", static_cast<u32>(src->format()));
			return false;
		}

		// The resolve shader declares an rgba8 storage image. Keep the internal
		// target in the matching channel order; the final blit performs any
		// BGRA swap required by the window surface.
		for (const VkFormat format : { VK_FORMAT_R8G8B8A8_UNORM })
		{
			if (supports_format(*pdev, format, output_features))
			{
				m_output_format = format;
				break;
			}
		}

		if (!supports_format(*pdev, VK_FORMAT_R16G16B16A16_SFLOAT, motion_features))
		{
			rsx_log.warning("DLSS temporal path: no storage-capable RGBA16F format for motion vectors");
			return false;
		}

		if (m_output_format == VK_FORMAT_UNDEFINED)
		{
			rsx_log.warning("DLSS temporal path: no storage-capable RGBA8 output format");
			return false;
		}

		auto make_image = [pdev](VkFormat format, u32 width, u32 height, VkImageUsageFlags usage, vmm_allocation_pool pool)
		{
			return std::make_unique<vk::viewable_image>(
				*pdev,
				pdev->get_memory_mapping().device_local,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D,
				format,
				width, height, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_TILING_OPTIMAL,
				usage,
				VK_IMAGE_CREATE_ALLOW_NULL_RPCS3,
				pool,
				RSX_FORMAT_CLASS_COLOR);
		};

		m_input_format = src->format();
		m_previous_color = make_image(m_input_format, src->width(), src->height(),
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);
		m_motion = make_image(VK_FORMAT_R16G16B16A16_SFLOAT, input_size.width, input_size.height,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);
		m_output = make_image(m_output_format, output_size.width, output_size.height,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);

		if (!m_previous_color || !m_motion || !m_output ||
			!m_previous_color->value || !m_motion->value || !m_output->value)
		{
			dispose_images();
			return false;
		}

		m_previous_color->set_debug_name("DLSS temporal previous color");
		m_motion->set_debug_name("DLSS temporal motion vectors");
		m_output->set_debug_name("DLSS temporal output");
		return true;
	}

	void dlss_upscale_pass::copy_current_to_history(const vk::command_buffer& cmd, vk::viewable_image* src, const size2u& /*input_size*/)
	{
		src->push_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		m_previous_color->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		VkImageCopy region{};
		region.srcSubresource = { src->aspect(), 0, 0, 1 };
		region.dstSubresource = { m_previous_color->aspect(), 0, 0, 1 };
		region.extent = { src->width(), src->height(), 1 };
		vkCmdCopyImage(cmd, src->value, src->current_layout, m_previous_color->value, m_previous_color->current_layout, 1, &region);

		m_previous_color->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		src->pop_layout(cmd);
	}

	vk::viewable_image* dlss_upscale_pass::scale_output(
		const vk::command_buffer& cmd,
		vk::viewable_image* src,
		VkImage present_surface,
		VkImageLayout present_surface_layout,
		const VkImageBlit& request,
		rsx::flags32_t mode)
	{
		return scale_output_temporal(cmd, src, present_surface, present_surface_layout, request, mode, {});
	}

	vk::viewable_image* dlss_upscale_pass::scale_output_temporal(
		const vk::command_buffer& cmd,
		vk::viewable_image* src,
		VkImage present_surface,
		VkImageLayout present_surface_layout,
		const VkImageBlit& request,
		rsx::flags32_t mode,
		const temporal_frame_inputs& inputs)
	{
		if (!src)
		{
			return nullptr;
		}

		const size2u input_size
		{
			static_cast<u32>(std::abs(request.srcOffsets[1].x - request.srcOffsets[0].x)),
			static_cast<u32>(std::abs(request.srcOffsets[1].y - request.srcOffsets[0].y))
		};
		const size2u output_size
		{
			static_cast<u32>(std::abs(request.dstOffsets[1].x - request.dstOffsets[0].x)),
			static_cast<u32>(std::abs(request.dstOffsets[1].y - request.dstOffsets[0].y))
		};

		if (!input_size.width || !input_size.height || !output_size.width || !output_size.height || src->samples() != 1)
		{
			if (mode & UPSCALE_AND_COMMIT)
			{
				ensure(present_surface);
				src->push_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
				vkCmdBlitImage(cmd, src->value, src->current_layout, present_surface, present_surface_layout, 1, &request, VK_FILTER_LINEAR);
				src->pop_layout(cmd);
				return nullptr;
			}

			return src;
		}

		if (!m_output || m_input_format != src->format() || m_previous_color->width() != src->width() ||
			m_previous_color->height() != src->height() || m_motion->width() != input_size.width ||
			m_motion->height() != input_size.height || m_output->width() != output_size.width ||
			m_output->height() != output_size.height)
		{
			if (!initialize_images(src, input_size, output_size))
			{
				if (mode & UPSCALE_AND_COMMIT)
				{
					ensure(present_surface);
					src->push_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
					vkCmdBlitImage(cmd, src->value, src->current_layout, present_surface, present_surface_layout, 1, &request, VK_FILTER_LINEAR);
					src->pop_layout(cmd);
					return nullptr;
				}

				return src;
			}
		}

		const bool reset = inputs.reset_history || !m_has_history;
		if (reset)
		{
			copy_current_to_history(cmd, src, input_size);
		}

		const bool has_depth = inputs.depth && inputs.depth->value && inputs.depth->samples() == 1;
		if (has_depth)
		{
			if (auto* depth_target = dynamic_cast<vk::render_target*>(inputs.depth))
			{
				depth_target->read_barrier(const_cast<vk::command_buffer&>(cmd));
			}
			inputs.depth->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}

		src->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_previous_color->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		if (m_motion->current_layout == VK_IMAGE_LAYOUT_GENERAL)
		{
			insert_temporal_write_read_barrier(cmd, *m_motion, VK_ACCESS_SHADER_WRITE_BIT);
		}
		m_motion->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		m_output->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);

		std::array<float, 16> clip_to_previous{};
		const bool has_camera_pair = has_depth && !reset && inputs.has_camera_view_projection &&
			m_has_previous_camera_view_projection && make_clip_to_previous(
				inputs.camera_view_projection, m_previous_camera_view_projection, clip_to_previous);

		vk::get_compute_task<vk::temporal::motion_pass>()->run(cmd, src, m_previous_color.get(),
			has_depth ? inputs.depth : nullptr, m_motion.get(), input_size, output_size,
			has_camera_pair ? &clip_to_previous : nullptr, reset);
		insert_temporal_write_read_barrier(cmd, *m_motion);

		if (inputs.has_camera_view_projection)
		{
			m_previous_camera_view_projection = inputs.camera_view_projection;
			m_has_previous_camera_view_projection = true;
		}

		if (!m_streamline_attempted)
		{
			m_streamline_attempted = true;
			get_streamline_dlss().initialize(*vk::get_current_renderer(), g_cfg.video.dlss_frame_generation.get());
		}

		bool native_dlss_evaluated = false;
		auto& streamline = get_streamline_dlss();
		if (!g_cfg.video.dlss_frame_generation.get() && streamline.frame_generation_proxy_armed())
		{
			streamline.prepare_swapchain_recreation();
		}
		if (has_depth && streamline.available())
		{
			const auto remap = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
			const auto* input_view = src->get_view(remap);
			const auto* output_view = m_output->get_view(remap);
			const auto* depth_view = inputs.depth->get_view(remap, VK_IMAGE_ASPECT_DEPTH_BIT);
			const auto* motion_view = m_motion->get_view(remap);

			streamline_dlss::texture input_texture
			{
				src->value, input_view->value, src->format(), src->current_layout, src->width(), src->height()
			};
			streamline_dlss::texture output_texture
			{
				m_output->value, output_view->value, m_output->format(), m_output->current_layout, m_output->width(), m_output->height()
			};
			streamline_dlss::texture depth_texture
			{
				inputs.depth->value, depth_view->value, inputs.depth->format(), inputs.depth->current_layout,
				inputs.depth->width(), inputs.depth->height()
			};
			streamline_dlss::texture motion_texture
			{
				m_motion->value, motion_view->value, m_motion->format(), m_motion->current_layout, m_motion->width(), m_motion->height()
			};

			streamline_dlss::mode dlss_mode = streamline_dlss::mode::balanced;
			switch (g_cfg.video.dlss_quality.get())
			{
			case dlss_quality_mode::quality: dlss_mode = streamline_dlss::mode::max_quality; break;
			case dlss_quality_mode::balanced: dlss_mode = streamline_dlss::mode::balanced; break;
			case dlss_quality_mode::performance: dlss_mode = streamline_dlss::mode::max_performance; break;
			case dlss_quality_mode::ultra_performance: dlss_mode = streamline_dlss::mode::ultra_performance; break;
			}

			if (streamline.set_options(0, dlss_mode, output_size.width, output_size.height, false))
			{
				native_dlss_evaluated = streamline.evaluate(cmd, 0, static_cast<u32>(vk::get_current_frame_id()), reset,
					inputs.jitter_x, inputs.jitter_y, input_texture, output_texture, depth_texture, motion_texture);

				if (native_dlss_evaluated && g_cfg.video.dlss_frame_generation.get() &&
					streamline.frame_generation_available() && streamline.frame_generation_proxy_armed())
				{
					const u32 color_width = inputs.present_width ? inputs.present_width : output_size.width;
					const u32 color_height = inputs.present_height ? inputs.present_height : output_size.height;
					const VkFormat color_format = inputs.present_format != VK_FORMAT_UNDEFINED ? inputs.present_format : m_output->format();
					const u32 backbuffer_count = inputs.present_buffer_count ? inputs.present_buffer_count : 2;

					if (streamline.configure_frame_generation(0, color_width, color_height, color_format, backbuffer_count,
						g_cfg.video.dlss_frame_generation_frames.get(),
						depth_texture, motion_texture))
					{
						streamline.tag_frame(cmd, depth_texture, motion_texture);
					}
				}
			}
		}

		if (!native_dlss_evaluated)
		{
			vk::get_compute_task<vk::temporal::resolve_pass>()->run(cmd, src, m_previous_color.get(), m_motion.get(),
				has_depth ? inputs.depth : nullptr, m_output.get(), input_size, output_size, reset);
		}
		// Both the local resolve and Streamline's native evaluator write the same
		// output image. Make that write visible before the present blit or the
		// caller's shader-read transition; this also keeps the native path from
		// relying on an implementation-specific Streamline barrier.
		insert_temporal_write_read_barrier(cmd, *m_output);

		if (has_depth)
		{
			inputs.depth->pop_layout(cmd);
		}
		src->pop_layout(cmd);
		m_previous_color->pop_layout(cmd);

		m_has_history = true;
		copy_current_to_history(cmd, src, input_size);

		if (mode & UPSCALE_AND_COMMIT)
		{
			ensure(present_surface);
			m_output->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

			VkImageBlit output_request = request;
			output_request.srcSubresource = { m_output->aspect(), 0, 0, 1 };
			output_request.srcOffsets[0] = { 0, 0, 0 };
			output_request.srcOffsets[1] = { static_cast<s32>(output_size.width), static_cast<s32>(output_size.height), 1 };
			if (request.srcOffsets[0].x > request.srcOffsets[1].x)
			{
				std::swap(output_request.srcOffsets[0].x, output_request.srcOffsets[1].x);
			}
			if (request.srcOffsets[0].y > request.srcOffsets[1].y)
			{
				std::swap(output_request.srcOffsets[0].y, output_request.srcOffsets[1].y);
			}

			vkCmdBlitImage(cmd, m_output->value, m_output->current_layout,
				present_surface, present_surface_layout, 1, &output_request, VK_FILTER_LINEAR);
			return nullptr;
		}

		m_output->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		return m_output.get();
	}
}
