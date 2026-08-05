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
		constexpr u32 scene_change_counter_count = 9;

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
			std::array<float, 32>& constants, const size2u& input_size, const size2u& output_size,
			bool reset, bool has_depth, const std::array<float, 16>* clip_to_previous,
			float jitter_delta_x, float jitter_delta_y, bool generate_motion_bias,
			bool dynamic_mask, bool far_rotation, u32 edge_mode, float max_motion,
			bool object_motion_valid)
		{
			constants[0] = static_cast<float>(input_size.width);
			constants[1] = static_cast<float>(input_size.height);
			constants[2] = static_cast<float>(output_size.width);
			constants[3] = static_cast<float>(output_size.height);
			constants[4] = reset ? 1.f : 0.f;
			constants[5] = has_depth ? 1.f : 0.f;
			constants[6] = clip_to_previous ? 1.f : 0.f;
			constants[7] = 0.f; // RPCS3's Vulkan depth attachment is in [0, 1].
			constants[8] = jitter_delta_x;
			constants[9] = jitter_delta_y;
			constants[10] = generate_motion_bias ? 1.f : 0.f;
			constants[11] = object_motion_valid ? 1.f : 0.f;
			constants[28] = dynamic_mask ? 1.f : 0.f;
			constants[29] = far_rotation ? 1.f : 0.f;
			constants[30] = static_cast<float>(std::min<u32>(edge_mode, 2));
			constants[31] = std::max(32.f, max_motion);

			if (clip_to_previous)
			{
				std::copy(clip_to_previous->begin(), clip_to_previous->end(), constants.begin() + 12);
			}
			else
			{
				std::fill(constants.begin() + 12, constants.begin() + 28, 0.f);
			}

			vkCmdPushConstants(cmd, program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
				static_cast<u32>(constants.size() * sizeof(float)), constants.data());
		}

		bool is_hdr_color_format(VkFormat format)
		{
			switch (format)
			{
			case VK_FORMAT_R16G16B16A16_SFLOAT:
			case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
			case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
				return true;
			default:
				return false;
			}
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

		bool camera_discontinuity(const std::array<float, 16>& current,
			const std::array<float, 16>& previous)
		{
			float rotation_delta = 0.f;
			for (const u32 index : { 0u, 1u, 2u, 4u, 5u, 6u, 8u, 9u, 10u })
			{
				rotation_delta = std::max(rotation_delta, std::abs(current[index] - previous[index]));
			}

			const float translation_x = current[3] - previous[3];
			const float translation_y = current[7] - previous[7];
			const float translation_z = current[11] - previous[11];
			const float translation_delta = std::sqrt(
				translation_x * translation_x + translation_y * translation_y + translation_z * translation_z);

			// Ordinary camera motion should accumulate through DLSS. A near-180-degree
			// turn or a large teleport is a different event: the previous color/depth
			// history no longer describes the current scene. This is intentionally a
			// high threshold; the GPU motion/confidence path handles normal pans.
			return rotation_delta > 1.5f || translation_delta > 16.f;
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
		depth_resample_pass::depth_resample_pass()
		{
			m_src =
			#include "Emu/RSX/Program/Upscalers/Temporal/depth_resample.glsl"
			;

			ssbo_count = 0;
			use_push_constants = true;
			push_constants_size = static_cast<u32>(m_constants.size() * sizeof(float));
			create();
		}

		std::vector<glsl::program_input> depth_resample_pass::get_inputs()
		{
			std::vector<vk::glsl::program_input> inputs =
			{
				glsl::program_input::make(::glsl::glsl_compute_program, "InputDepth", vk::glsl::input_type_texture, 0, 0),
				glsl::program_input::make(::glsl::glsl_compute_program, "OutputDepth", vk::glsl::input_type_storage_texture, 0, 1),
			};

			auto result = compute_task::get_inputs();
			result.insert(result.end(), inputs.begin(), inputs.end());
			return result;
		}

		void depth_resample_pass::bind_resources(const vk::command_buffer& /*cmd*/)
		{
			make_temporal_sampler(m_sampler);
			m_program->bind_uniform({ *m_input_image, *m_sampler }, 0, 0);
			m_program->bind_uniform({ *m_output_image }, 0, 1);
		}

		void depth_resample_pass::run(const vk::command_buffer& cmd,
			vk::viewable_image* input,
			vk::viewable_image* output,
			const size2u& input_size,
			const size2u& output_size)
		{
			const auto remap = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
			m_input_image = input->get_view(remap, VK_IMAGE_ASPECT_DEPTH_BIT);
			m_output_image = output->get_view(remap);

			if (!m_program)
			{
				load_program(cmd);
			}

			m_constants[0] = static_cast<float>(input_size.width);
			m_constants[1] = static_cast<float>(input_size.height);
			m_constants[2] = static_cast<float>(output_size.width);
			m_constants[3] = static_cast<float>(output_size.height);
			vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
				static_cast<u32>(m_constants.size() * sizeof(float)), m_constants.data());
			compute_task::run(cmd, utils::aligned_div(output_size.width, temporal_workgroup_size),
				utils::aligned_div(output_size.height, temporal_workgroup_size), 1);
		}

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
				glsl::program_input::make(::glsl::glsl_compute_program, "MotionMetadataTexture", vk::glsl::input_type_storage_texture, 0, 4),
			};

			auto result = compute_task::get_inputs();
			result.insert(result.end(), inputs.begin(), inputs.end());
			result.push_back(glsl::program_input::make(::glsl::glsl_compute_program,
				"SceneChange", vk::glsl::input_type_storage_buffer, 0, 5));
			result.push_back(glsl::program_input::make(::glsl::glsl_compute_program,
				"PreviousMotionMetadata", vk::glsl::input_type_texture, 0, 6));
			result.push_back(glsl::program_input::make(::glsl::glsl_compute_program,
				"BiasTexture", vk::glsl::input_type_storage_texture, 0, 7));
			result.push_back(glsl::program_input::make(::glsl::glsl_compute_program,
				"ObjectMotionTexture", vk::glsl::input_type_texture, 0, 8));
			return result;
		}

		void motion_pass::bind_resources(const vk::command_buffer& /*cmd*/)
		{
			make_temporal_sampler(m_sampler);
			m_program->bind_uniform({ *m_current_image, *m_sampler }, 0, 0);
			m_program->bind_uniform({ *m_previous_image, *m_sampler }, 0, 1);
			m_program->bind_uniform({ *m_depth_image, *m_sampler }, 0, 2);
			m_program->bind_uniform({ *m_motion_image }, 0, 3);
			m_program->bind_uniform({ *m_motion_meta_image }, 0, 4);
			m_program->bind_uniform({ *m_scene_change_buffer, 0, scene_change_counter_count * sizeof(u32) }, 0, 5);
			m_program->bind_uniform({ *m_previous_motion_image, *m_sampler }, 0, 6);
			m_program->bind_uniform({ *m_motion_bias_image }, 0, 7);
			m_program->bind_uniform({ *m_object_motion_image, *m_sampler }, 0, 8);
		}

		void motion_pass::run(const vk::command_buffer& cmd,
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
			bool reset)
		{
			const auto remap = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
			m_current_image = current->get_view(remap);
			m_previous_image = previous->get_view(remap);
			m_depth_image = depth
				? depth->get_view(remap, (depth->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT)
				: m_current_image;
			m_motion_image = motion->get_view(remap);
			m_motion_meta_image = motion_meta->get_view(remap);
			m_previous_motion_image = previous_motion->get_view(remap);
			m_motion_bias_image = motion_bias->get_view(remap);
			m_object_motion_image = (object_motion && object_motion->value)
				? object_motion->get_view(remap)
				: m_previous_motion_image;
			m_scene_change_buffer = scene_change_buffer;

			if (!m_program)
			{
				load_program(cmd);
			}

			insert_buffer_memory_barrier(cmd, m_scene_change_buffer->value, 0,
				scene_change_counter_count * sizeof(u32), VK_PIPELINE_STAGE_HOST_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
			push_motion_constants(cmd, m_program.get(), m_constants, input_size, output_size, reset, depth != nullptr,
				clip_to_previous, jitter_delta_x, jitter_delta_y, generate_motion_bias,
				dynamic_mask, far_rotation, edge_mode, max_motion,
				object_motion_valid && object_motion && object_motion->value);
			compute_task::run(cmd, utils::aligned_div(input_size.width, temporal_workgroup_size),
				utils::aligned_div(input_size.height, temporal_workgroup_size), 1);
			insert_buffer_memory_barrier(cmd, m_scene_change_buffer->value, 0,
				scene_change_counter_count * sizeof(u32), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
		}

		motion_filter_pass::motion_filter_pass()
		{
			m_src =
			#include "Emu/RSX/Program/Upscalers/Temporal/motion_filter.glsl"
			;

			ssbo_count = 0;
			use_push_constants = true;
			push_constants_size = static_cast<u32>(m_constants.size() * sizeof(float));
			create();
		}

		std::vector<glsl::program_input> motion_filter_pass::get_inputs()
		{
			std::vector<vk::glsl::program_input> inputs =
			{
				glsl::program_input::make(::glsl::glsl_compute_program, "InputMotion", vk::glsl::input_type_texture, 0, 0),
				glsl::program_input::make(::glsl::glsl_compute_program, "OutputMotion", vk::glsl::input_type_storage_texture, 0, 1),
			};

			auto result = compute_task::get_inputs();
			result.insert(result.end(), inputs.begin(), inputs.end());
			return result;
		}

		void motion_filter_pass::bind_resources(const vk::command_buffer& /*cmd*/)
		{
			make_temporal_sampler(m_sampler);
			m_program->bind_uniform({ *m_input_image, *m_sampler }, 0, 0);
			m_program->bind_uniform({ *m_output_image }, 0, 1);
		}

		void motion_filter_pass::run(const vk::command_buffer& cmd, vk::viewable_image* input,
			vk::viewable_image* output, const size2u& input_size)
		{
			const auto remap = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
			m_input_image = input->get_view(remap);
			m_output_image = output->get_view(remap);

			if (!m_program)
			{
				load_program(cmd);
			}

			m_constants[0] = static_cast<float>(input_size.width);
			m_constants[1] = static_cast<float>(input_size.height);
			m_constants[2] = 0.f;
			m_constants[3] = 0.f;
			vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
				static_cast<u32>(m_constants.size() * sizeof(float)), m_constants.data());
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
				glsl::program_input::make(::glsl::glsl_compute_program, "PreviousMotionTexture", vk::glsl::input_type_texture, 0, 4),
				glsl::program_input::make(::glsl::glsl_compute_program, "OutputTexture", vk::glsl::input_type_storage_texture, 0, 5),
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
			m_program->bind_uniform({ *m_previous_motion_image, *m_sampler }, 0, 4);
			m_program->bind_uniform({ *m_output_image }, 0, 5);
		}

		void resolve_pass::run(const vk::command_buffer& cmd,
			vk::viewable_image* current,
			vk::viewable_image* previous,
			vk::viewable_image* motion,
			vk::viewable_image* previous_motion,
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
			m_previous_motion_image = previous_motion->get_view(remap);
			m_depth_image = depth
				? depth->get_view(remap, (depth->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT)
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
		dispose(m_native_output);
		dispose(m_previous_color);
		dispose(m_depth_resampled);
		dispose(m_dummy_depth);
		dispose(m_motion);
		dispose(m_motion_meta);
		dispose(m_motion_filtered);
		dispose(m_motion_bias);
		dispose(m_previous_motion);
		m_output_format = VK_FORMAT_UNDEFINED;
		m_input_format = VK_FORMAT_UNDEFINED;
		m_has_history = false;
		m_previous_camera_view_projection = {};
		m_has_previous_camera_view_projection = false;
		m_previous_jitter_x = 0.f;
		m_previous_jitter_y = 0.f;
		m_has_previous_jitter = false;
		m_active_scene_change_buffer = nullptr;
		m_scene_was_changing = false;
		m_scene_motion_meter_warm = false;
		m_scene_frames_since_reset = 0;
		m_last_native_evaluated = false;
		m_midframe_source = VK_NULL_HANDLE;
		m_midframe_destination = VK_NULL_HANDLE;
		m_midframe_frame = 0;
		m_midframe_native_history_valid = false;
	}

	bool dlss_upscale_pass::read_scene_change_counters(const size2u& input_size, u32& changed, u32& motion)
	{
		const u64 completed_frame = vk::get_last_completed_frame_id();
		scene_change_buffer_slot* newest = nullptr;

		for (auto& slot : m_scene_change_buffers)
		{
			if (!slot.buffer || slot.consumed || !slot.frame_tag || slot.frame_tag > completed_frame)
			{
				continue;
			}

			if (slot.width == input_size.width && slot.height == input_size.height &&
				(!newest || slot.frame_tag > newest->frame_tag))
			{
				newest = &slot;
			}
		}

		// Only the newest completed result is useful. Mark older completed slots
		// consumed too, so a delayed present cannot replay stale cut evidence.
		for (auto& slot : m_scene_change_buffers)
		{
			if (slot.buffer && slot.frame_tag && slot.frame_tag <= completed_frame)
			{
				slot.consumed = true;
			}
		}

		if (!newest)
		{
			return false;
		}

		const auto values = static_cast<const u32*>(newest->buffer->map(0, scene_change_counter_count * sizeof(u32)));
		changed = values[0];
		motion = values[1];
		newest->buffer->unmap();
		return true;
	}

	vk::buffer* dlss_upscale_pass::prepare_scene_change_buffer(const size2u& input_size)
	{
		const u64 completed_frame = vk::get_last_completed_frame_id();
		scene_change_buffer_slot* slot = nullptr;

		for (auto& candidate : m_scene_change_buffers)
		{
			if (!candidate.buffer || !candidate.frame_tag || candidate.frame_tag <= completed_frame)
			{
				slot = &candidate;
				break;
			}
		}

		if (!slot)
		{
			m_scene_change_buffers.emplace_back();
			slot = &m_scene_change_buffers.back();
		}

		if (!slot->buffer)
		{
			const auto* device = vk::get_current_renderer();
			const auto& memory = device->get_memory_mapping();
			slot->buffer = std::make_unique<vk::buffer>(*device,
				scene_change_counter_count * sizeof(u32), memory.host_visible_coherent,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, VMM_ALLOCATION_POOL_SYSTEM);
		}

		auto* values = static_cast<u32*>(slot->buffer->map(0, scene_change_counter_count * sizeof(u32)));
		std::fill_n(values, scene_change_counter_count, 0u);
		slot->buffer->unmap();
		slot->frame_tag = vk::get_current_frame_id() + 1;
		slot->width = input_size.width;
		slot->height = input_size.height;
		slot->consumed = false;
		m_active_scene_change_buffer = slot->buffer.get();
		return m_active_scene_change_buffer;
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
		const VkFormatFeatureFlags motion_bias_features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | sampled;
		const VkFormatFeatureFlags motion_history_features = sampled |
			VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
		const VkFormatFeatureFlags depth_resample_features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | sampled;
		const VkFormatFeatureFlags dummy_depth_features = sampled | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;

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

		if (!supports_format(*pdev, VK_FORMAT_R16G16_SFLOAT, motion_features) ||
			!supports_format(*pdev, VK_FORMAT_R16G16B16A16_SFLOAT, motion_history_features) ||
			!supports_format(*pdev, VK_FORMAT_R8_UNORM, motion_bias_features))
		{
			rsx_log.warning("DLSS temporal path: no storage-capable motion formats");
			return false;
		}

		if (m_output_format == VK_FORMAT_UNDEFINED)
		{
			rsx_log.warning("DLSS temporal path: no storage-capable RGBA8 output format");
			return false;
		}

		auto make_image = [pdev](VkFormat format, u32 width, u32 height, VkImageUsageFlags usage,
			vmm_allocation_pool pool, rsx::format_class format_class = RSX_FORMAT_CLASS_COLOR)
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
				format_class);
		};

		m_input_format = src->format();
		m_previous_color = make_image(m_input_format, src->width(), src->height(),
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);
		if (supports_format(*pdev, VK_FORMAT_R32_SFLOAT, depth_resample_features))
		{
			m_depth_resampled = make_image(VK_FORMAT_R32_SFLOAT, input_size.width, input_size.height,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
				VMM_ALLOCATION_POOL_SWAPCHAIN);
		}
		if (supports_format(*pdev, VK_FORMAT_D32_SFLOAT, dummy_depth_features))
		{
			m_dummy_depth = make_image(VK_FORMAT_D32_SFLOAT, input_size.width, input_size.height,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
				VMM_ALLOCATION_POOL_SWAPCHAIN, RSX_FORMAT_CLASS_DEPTH16_FLOAT);
		}
		m_motion = make_image(VK_FORMAT_R16G16_SFLOAT, input_size.width, input_size.height,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);
		m_motion_meta = make_image(VK_FORMAT_R16G16B16A16_SFLOAT, input_size.width, input_size.height,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);
		m_motion_filtered = make_image(VK_FORMAT_R16G16_SFLOAT, input_size.width, input_size.height,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);
		m_motion_bias = make_image(VK_FORMAT_R8_UNORM, input_size.width, input_size.height,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);
		m_previous_motion = make_image(VK_FORMAT_R16G16B16A16_SFLOAT, input_size.width, input_size.height,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);
		m_output = make_image(m_output_format, output_size.width, output_size.height,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VMM_ALLOCATION_POOL_SWAPCHAIN);
		if (supports_format(*pdev, m_input_format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
			VK_FORMAT_FEATURE_TRANSFER_SRC_BIT))
		{
			m_native_output = make_image(m_input_format, output_size.width, output_size.height,
				VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
				VMM_ALLOCATION_POOL_SWAPCHAIN);
		}

		if (!m_previous_color || !m_motion || !m_motion_meta || !m_motion_filtered || !m_motion_bias || !m_previous_motion || !m_output ||
			!m_previous_color->value || !m_motion->value || !m_motion_meta->value ||
			!m_motion_filtered->value || !m_motion_bias->value || !m_previous_motion->value || !m_output->value)
		{
			dispose_images();
			return false;
		}

		m_previous_color->set_debug_name("DLSS temporal previous color");
		if (m_depth_resampled)
		{
			m_depth_resampled->set_debug_name("DLSS temporal normalized depth");
		}
		if (m_dummy_depth)
		{
			m_dummy_depth->set_debug_name("DLSS temporal dummy depth");
		}
		m_motion->set_debug_name("DLSS temporal motion vectors");
		m_motion_meta->set_debug_name("DLSS temporal motion metadata");
		m_motion_filtered->set_debug_name("DLSS temporal filtered motion vectors");
		m_motion_bias->set_debug_name("DLSS temporal motion bias mask");
		m_previous_motion->set_debug_name("DLSS temporal previous motion vectors");
		m_output->set_debug_name("DLSS temporal output");
		if (m_native_output)
		{
			m_native_output->set_debug_name("DLSS native output");
		}
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

	void dlss_upscale_pass::copy_motion_to_history(const vk::command_buffer& cmd)
	{
		m_motion_meta->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		m_previous_motion->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		VkImageCopy region{};
		region.srcSubresource = { m_motion_meta->aspect(), 0, 0, 1 };
		region.dstSubresource = { m_previous_motion->aspect(), 0, 0, 1 };
		region.extent = { m_motion_meta->width(), m_motion_meta->height(), 1 };
		vkCmdCopyImage(cmd, m_motion_meta->value, m_motion_meta->current_layout,
			m_previous_motion->value, m_previous_motion->current_layout, 1, &region);

		m_previous_motion->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_motion_meta->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
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

		m_last_native_evaluated = false;

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

		auto& streamline = get_streamline_dlss();
		if (!m_streamline_attempted)
		{
			m_streamline_attempted = true;
			streamline.initialize(*vk::get_current_renderer(), g_cfg.video.dlss_frame_generation.get());
		}

		streamline_dlss::mode requested_dlss_mode = streamline_dlss::mode::balanced;
		switch (g_cfg.video.dlss_quality.get())
		{
		case dlss_quality_mode::quality: requested_dlss_mode = streamline_dlss::mode::max_quality; break;
		case dlss_quality_mode::balanced: requested_dlss_mode = streamline_dlss::mode::balanced; break;
		case dlss_quality_mode::performance: requested_dlss_mode = streamline_dlss::mode::max_performance; break;
		case dlss_quality_mode::ultra_performance: requested_dlss_mode = streamline_dlss::mode::ultra_performance; break;
		case dlss_quality_mode::native: requested_dlss_mode = streamline_dlss::mode::dlaa; break;
		}

		const size2u requested_output_size = output_size;
		const size2u dlss_output_size = requested_output_size;
		streamline_dlss::mode selected_dlss_mode = requested_dlss_mode;
		const dlss_preset selected_dlss_preset = g_cfg.video.dlss_preset_selection.get();
		bool native_configuration_valid = streamline.available() &&
			((inputs.depth && inputs.depth->value && inputs.depth->samples() == 1) || inputs.allow_dummy_depth);

		if (!m_output || !m_previous_color || !m_motion || !m_motion_meta || !m_motion_filtered || !m_motion_bias || !m_previous_motion || m_input_format != src->format() ||
			m_previous_color->width() != src->width() ||
			m_previous_color->height() != src->height() || m_motion->width() != input_size.width ||
			m_motion_meta->width() != input_size.width || m_motion_filtered->width() != input_size.width ||
			m_motion_bias->width() != input_size.width ||
			m_motion->height() != input_size.height || m_motion_meta->height() != input_size.height ||
			m_motion_filtered->height() != input_size.height || m_motion_bias->height() != input_size.height ||
			m_output->width() != dlss_output_size.width ||
			m_output->height() != dlss_output_size.height)
		{
			if (!initialize_images(src, input_size, dlss_output_size))
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

		const bool native_output_available = m_native_output && m_native_output->value;
		native_configuration_valid &= native_output_available;
		if (!m_configuration_logged)
		{
			rsx_log.notice("DLSS: temporal configuration requested_mode=%u selected_mode=%u preset=%c input=%ux%u output=%ux%u native_valid=%u",
				static_cast<u32>(requested_dlss_mode), static_cast<u32>(selected_dlss_mode),
				static_cast<char>('A' + static_cast<u32>(selected_dlss_preset)),
				input_size.width, input_size.height, dlss_output_size.width, dlss_output_size.height,
				native_configuration_valid ? 1u : 0u);
			m_configuration_logged = true;
		}

		const bool camera_cut = m_has_history && inputs.has_camera_view_projection &&
			m_has_previous_camera_view_projection && camera_discontinuity(
				inputs.camera_view_projection, m_previous_camera_view_projection);

		u32 unexplained_change = 0;
		u32 confident_motion = 0;
		const bool have_scene_metrics = read_scene_change_counters(input_size, unexplained_change, confident_motion);
		const u64 pixel_count = static_cast<u64>(input_size.width) * input_size.height;
		const float changed_fraction = have_scene_metrics && pixel_count
			? static_cast<float>(unexplained_change) / pixel_count
			: 0.f;
		const float motion_fraction = have_scene_metrics && pixel_count
			? static_cast<float>(confident_motion) / pixel_count
			: 0.f;

		++m_scene_frames_since_reset;
		bool scene_cut = false;
		if (have_scene_metrics)
		{
			const bool changing = m_scene_was_changing
				? changed_fraction >= 0.55f
				: changed_fraction >= 0.85f;
			if (motion_fraction >= 0.02f)
			{
				m_scene_motion_meter_warm = true;
			}
			scene_cut = m_has_history && changing && !m_scene_was_changing &&
				m_scene_motion_meter_warm && motion_fraction < 0.02f &&
				m_scene_frames_since_reset >= 45;
			m_scene_was_changing = changing;
		}

		const bool reset = inputs.reset_history || !m_has_history || camera_cut || scene_cut;
		if (reset)
		{
			m_scene_frames_since_reset = 0;
			copy_current_to_history(cmd, src, input_size);
		}

		const bool source_depth_available = inputs.depth && inputs.depth->value && inputs.depth->samples() == 1;
		const bool depth_needs_resample = source_depth_available &&
			(inputs.depth->width() != input_size.width || inputs.depth->height() != input_size.height);
		vk::viewable_image* depth_for_temporal = source_depth_available ? inputs.depth : nullptr;
		bool source_depth_pushed = false;

		if (depth_needs_resample && m_depth_resampled && m_depth_resampled->value)
		{
			if (auto* depth_target = dynamic_cast<vk::render_target*>(inputs.depth))
			{
				depth_target->read_barrier(const_cast<vk::command_buffer&>(cmd));
			}
			inputs.depth->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			source_depth_pushed = true;

			m_depth_resampled->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			vk::get_compute_task<vk::temporal::depth_resample_pass>()->run(cmd, inputs.depth,
				m_depth_resampled.get(), { inputs.depth->width(), inputs.depth->height() }, input_size);
			insert_temporal_write_read_barrier(cmd, *m_depth_resampled);
			m_depth_resampled->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			inputs.depth->pop_layout(cmd);
			source_depth_pushed = false;
			depth_for_temporal = m_depth_resampled.get();
		}
		else if (source_depth_available && !depth_needs_resample)
		{
			if (auto* depth_target = dynamic_cast<vk::render_target*>(inputs.depth))
			{
				depth_target->read_barrier(const_cast<vk::command_buffer&>(cmd));
			}
			inputs.depth->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			source_depth_pushed = true;
		}
		else
		{
			if (!source_depth_available && inputs.allow_dummy_depth && m_dummy_depth && m_dummy_depth->value &&
				m_dummy_depth->width() == input_size.width && m_dummy_depth->height() == input_size.height)
			{
				if (m_dummy_depth->current_layout == VK_IMAGE_LAYOUT_UNDEFINED)
				{
					m_dummy_depth->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
					VkClearDepthStencilValue clear{ 1.f, 0 };
					const VkImageSubresourceRange range{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
					vkCmdClearDepthStencilImage(cmd, m_dummy_depth->value, m_dummy_depth->current_layout,
						&clear, 1, &range);
					m_dummy_depth->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				}
				depth_for_temporal = m_dummy_depth.get();
			}
			else
			{
				depth_for_temporal = nullptr;
			}
		}

		const bool has_depth = depth_for_temporal && depth_for_temporal->value;
		native_configuration_valid &= has_depth;

		src->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_previous_color->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		if (m_previous_motion->current_layout == VK_IMAGE_LAYOUT_UNDEFINED)
		{
			// The first motion pass only uses this image for the optional bias
			// probe, which is disabled on the reset frame. Establish a valid
			// sampled layout before the first real history copy arrives.
			m_previous_motion->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}

		if (m_motion->current_layout == VK_IMAGE_LAYOUT_GENERAL)
		{
			insert_temporal_write_read_barrier(cmd, *m_motion, VK_ACCESS_SHADER_WRITE_BIT);
		}
		if (m_motion_meta->current_layout == VK_IMAGE_LAYOUT_GENERAL)
		{
			insert_temporal_write_read_barrier(cmd, *m_motion_meta, VK_ACCESS_SHADER_WRITE_BIT);
		}
		if (m_motion_bias->current_layout == VK_IMAGE_LAYOUT_GENERAL)
		{
			insert_temporal_write_read_barrier(cmd, *m_motion_bias, VK_ACCESS_SHADER_WRITE_BIT);
		}
		m_motion->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		m_motion_meta->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		m_motion_bias->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		m_output->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		if (native_configuration_valid)
		{
			m_native_output->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		}

		std::array<float, 16> clip_to_previous{};
		const bool has_camera_pair = has_depth && !reset && inputs.has_camera_view_projection &&
			m_has_previous_camera_view_projection && make_clip_to_previous(
				inputs.camera_view_projection, m_previous_camera_view_projection, clip_to_previous);
		const float jitter_delta_x = !reset && m_has_previous_jitter
			? inputs.jitter_x - m_previous_jitter_x
			: 0.f;
		const float jitter_delta_y = !reset && m_has_previous_jitter
			? inputs.jitter_y - m_previous_jitter_y
			: 0.f;

		vk::get_compute_task<vk::temporal::motion_pass>()->run(cmd, src, m_previous_color.get(),
			has_depth ? depth_for_temporal : nullptr, m_motion.get(), m_motion_meta.get(), m_previous_motion.get(), m_motion_bias.get(),
			inputs.object_motion_valid ? inputs.object_motion : nullptr,
			input_size, dlss_output_size,
			has_camera_pair ? &clip_to_previous : nullptr, jitter_delta_x, jitter_delta_y,
			g_cfg.video.dlss_motion_dynamic_mask.get(), g_cfg.video.dlss_motion_far_rotation.get(),
			g_cfg.video.dlss_motion_edge_mode.get(), has_camera_pair ? 128.f : 32.f,
			prepare_scene_change_buffer(input_size), g_cfg.video.dlss_motion_bias.get(),
			inputs.object_motion_valid, reset);
		insert_temporal_write_read_barrier(cmd, *m_motion);
		insert_temporal_write_read_barrier(cmd, *m_motion_meta);
		insert_temporal_write_read_barrier(cmd, *m_motion_bias);
		m_motion->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_motion_filtered->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
		vk::get_compute_task<vk::temporal::motion_filter_pass>()->run(cmd, m_motion.get(), m_motion_filtered.get(), input_size);
		insert_temporal_write_read_barrier(cmd, *m_motion_filtered);
		m_motion_filtered->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		m_motion_bias->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		if (reset)
		{
			// Prime the previous-depth channel before the first resolve. History is
			// disabled on this frame, but the resource must still be initialized and
			// transitioned for the following frame's disocclusion test.
			copy_motion_to_history(cmd);
		}

		if (inputs.has_camera_view_projection)
		{
			m_previous_camera_view_projection = inputs.camera_view_projection;
			m_has_previous_camera_view_projection = true;
		}
		else
		{
			// Do not carry a camera pair across a frame where capture failed. A
			// stale matrix would turn the next valid depth sample into a fabricated
			// camera vector.
			m_has_previous_camera_view_projection = false;
		}
		m_previous_jitter_x = inputs.jitter_x;
		m_previous_jitter_y = inputs.jitter_y;
		m_has_previous_jitter = true;

		bool native_dlss_evaluated = false;
		if (!g_cfg.video.dlss_frame_generation.get() && streamline.frame_generation_proxy_armed())
		{
			streamline.prepare_swapchain_recreation();
		}
		if (has_depth && streamline.available() && native_configuration_valid)
		{
			const auto remap = rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY);
			const auto* input_view = src->get_view(remap);
			const auto* output_view = m_native_output->get_view(remap);
			const auto* depth_view = depth_for_temporal->get_view(remap,
				(depth_for_temporal->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
			const auto* motion_view = m_motion_filtered->get_view(remap);
			const auto* bias_view = m_motion_bias->get_view(remap);
			const bool use_raw_frame_generation_motion = g_cfg.video.dlss_frame_generation_raw_motion.get();
			auto* frame_generation_motion = use_raw_frame_generation_motion ? m_motion.get() : m_motion_filtered.get();
			const auto* frame_generation_motion_view = frame_generation_motion->get_view(remap);

			streamline_dlss::texture input_texture
			{
				src->value, input_view->value, src->format(), src->current_layout, src->width(), src->height()
			};
			streamline_dlss::texture output_texture
			{
				m_native_output->value, output_view->value, m_native_output->format(), m_native_output->current_layout,
				m_native_output->width(), m_native_output->height()
			};
			streamline_dlss::texture depth_texture
			{
				depth_for_temporal->value, depth_view->value, depth_for_temporal->format(), depth_for_temporal->current_layout,
				depth_for_temporal->width(), depth_for_temporal->height()
			};
			streamline_dlss::texture motion_texture
			{
				m_motion_filtered->value, motion_view->value, m_motion_filtered->format(), m_motion_filtered->current_layout,
				m_motion_filtered->width(), m_motion_filtered->height()
			};
			streamline_dlss::texture bias_texture
			{
				m_motion_bias->value, bias_view->value, m_motion_bias->format(), m_motion_bias->current_layout,
				m_motion_bias->width(), m_motion_bias->height()
			};
			streamline_dlss::texture frame_generation_motion_texture
			{
				frame_generation_motion->value, frame_generation_motion_view->value, frame_generation_motion->format(),
				frame_generation_motion->current_layout, frame_generation_motion->width(), frame_generation_motion->height()
			};

			if (streamline.set_options(inputs.viewport_id, selected_dlss_mode, dlss_output_size.width, dlss_output_size.height,
				is_hdr_color_format(src->format()), static_cast<u32>(selected_dlss_preset)))
			{
				native_dlss_evaluated = streamline.evaluate(cmd, inputs.viewport_id, static_cast<u32>(vk::get_current_frame_id()), reset,
					inputs.jitter_x, inputs.jitter_y, input_texture, output_texture, depth_texture, motion_texture,
					g_cfg.video.dlss_motion_bias.get() ? &bias_texture : nullptr);

				if (native_dlss_evaluated && inputs.allow_frame_generation && inputs.viewport_id == 0 &&
					g_cfg.video.dlss_frame_generation.get() &&
					streamline.frame_generation_available() && streamline.frame_generation_proxy_armed())
				{
					const u32 color_width = inputs.present_width ? inputs.present_width : requested_output_size.width;
					const u32 color_height = inputs.present_height ? inputs.present_height : requested_output_size.height;
					const VkFormat color_format = inputs.present_format != VK_FORMAT_UNDEFINED ? inputs.present_format : m_native_output->format();
					const u32 backbuffer_count = inputs.present_buffer_count ? inputs.present_buffer_count : 2;

					if (streamline.configure_frame_generation(0, color_width, color_height, color_format, backbuffer_count,
						g_cfg.video.dlss_frame_generation_frames.get(),
						depth_texture, frame_generation_motion_texture))
					{
						streamline.tag_frame(cmd, depth_texture, frame_generation_motion_texture);
					}
				}
			}
		}
		m_last_native_evaluated = native_dlss_evaluated;

		if (!native_dlss_evaluated)
		{
			vk::get_compute_task<vk::temporal::resolve_pass>()->run(cmd, src, m_previous_color.get(), m_motion_meta.get(), m_previous_motion.get(),
				has_depth ? depth_for_temporal : nullptr, m_output.get(), input_size, dlss_output_size, reset);
		}
		// Both the local resolve and Streamline's native evaluator write an image.
		// Make the selected write visible before the present blit or the caller's
		// shader-read transition; this keeps the native path from relying on an
		// implementation-specific Streamline barrier.
		vk::viewable_image* final_output = native_dlss_evaluated ? m_native_output.get() : m_output.get();
		insert_temporal_write_read_barrier(cmd, *final_output);
		copy_motion_to_history(cmd);

		if (source_depth_pushed)
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
			final_output->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

			VkImageBlit output_request = request;
			output_request.srcSubresource = { final_output->aspect(), 0, 0, 1 };
			output_request.srcOffsets[0] = { 0, 0, 0 };
			output_request.srcOffsets[1] = { static_cast<s32>(dlss_output_size.width), static_cast<s32>(dlss_output_size.height), 1 };
			if (request.srcOffsets[0].x > request.srcOffsets[1].x)
			{
				std::swap(output_request.srcOffsets[0].x, output_request.srcOffsets[1].x);
			}
			if (request.srcOffsets[0].y > request.srcOffsets[1].y)
			{
				std::swap(output_request.srcOffsets[0].y, output_request.srcOffsets[1].y);
			}

			vkCmdBlitImage(cmd, final_output->value, final_output->current_layout,
				present_surface, present_surface_layout, 1, &output_request, VK_FILTER_LINEAR);
			return nullptr;
		}

		final_output->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		return final_output;
	}

	bool dlss_upscale_pass::run_mid_frame(
		const vk::command_buffer& cmd,
		vk::viewable_image* src,
		vk::viewable_image* dst,
		vk::viewable_image* depth,
		const temporal_frame_inputs& inputs)
	{
		m_last_native_evaluated = false;

		if (!src || !dst || src == dst || src->samples() != 1 || dst->samples() != 1 ||
			!src->value || !dst->value || src->width() >= dst->width() || src->height() >= dst->height())
		{
			return false;
		}

		const size2u input_size{ src->width(), src->height() };
		const size2u output_size{ dst->width(), dst->height() };
		VkImageBlit request{};
		request.srcSubresource = { src->aspect(), 0, 0, 1 };
		request.dstSubresource = { dst->aspect(), 0, 0, 1 };
		request.srcOffsets[1] = { static_cast<s32>(input_size.width), static_cast<s32>(input_size.height), 1 };
		request.dstOffsets[1] = { static_cast<s32>(output_size.width), static_cast<s32>(output_size.height), 1 };

		temporal_frame_inputs midframe_inputs = inputs;
		midframe_inputs.depth = depth;
		midframe_inputs.viewport_id = 1;
		midframe_inputs.allow_frame_generation = false;
		midframe_inputs.allow_dummy_depth = true;
		midframe_inputs.present_width = 0;
		midframe_inputs.present_height = 0;
		midframe_inputs.present_buffer_count = 0;
		midframe_inputs.present_format = VK_FORMAT_UNDEFINED;
		const u64 frame_id = vk::get_current_frame_id();
		const bool resource_changed = m_midframe_source != src->value || m_midframe_destination != dst->value;
		const bool frame_gap = m_midframe_frame && frame_id > m_midframe_frame + 8;
		midframe_inputs.reset_history |= resource_changed || frame_gap || !m_midframe_native_history_valid;

		dst->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		scale_output_temporal(cmd, src, dst->value, dst->current_layout, request,
			UPSCALE_AND_COMMIT | UPSCALE_DEFAULT_VIEW, midframe_inputs);
		dst->change_layout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
		m_midframe_source = src->value;
		m_midframe_destination = dst->value;
		m_midframe_frame = frame_id;
		m_midframe_native_history_valid = m_last_native_evaluated;
		return m_last_native_evaluated;
	}
}
