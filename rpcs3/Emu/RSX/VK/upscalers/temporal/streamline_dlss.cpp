#include "stdafx.h"

#include "streamline_dlss.h"

#include "../../vkutils/device.h"
#include "rpcs3_version.h"

#include "Utilities/File.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace vk
{
	streamline_dlss g_streamline_dlss;

	streamline_dlss& get_streamline_dlss()
	{
		return g_streamline_dlss;
	}

	streamline_dlss::~streamline_dlss()
	{
		shutdown();
	}

	streamline_dlss::struct_type streamline_dlss::make_guid(u32 data1, u16 data2, u16 data3, std::initializer_list<u8> bytes)
	{
		struct_type result{ data1, data2, data3, {} };
		std::copy_n(bytes.begin(), std::min<size_t>(bytes.size(), std::size(result.bytes)), result.bytes);
		return result;
	}

	streamline_dlss::mat4 streamline_dlss::identity_matrix()
	{
		mat4 result{};
		result.values[0] = result.values[5] = result.values[10] = result.values[15] = 1.f;
		return result;
	}

	streamline_dlss::viewport_handle streamline_dlss::make_viewport(u32 viewport_id)
	{
		viewport_handle result{};
		result.type = make_guid(0x171b6435, 0x9b3c, 0x4fc8, { 0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4 });
		result.version = 1;
		result.value = viewport_id;
		return result;
	}

	streamline_dlss::resource streamline_dlss::make_resource(const texture& texture)
	{
		resource result{};
		result.type = make_guid(0x3a9d70cf, 0x2418, 0x4b72, { 0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61 });
		result.version = 1;
		result.resource_type = 0; // sl::ResourceType::eTex2d
		result.native = texture.image;
		result.view = texture.view;
		result.state = static_cast<u32>(texture.layout);
		result.width = texture.width;
		result.height = texture.height;
		result.native_format = texture.format;
		result.mip_levels = 1;
		result.array_layers = 1;
		return result;
	}

	streamline_dlss::resource_tag streamline_dlss::make_tag(resource* resource_ptr, u32 buffer_type, u32 width, u32 height, u32 lifecycle)
	{
		resource_tag result{};
		result.type = make_guid(0x4c6a5aad, 0xb445, 0x496c, { 0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53 });
		result.version = 1;
		result.resource_ptr = resource_ptr;
		result.buffer_type = buffer_type;
		result.lifecycle = lifecycle;
		result.extent = { 0, 0, width, height };
		return result;
	}

	streamline_dlss::constants streamline_dlss::make_constants(u32 output_width, u32 output_height,
		u32 render_width, u32 render_height, bool reset, float jitter_x, float jitter_y)
	{
		constants result{};
		result.type = make_guid(0xdcd35ad7, 0x4e4a, 0x4bad, { 0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe });
		result.version = 2;
		result.camera_view_to_clip = identity_matrix();
		result.clip_to_camera_view = identity_matrix();
		result.clip_to_lens_clip = identity_matrix();
		result.clip_to_prev_clip = identity_matrix();
		result.prev_clip_to_clip = identity_matrix();
		result.jitter_offset_x = jitter_x;
		result.jitter_offset_y = jitter_y;
		result.mvec_scale_x = render_width ? 1.f / render_width : 1.f;
		result.mvec_scale_y = render_height ? 1.f / render_height : 1.f;
		result.camera_up_y = 1.f;
		result.camera_right_x = 1.f;
		result.camera_fwd_z = 1.f;
		result.camera_near = 0.1f;
		result.camera_far = 10000.f;
		result.camera_fov = 1.f;
		result.camera_aspect_ratio = output_height ? static_cast<float>(output_width) / output_height : 1.7777f;
		result.motion_vectors_invalid_value = 0.f;
		result.depth_inverted = 0;
		result.camera_motion_included = 1;
		result.motion_vectors_3d = 0;
		result.reset = reset ? 1 : 0;
		result.orthographic_projection = 0;
		result.motion_vectors_dilated = 0;
		result.motion_vectors_jittered = 0;
		result.min_relative_linear_depth_object_separation = 40.f;
		return result;
	}

	bool streamline_dlss::load_interposer()
	{
#ifndef _WIN32
		// NVIDIA's Streamline distribution is Windows-only. Keeping this gate
		// explicit avoids probing arbitrary shared libraries on other hosts.
		return false;
#else
		if (m_interposer)
		{
			return true;
		}

		const char* configured_path = std::getenv("RPCS3_DLSS_PATH");
		std::string path = configured_path ? configured_path : "";
		bool loaded = false;

		if (!path.empty())
		{
			loaded = m_interposer.load(path);
			if (!loaded)
			{
				if (!path.ends_with(".dll") && !path.ends_with(".DLL"))
				{
					loaded = m_interposer.load(path + "/sl.interposer.dll");
				}
			}
		}

		if (!loaded)
		{
			loaded = m_interposer.load("sl.interposer.dll");
		}

		// Windows' default DLL search policy is not guaranteed to include the
		// current working directory. Resolve the bundled Streamline interposer
		// beside rpcs3.exe so launching a game from a shortcut or another cwd
		// behaves the same as a portable install.
		if (!loaded)
		{
			loaded = m_interposer.load(fs::get_executable_dir() + "/sl.interposer.dll");
		}

		if (!loaded)
		{
			rsx_log.notice("DLSS: Streamline interposer not found; Vulkan temporal fallback remains available");
			return false;
		}

		m_sl_init = m_interposer.get<sl_init_fn>("slInit");
		m_sl_shutdown = m_interposer.get<sl_shutdown_fn>("slShutdown");
		m_sl_is_feature_supported = m_interposer.get<sl_is_feature_supported_fn>("slIsFeatureSupported");
		m_sl_set_vulkan_info = m_interposer.get<sl_set_vulkan_info_fn>("slSetVulkanInfo");
		m_sl_get_new_frame_token = m_interposer.get<sl_get_new_frame_token_fn>("slGetNewFrameToken");
		m_sl_set_constants = m_interposer.get<sl_set_constants_fn>("slSetConstants");
		m_sl_evaluate_feature = m_interposer.get<sl_evaluate_feature_fn>("slEvaluateFeature");
		m_sl_get_feature_function = m_interposer.get<sl_get_feature_function_fn>("slGetFeatureFunction");
		m_sl_set_tag_for_frame = m_interposer.get<sl_set_tag_for_frame_fn>("slSetTagForFrame");
		m_sl_vk_get_device_proc_addr = m_interposer.get<sl_vk_get_device_proc_addr_fn>("vkGetDeviceProcAddr");

		const bool complete = m_sl_init && m_sl_shutdown && m_sl_is_feature_supported && m_sl_set_vulkan_info &&
			m_sl_get_new_frame_token && m_sl_set_constants && m_sl_evaluate_feature && m_sl_get_feature_function;
		if (!complete)
		{
			rsx_log.warning("DLSS: Streamline interposer is missing one or more required core exports");
			m_interposer.close();
			m_sl_init = nullptr;
			m_sl_shutdown = nullptr;
			m_sl_is_feature_supported = nullptr;
			m_sl_set_vulkan_info = nullptr;
			m_sl_get_new_frame_token = nullptr;
			m_sl_set_constants = nullptr;
			m_sl_evaluate_feature = nullptr;
			m_sl_get_feature_function = nullptr;
			m_sl_set_tag_for_frame = nullptr;
			m_sl_vk_get_device_proc_addr = nullptr;
		}

		return complete;
#endif
	}

	bool streamline_dlss::initialize(const vk::render_device& device, bool load_frame_generation)
	{
		if (m_initialized)
		{
			return true;
		}

		m_frame_generation_requested = load_frame_generation;

		if (!load_interposer())
		{
			return false;
		}

		m_features = load_frame_generation
			? std::vector<u32>{ 0, 1000, 3, 4 } // DLSS, DLSS-G, Reflex, PCL
			: std::vector<u32>{ 0 };             // sl::Feature::kFeatureDLSS
		m_log_path = L".";

		preferences pref{};
		pref.type = make_guid(0x1ca10965, 0xbf8e, 0x432b, { 0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14 });
		pref.version = 1;
		pref.log_level = 1;
		pref.path_to_logs_and_data = m_log_path.c_str();
		pref.flags = 1ull << 0; // eDisableCLStateTracking
		if (load_frame_generation)
		{
			pref.flags |= 1ull << 2; // eUseManualHooking
			pref.flags |= 1ull << 7; // eUseFrameBasedResourceTagging
		}
		pref.features_to_load = m_features.data();
		pref.num_features_to_load = static_cast<u32>(m_features.size());
		pref.engine = 0; // eCustom
		const std::string engine_version = rpcs3::get_version().to_string(true);
		pref.engine_version = engine_version.c_str();
		// UUID v5 derived from RPCS3's canonical repository URL.
		pref.project_id = "24cb96f8-8496-5200-9346-3f47864b0f68";
		pref.render_api = 2; // eVulkan

		constexpr u64 sdk_version = (u64(2) << 48) | (u64(12) << 32) | (u64(0) << 16) | 0xfedc;
		if (m_sl_init(pref, sdk_version) != 0)
		{
			rsx_log.warning("DLSS: Streamline initialization failed");
			m_interposer.close();
			m_sl_init = nullptr;
			m_sl_shutdown = nullptr;
			m_sl_is_feature_supported = nullptr;
			m_sl_set_vulkan_info = nullptr;
			m_sl_get_new_frame_token = nullptr;
			m_sl_set_constants = nullptr;
			m_sl_evaluate_feature = nullptr;
			m_sl_get_feature_function = nullptr;
			m_sl_set_tag_for_frame = nullptr;
			m_sl_vk_get_device_proc_addr = nullptr;
			return false;
		}

		vulkan_info info{};
		info.type = make_guid(0x0eed6fd5, 0x82cd, 0x43a9, { 0xbd, 0xb5, 0x47, 0xa5, 0xba, 0x2f, 0x45, 0xd6 });
		info.version = 3;
		info.device = static_cast<VkDevice>(device);
		info.instance = static_cast<VkInstance>(device.gpu());
		info.physical_device = static_cast<VkPhysicalDevice>(device.gpu());
		info.compute_queue_family = device.get_graphics_queue_family();
		info.graphics_queue_family = device.get_graphics_queue_family();
		info.optical_flow_queue_family = device.get_graphics_queue_family();
		// Streamline's queues begin after the queue slots retained by RPCS3. The
		// device creation path expands the graphics-family queue create info when
		// FG is requested; if the physical device has no spare queue, SL rejects
		// this registration and the integration falls back safely.
		const u32 streamline_queue_start = load_frame_generation ? device.get_streamline_queue_start_index() : 0;
		info.compute_queue_index = streamline_queue_start;
		info.graphics_queue_index = streamline_queue_start;
		info.optical_flow_queue_index = streamline_queue_start;

		m_initialized = true;
		if (m_sl_set_vulkan_info(info) != 0)
		{
			rsx_log.warning("DLSS: slSetVulkanInfo rejected the RPCS3 Vulkan device");
			shutdown();
			return false;
		}

		// RPCS3 keeps Vulkan's native present path instead of loading Streamline
		// as the Vulkan loader. Bind the common plugin's present hooks so its
		// per-frame bookkeeping and garbage collection still run.
		void* hook_present = nullptr;
		void* hook_after_present = nullptr;
		if (m_sl_get_feature_function(0xffffffffu, "slHookVkPresent", &hook_present) == 0 && hook_present)
		{
			m_sl_hook_vk_present = reinterpret_cast<sl_hook_vk_present_fn>(hook_present);
		}
		if (m_sl_get_feature_function(0xffffffffu, "slHookVkAfterPresent", &hook_after_present) == 0 && hook_after_present)
		{
			m_sl_hook_vk_after_present = reinterpret_cast<sl_hook_vk_after_present_fn>(hook_after_present);
		}

		adapter_info adapter{};
		adapter.type = make_guid(0x0677315f, 0xa746, 0x4492, { 0x9f, 0x42, 0xcb, 0x61, 0x42, 0xc9, 0xc3, 0xd4 });
		adapter.version = 1;
		adapter.vk_physical_device = static_cast<VkPhysicalDevice>(device.gpu());

		m_dlss_supported = m_sl_is_feature_supported(0, adapter) == 0;
		m_frame_generation_supported = load_frame_generation && m_sl_is_feature_supported(1000, adapter) == 0;
		if (!m_dlss_supported)
		{
			rsx_log.notice("DLSS: Streamline loaded but DLSS is not supported by this adapter");
			shutdown();
			return false;
		}
		if (load_frame_generation && !m_frame_generation_supported)
		{
			rsx_log.notice("DLSS-FG: Streamline loaded but DLSS-G is not supported; frame generation remains disabled");
		}

		rsx_log.notice("DLSS: Streamline initialized and DLSS is available");
		return true;
	}

	void streamline_dlss::shutdown()
	{
		if (m_frame_generation_configured && m_sl_dlssg_set_options)
		{
			auto options = m_last_fg_options;
			options.mode = 0; // sl::DLSSGMode::eOff
			const auto viewport = make_viewport(m_fg_viewport_id);
			m_sl_dlssg_set_options(viewport, options);
		}

		if (m_initialized && m_sl_shutdown)
		{
			m_sl_shutdown();
		}

		m_initialized = false;
		m_dlss_supported = false;
		m_functions_bound = false;
		m_frame_generation_requested = false;
		m_frame_generation_supported = false;
		m_frame_generation_proxy_armed = false;
		m_frame_generation_functions_bound = false;
		m_reflex_enabled = false;
		m_frame_generation_configured = false;
		m_frame_generation_frame_active = false;
		m_fg_state_failed_logged = false;
		m_fg_color_width = 0;
		m_fg_color_height = 0;
		m_fg_motion_width = 0;
		m_fg_motion_height = 0;
		m_fg_depth_width = 0;
		m_fg_depth_height = 0;
		m_fg_backbuffer_count = 0;
		m_fg_color_format = VK_FORMAT_UNDEFINED;
		m_fg_motion_format = VK_FORMAT_UNDEFINED;
		m_fg_depth_format = VK_FORMAT_UNDEFINED;
		m_fg_frames_to_generate = 1;
		m_fg_seen_color_width = 0;
		m_fg_seen_color_height = 0;
		m_fg_seen_motion_width = 0;
		m_fg_seen_motion_height = 0;
		m_fg_seen_depth_width = 0;
		m_fg_seen_depth_height = 0;
		m_fg_seen_backbuffer_count = 0;
		m_fg_seen_color_format = VK_FORMAT_UNDEFINED;
		m_fg_seen_motion_format = VK_FORMAT_UNDEFINED;
		m_fg_seen_depth_format = VK_FORMAT_UNDEFINED;
		m_fg_size_stable_frames = 0;
		m_last_fg_options = {};
		m_evaluate_success_logged = false;
		m_evaluate_failure_logged = false;
		m_last_frame_token = nullptr;
		m_options_set = false;
		m_sl_dlss_set_options = nullptr;
		m_sl_dlss_get_optimal_settings = nullptr;
		m_sl_set_tag_for_frame = nullptr;
		m_sl_vk_get_device_proc_addr = nullptr;
		m_sl_reflex_set_options = nullptr;
		m_sl_reflex_sleep = nullptr;
		m_sl_pcl_set_marker = nullptr;
		m_sl_dlssg_set_options = nullptr;
		m_sl_dlssg_get_state = nullptr;
		m_sl_hook_vk_present = nullptr;
		m_sl_hook_vk_after_present = nullptr;
		m_present_hook_failure_logged = false;
		m_sl_init = nullptr;
		m_sl_shutdown = nullptr;
		m_sl_is_feature_supported = nullptr;
		m_sl_set_vulkan_info = nullptr;
		m_sl_get_new_frame_token = nullptr;
		m_sl_set_constants = nullptr;
		m_sl_evaluate_feature = nullptr;
		m_sl_get_feature_function = nullptr;
		m_interposer.close();
	}

	bool streamline_dlss::bind_feature_functions()
	{
		if (m_functions_bound)
		{
			return true;
		}

		void* set_options = nullptr;
		void* get_optimal = nullptr;
		if (!m_sl_get_feature_function || m_sl_get_feature_function(0, "slDLSSSetOptions", &set_options) != 0 ||
			m_sl_get_feature_function(0, "slDLSSGetOptimalSettings", &get_optimal) != 0 || !set_options || !get_optimal)
		{
			rsx_log.warning("DLSS: Streamline DLSS feature functions could not be bound");
			return false;
		}

		m_sl_dlss_set_options = reinterpret_cast<sl_dlss_set_options_fn>(set_options);
		m_sl_dlss_get_optimal_settings = reinterpret_cast<sl_dlss_get_optimal_settings_fn>(get_optimal);
		m_functions_bound = true;
		return true;
	}

	bool streamline_dlss::set_options(u32 viewport_id, mode dlss_mode, u32 output_width, u32 output_height, bool hdr, u32 preset)
	{
		if (!available() || !bind_feature_functions())
		{
			return false;
		}

		if (m_options_set && m_viewport_id == viewport_id && m_mode == dlss_mode && m_output_width == output_width &&
			m_output_height == output_height && m_hdr == hdr && m_preset == preset)
		{
			return true;
		}

		dlss_options options{};
		options.type = make_guid(0x6ac826e4, 0x4c61, 0x4101, { 0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8 });
		options.version = 3;
		options.mode = static_cast<u32>(dlss_mode);
		options.output_width = output_width;
		options.output_height = output_height;
		options.pre_exposure = 1.f;
		options.exposure_scale = 1.f;
		options.color_buffers_hdr = hdr ? 1 : 0;
		options.use_auto_exposure = 1;
		// Apply the user-selected letter to every DLSS quality slot. The selected
		// mode still controls the requested input/output ratio; the preset controls
		// which model variant Streamline uses for that ratio.
		const u32 streamline_preset = preset + 1;
		options.dlaa_preset = streamline_preset;
		options.quality_preset = streamline_preset;
		options.balanced_preset = streamline_preset;
		options.performance_preset = streamline_preset;
		options.ultra_performance_preset = streamline_preset;
		options.ultra_quality_preset = streamline_preset;

		const auto viewport = make_viewport(viewport_id);
		if (m_sl_dlss_set_options(viewport, options) != 0)
		{
			rsx_log.warning("DLSS: slDLSSSetOptions failed");
			return false;
		}
		rsx_log.notice("DLSS: native options accepted mode=%u preset=%u output=%ux%u hdr=%u",
			static_cast<u32>(dlss_mode), streamline_preset, output_width, output_height, hdr ? 1u : 0u);

		m_options_set = true;
		m_viewport_id = viewport_id;
		m_mode = dlss_mode;
		m_output_width = output_width;
		m_output_height = output_height;
		m_hdr = hdr;
		m_preset = preset;
		return true;
	}

	bool streamline_dlss::get_optimal_render_size(mode dlss_mode, u32 output_width, u32 output_height,
		u32& render_width, u32& render_height)
	{
		render_width = render_height = 0;

		if (!available() || !bind_feature_functions() || !output_width || !output_height)
		{
			return false;
		}

		dlss_options options{};
		options.type = make_guid(0x6ac826e4, 0x4c61, 0x4101, { 0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8 });
		options.version = 3;
		options.mode = static_cast<u32>(dlss_mode);
		options.output_width = output_width;
		options.output_height = output_height;

		dlss_optimal_settings settings{};
		settings.type = make_guid(0xef1d0957, 0xfd58, 0x4df7, { 0xb5, 0x04, 0x8b, 0x69, 0xd8, 0xaa, 0x6b, 0x76 });
		settings.version = 1;

		if (m_sl_dlss_get_optimal_settings(options, settings) != 0 ||
			!settings.optimal_render_width || !settings.optimal_render_height)
		{
			return false;
		}

		render_width = settings.optimal_render_width;
		render_height = settings.optimal_render_height;
		return true;
	}

	bool streamline_dlss::get_render_range(mode dlss_mode, u32 output_width, u32 output_height,
		u32& min_width, u32& min_height, u32& max_width, u32& max_height)
	{
		min_width = min_height = max_width = max_height = 0;

		if (!available() || !bind_feature_functions() || !output_width || !output_height)
		{
			return false;
		}

		dlss_options options{};
		options.type = make_guid(0x6ac826e4, 0x4c61, 0x4101, { 0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8 });
		options.version = 3;
		options.mode = static_cast<u32>(dlss_mode);
		options.output_width = output_width;
		options.output_height = output_height;

		dlss_optimal_settings settings{};
		settings.type = make_guid(0xef1d0957, 0xfd58, 0x4df7, { 0xb5, 0x04, 0x8b, 0x69, 0xd8, 0xaa, 0x6b, 0x76 });
		settings.version = 1;

		if (m_sl_dlss_get_optimal_settings(options, settings) != 0 ||
			!settings.optimal_render_width || !settings.optimal_render_height)
		{
			return false;
		}

		min_width = settings.render_width_min;
		min_height = settings.render_height_min;
		max_width = settings.render_width_max;
		max_height = settings.render_height_max;
		return min_width && min_height && max_width && max_height;
	}

	bool streamline_dlss::bind_frame_generation_functions()
	{
		if (m_frame_generation_functions_bound)
		{
			return true;
		}

		if (!frame_generation_available() || !m_sl_get_feature_function)
		{
			return false;
		}

		void* dlssg_set_options = nullptr;
		void* dlssg_get_state = nullptr;
		void* reflex_set_options = nullptr;
		void* reflex_sleep = nullptr;
		void* pcl_set_marker = nullptr;

		const bool bound =
			m_sl_get_feature_function(1000, "slDLSSGSetOptions", &dlssg_set_options) == 0 && dlssg_set_options &&
			m_sl_get_feature_function(1000, "slDLSSGGetState", &dlssg_get_state) == 0 && dlssg_get_state &&
			m_sl_get_feature_function(3, "slReflexSetOptions", &reflex_set_options) == 0 && reflex_set_options &&
			m_sl_get_feature_function(3, "slReflexSleep", &reflex_sleep) == 0 && reflex_sleep &&
			m_sl_get_feature_function(4, "slPCLSetMarker", &pcl_set_marker) == 0 && pcl_set_marker;

		if (!bound)
		{
			rsx_log.warning("DLSS-FG: Streamline frame-generation feature functions could not be bound");
			return false;
		}

		m_sl_dlssg_set_options = reinterpret_cast<sl_dlssg_set_options_fn>(dlssg_set_options);
		m_sl_dlssg_get_state = reinterpret_cast<sl_dlssg_get_state_fn>(dlssg_get_state);
		m_sl_reflex_set_options = reinterpret_cast<sl_reflex_set_options_fn>(reflex_set_options);
		m_sl_reflex_sleep = reinterpret_cast<sl_reflex_sleep_fn>(reflex_sleep);
		m_sl_pcl_set_marker = reinterpret_cast<sl_pcl_set_marker_fn>(pcl_set_marker);
		m_frame_generation_functions_bound = true;
		return true;
	}

	bool streamline_dlss::configure_frame_generation(u32 viewport_id, u32 color_width, u32 color_height, VkFormat color_format,
		u32 backbuffer_count, u32 frames_to_generate, const texture& depth, const texture& motion)
	{
		if (!frame_generation_available() || !m_frame_generation_proxy_armed || !bind_frame_generation_functions() || !m_sl_set_tag_for_frame ||
			color_width == 0 || color_height == 0 || color_format == VK_FORMAT_UNDEFINED ||
			depth.width == 0 || depth.height == 0 || depth.format == VK_FORMAT_UNDEFINED ||
			motion.width == 0 || motion.height == 0 || motion.format == VK_FORMAT_UNDEFINED)
		{
			return false;
		}

		if (!m_reflex_enabled)
		{
			reflex_options reflex{};
			reflex.type = make_guid(0xf03af81a, 0x6d0b, 0x4902, { 0xa6, 0x51, 0xc4, 0x96, 0x5e, 0x21, 0x54, 0x34 });
			reflex.version = 1;
			reflex.mode = 1; // sl::ReflexMode::eLowLatency

			if (m_sl_reflex_set_options(reflex) != 0)
			{
				rsx_log.warning("DLSS-FG: slReflexSetOptions(eLowLatency) failed");
				return false;
			}

			m_reflex_enabled = true;
		}

		const u32 requested_frames = std::clamp(frames_to_generate, 1u, 5u);
		// Pass the actual swapchain count through. Streamline's DLSS-G option is
		// describing the intercepted present chain, not asking the plugin to
		// synthesize a minimum number of buffers.
		const u32 requested_backbuffers = backbuffer_count;
		const bool shape_changed =
			m_fg_seen_color_width != color_width || m_fg_seen_color_height != color_height ||
			m_fg_seen_motion_width != motion.width || m_fg_seen_motion_height != motion.height ||
			m_fg_seen_depth_width != depth.width || m_fg_seen_depth_height != depth.height ||
			m_fg_seen_color_format != color_format || m_fg_seen_motion_format != motion.format ||
			m_fg_seen_depth_format != depth.format || m_fg_seen_backbuffer_count != requested_backbuffers;

		if (shape_changed)
		{
			m_fg_seen_color_width = color_width;
			m_fg_seen_color_height = color_height;
			m_fg_seen_motion_width = motion.width;
			m_fg_seen_motion_height = motion.height;
			m_fg_seen_depth_width = depth.width;
			m_fg_seen_depth_height = depth.height;
			m_fg_seen_backbuffer_count = requested_backbuffers;
			m_fg_seen_color_format = color_format;
			m_fg_seen_motion_format = motion.format;
			m_fg_seen_depth_format = depth.format;
			m_fg_size_stable_frames = 1;

			if (m_frame_generation_configured)
			{
				auto disable = m_last_fg_options;
				disable.mode = 0; // sl::DLSSGMode::eOff
				const auto old_viewport = make_viewport(m_fg_viewport_id);
				m_sl_dlssg_set_options(old_viewport, disable);
				m_frame_generation_configured = false;
				m_frame_generation_frame_active = false;
			}

			// A swapchain or input resize must be observed for several complete
			// frames before DLSS-G is enabled again. This mirrors Streamline's
			// documented queue-flush requirement and avoids enabling it against
			// transient dimensions during a window resize.
			return false;
		}

		if (m_frame_generation_configured &&
			m_fg_viewport_id == viewport_id &&
			m_fg_color_width == color_width && m_fg_color_height == color_height &&
			m_fg_motion_width == motion.width && m_fg_motion_height == motion.height &&
			m_fg_depth_width == depth.width && m_fg_depth_height == depth.height &&
			m_fg_backbuffer_count == requested_backbuffers &&
			m_fg_color_format == color_format && m_fg_motion_format == motion.format &&
			m_fg_depth_format == depth.format && m_fg_frames_to_generate == requested_frames)
		{
			return true;
		}

		// Reconfiguration is deliberately conservative. The Beast integration
		// holds dimensions for 30 complete frames after a resize before turning
		// DLSS-G back on; the worker/present queues need time to drain and settle.
		if (!m_frame_generation_configured && m_fg_size_stable_frames < 30)
		{
			++m_fg_size_stable_frames;
			return false;
		}

		// Streamline requires DLSS-G to be disabled before its swapchain-sized
		// inputs are changed. Reuse the accepted option block for the transition;
		// a zeroed block is rejected by current Streamline releases.
		if (m_frame_generation_configured)
		{
			auto disable = m_last_fg_options;
			disable.mode = 0; // sl::DLSSGMode::eOff
			const auto old_viewport = make_viewport(m_fg_viewport_id);
			m_sl_dlssg_set_options(old_viewport, disable);
			m_frame_generation_configured = false;
			m_frame_generation_frame_active = false;
		}

		dlssg_options options{};
		options.type = make_guid(0xfac5f1cb, 0x2dfd, 0x4f36, { 0xa1, 0xe6, 0x3a, 0x9e, 0x86, 0x52, 0x56, 0xc5 });
		options.version = 5;
		options.mode = 1; // sl::DLSSGMode::eOn
		options.num_frames_to_generate = requested_frames;
		options.num_back_buffers = requested_backbuffers;
		options.mvec_depth_width = motion.width;
		options.mvec_depth_height = motion.height;
		options.color_width = color_width;
		options.color_height = color_height;
		options.color_buffer_format = color_format;
		options.mvec_buffer_format = motion.format;
		options.depth_buffer_format = depth.format;
		options.reserved15 = 2; // sl::Boolean::eInvalid

		const auto viewport = make_viewport(viewport_id);
		if (m_sl_dlssg_set_options(viewport, options) != 0)
		{
			rsx_log.warning("DLSS-FG: slDLSSGSetOptions(eOn) failed");
			return false;
		}

		m_fg_viewport_id = viewport_id;
		m_fg_color_width = color_width;
		m_fg_color_height = color_height;
		m_fg_motion_width = motion.width;
		m_fg_motion_height = motion.height;
		m_fg_depth_width = depth.width;
		m_fg_depth_height = depth.height;
		m_fg_backbuffer_count = requested_backbuffers;
		m_fg_color_format = color_format;
		m_fg_motion_format = motion.format;
		m_fg_depth_format = depth.format;
		m_fg_frames_to_generate = requested_frames;
		m_last_fg_options = options;
		m_frame_generation_configured = true;
		return true;
	}

	bool streamline_dlss::tag_frame(VkCommandBuffer command_buffer, const texture& depth, const texture& motion)
	{
		if (!m_frame_generation_configured || !m_last_frame_token || !m_sl_set_tag_for_frame)
		{
			return false;
		}

		resource depth_resource = make_resource(depth);
		resource motion_resource = make_resource(motion);
		resource_tag tags[] =
		{
			make_tag(&depth_resource, 0, depth.width, depth.height, 1),
			make_tag(&motion_resource, 1, motion.width, motion.height, 1)
		};

		const auto viewport = make_viewport(m_fg_viewport_id);
		if (m_sl_set_tag_for_frame(m_last_frame_token, viewport, tags, static_cast<u32>(std::size(tags)), command_buffer) != 0)
		{
			if (!m_frame_generation_frame_active)
			{
				rsx_log.warning("DLSS-FG: slSetTagForFrame failed; frame generation input tagging is inactive");
			}
			m_frame_generation_frame_active = false;
			return false;
		}

		m_frame_generation_frame_active = true;
		if (m_sl_pcl_set_marker)
		{
			// The emulator does not expose a separately paced simulation loop;
			// bracket that phase honestly, then mark the GPU submission start.
			m_sl_pcl_set_marker(0, m_last_frame_token); // eSimulationStart
			m_sl_pcl_set_marker(1, m_last_frame_token); // eSimulationEnd
			m_sl_pcl_set_marker(2, m_last_frame_token); // eRenderSubmitStart
		}
		return true;
	}

	void streamline_dlss::prepare_swapchain_recreation()
	{
		if (m_frame_generation_configured && m_sl_dlssg_set_options)
		{
			auto disable = m_last_fg_options;
			disable.mode = 0; // sl::DLSSGMode::eOff
			const auto viewport = make_viewport(m_fg_viewport_id);
			m_sl_dlssg_set_options(viewport, disable);
		}

		m_frame_generation_configured = false;
		m_frame_generation_frame_active = false;
		m_last_frame_token = nullptr;
		m_fg_seen_color_width = 0;
		m_fg_seen_color_height = 0;
		m_fg_seen_motion_width = 0;
		m_fg_seen_motion_height = 0;
		m_fg_seen_depth_width = 0;
		m_fg_seen_depth_height = 0;
		m_fg_seen_backbuffer_count = 0;
		m_fg_seen_color_format = VK_FORMAT_UNDEFINED;
		m_fg_seen_motion_format = VK_FORMAT_UNDEFINED;
		m_fg_seen_depth_format = VK_FORMAT_UNDEFINED;
		m_fg_size_stable_frames = 0;
	}

	void streamline_dlss::before_present(VkQueue queue)
	{
		if (!m_frame_generation_proxy_armed && m_sl_hook_vk_present && queue)
		{
			bool skip = false;
			if (m_sl_hook_vk_present(queue, nullptr, skip) != VK_SUCCESS && !m_present_hook_failure_logged)
			{
				rsx_log.warning("DLSS: Streamline Vulkan present hook failed");
				m_present_hook_failure_logged = true;
			}
		}

		if (!m_frame_generation_frame_active || !m_last_frame_token || !m_sl_pcl_set_marker)
		{
			return;
		}

		// eRenderSubmitEnd, ePresentStart
		m_sl_pcl_set_marker(3, m_last_frame_token);
		m_sl_pcl_set_marker(4, m_last_frame_token);
	}

	void streamline_dlss::after_present()
	{
		if (!m_frame_generation_proxy_armed && m_sl_hook_vk_after_present &&
			m_sl_hook_vk_after_present() != VK_SUCCESS && !m_present_hook_failure_logged)
		{
			rsx_log.warning("DLSS: Streamline Vulkan after-present hook failed");
			m_present_hook_failure_logged = true;
		}

		if (!m_last_frame_token)
		{
			return;
		}

		if (m_frame_generation_frame_active)
		{
			if (m_sl_pcl_set_marker)
			{
				m_sl_pcl_set_marker(5, m_last_frame_token); // ePresentEnd
			}

			if (m_sl_reflex_sleep)
			{
				m_sl_reflex_sleep(m_last_frame_token);
			}

			if (m_sl_dlssg_get_state)
			{
				dlssg_state state{};
				state.type = make_guid(0xcc8ac8e1, 0xa179, 0x44f5, { 0x97, 0xfa, 0xe7, 0x41, 0x12, 0xf9, 0xbc, 0x61 });
				state.version = 4;
				const auto viewport = make_viewport(m_fg_viewport_id);
				if (m_sl_dlssg_get_state(viewport, state, nullptr) != 0 && !m_fg_state_failed_logged)
				{
					rsx_log.warning("DLSS-FG: slDLSSGGetState failed while polling present statistics");
					m_fg_state_failed_logged = true;
				}
			}
		}

		m_frame_generation_frame_active = false;
		m_last_frame_token = nullptr;
	}

	void* streamline_dlss::get_device_proc_addr(VkDevice device, const char* name) const
	{
		if (!m_sl_vk_get_device_proc_addr || !device || !name)
		{
			return nullptr;
		}

		return reinterpret_cast<void*>(m_sl_vk_get_device_proc_addr(device, name));
	}

	VkResult streamline_dlss::device_wait_idle(VkDevice device) const
	{
		if (m_frame_generation_proxy_armed)
		{
			if (const auto wait_idle = reinterpret_cast<PFN_vkDeviceWaitIdle>(get_device_proc_addr(device, "vkDeviceWaitIdle")))
			{
				return wait_idle(device);
			}
		}

		return vkDeviceWaitIdle(device);
	}

	bool streamline_dlss::evaluate(VkCommandBuffer command_buffer,
		u32 viewport_id,
		u32 frame_index,
		bool reset,
		float jitter_x,
		float jitter_y,
		const texture& input,
		const texture& output,
		const texture& depth,
		const texture& motion,
		const texture* bias)
	{
		if (!available() || input.format == VK_FORMAT_UNDEFINED || output.format == VK_FORMAT_UNDEFINED ||
			depth.format == VK_FORMAT_UNDEFINED || motion.format == VK_FORMAT_UNDEFINED)
		{
			return false;
		}

		void* frame_token = nullptr;
		const sl_result frame_token_result = m_sl_get_new_frame_token(&frame_token, frame_index);
		if (frame_token_result != 0 || !frame_token)
		{
			if (!m_evaluate_failure_logged)
			{
				rsx_log.warning("DLSS: native evaluation could not acquire a frame token (result=%d)", frame_token_result);
				m_evaluate_failure_logged = true;
			}
			return false;
		}

		const auto viewport = make_viewport(viewport_id);
		const auto frame_constants = make_constants(output.width, output.height, motion.width, motion.height, reset, jitter_x, jitter_y);
		const sl_result constants_result = m_sl_set_constants(frame_constants, frame_token, viewport);
		if (constants_result != 0)
		{
			if (!m_evaluate_failure_logged)
			{
				rsx_log.warning("DLSS: native evaluation rejected frame constants (result=%d)", constants_result);
				m_evaluate_failure_logged = true;
			}
			m_last_frame_token = nullptr;
			return false;
		}

		resource input_resource = make_resource(input);
		resource output_resource = make_resource(output);
		resource depth_resource = make_resource(depth);
		resource motion_resource = make_resource(motion);
		const bool use_bias = bias && bias->image && bias->view && bias->width && bias->height &&
			bias->format != VK_FORMAT_UNDEFINED;
		resource bias_resource{};
		if (use_bias)
		{
			bias_resource = make_resource(*bias);
		}
		resource_tag input_tag = make_tag(&input_resource, 3, input.width, input.height);   // ScalingInputColor
		resource_tag output_tag = make_tag(&output_resource, 4, output.width, output.height); // ScalingOutputColor
		resource_tag depth_tag = make_tag(&depth_resource, 0, depth.width, depth.height);   // Depth
		resource_tag motion_tag = make_tag(&motion_resource, 1, motion.width, motion.height); // MotionVectors
		resource_tag bias_tag{};
		if (use_bias)
		{
			bias_tag = make_tag(&bias_resource, 29, bias->width, bias->height);
		}

		void* inputs[] = { const_cast<viewport_handle*>(&viewport), &depth_tag, &motion_tag, &input_tag, &output_tag, &bias_tag };
		const u32 input_count = use_bias ? 6 : 5;
		const sl_result evaluate_result = m_sl_evaluate_feature(0, frame_token, inputs, input_count, command_buffer);
		const bool evaluated = evaluate_result == 0;
		if (evaluated && !m_evaluate_success_logged)
		{
			rsx_log.notice("DLSS: native evaluation succeeded input=%ux%u output=%ux%u depth=%ux%u motion=%ux%u",
				input.width, input.height, output.width, output.height, depth.width, depth.height, motion.width, motion.height);
			m_evaluate_success_logged = true;
		}
		else if (!evaluated && !m_evaluate_failure_logged)
		{
			rsx_log.warning("DLSS: native evaluation failed (result=%d) input=%ux%u output=%ux%u depth=%ux%u motion=%ux%u",
				evaluate_result, input.width, input.height, output.width, output.height, depth.width, depth.height, motion.width, motion.height);
			m_evaluate_failure_logged = true;
		}
		m_last_frame_token = evaluated ? frame_token : nullptr;
		return evaluated;
	}
}
