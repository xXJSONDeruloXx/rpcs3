#pragma once

#include "../../VulkanAPI.h"

#include "util/dyn_lib.hpp"

#include <initializer_list>
#include <string>
#include <vector>

namespace vk
{
	class render_device;

	// Clean-room declarations of the small Streamline ABI surface needed by
	// DLSS-SR. NVIDIA's interposer and DLSS plugins remain user-supplied; no
	// proprietary binary or SDK header is part of RPCS3.
	class streamline_dlss final
	{
	public:
		enum class mode : u32
		{
			max_performance = 1,
			balanced = 2,
			max_quality = 3,
			ultra_performance = 4,
			ultra_quality = 5,
			dlaa = 6
		};

		struct texture
		{
			VkImage image = VK_NULL_HANDLE;
			VkImageView view = VK_NULL_HANDLE;
			VkFormat format = VK_FORMAT_UNDEFINED;
			VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
			u32 width = 0;
			u32 height = 0;
		};

		streamline_dlss() = default;
		~streamline_dlss();

		streamline_dlss(const streamline_dlss&) = delete;
		streamline_dlss& operator=(const streamline_dlss&) = delete;

		bool initialize(const vk::render_device& device, bool load_frame_generation = false);
		void shutdown();

		bool available() const { return m_initialized && m_dlss_supported; }
		bool initialized() const { return m_initialized; }
		bool frame_generation_available() const { return m_initialized && m_frame_generation_supported; }
		bool frame_generation_proxy_armed() const { return m_frame_generation_proxy_armed; }
		void set_frame_generation_proxy_armed(bool armed) { m_frame_generation_proxy_armed = armed; }

		bool set_options(u32 viewport_id, mode dlss_mode, u32 output_width, u32 output_height, bool hdr, u32 preset);
		bool get_optimal_render_size(mode dlss_mode, u32 output_width, u32 output_height,
			u32& render_width, u32& render_height);
		bool get_render_range(mode dlss_mode, u32 output_width, u32 output_height,
			u32& min_width, u32& min_height, u32& max_width, u32& max_height);
		bool evaluate(VkCommandBuffer command_buffer,
			u32 viewport_id,
			u32 frame_index,
			bool reset,
			float jitter_x,
			float jitter_y,
			const texture& input,
			const texture& output,
			const texture& depth,
			const texture& motion,
			const texture* bias = nullptr);

		bool configure_frame_generation(u32 viewport_id, u32 color_width, u32 color_height, VkFormat color_format,
			u32 backbuffer_count, u32 frames_to_generate, const texture& depth, const texture& motion);
		bool tag_frame(VkCommandBuffer command_buffer, const texture& depth, const texture& motion);
		void prepare_swapchain_recreation();
		void before_present(VkQueue queue);
		void after_present();
		void* get_device_proc_addr(VkDevice device, const char* name) const;
		VkResult device_wait_idle(VkDevice device) const;

	private:
		struct struct_type
		{
			u32 data1;
			u16 data2;
			u16 data3;
			u8 bytes[8];
		};

		struct viewport_handle
		{
			void* next;
			struct_type type;
			size_t version;
			u32 value;
		};

		struct resource
		{
			void* next;
			struct_type type;
			size_t version;
			u8 resource_type;
			VkImage native;
			void* memory;
			VkImageView view;
			u32 state;
			u32 width;
			u32 height;
			VkFormat native_format;
			u32 mip_levels;
			u32 array_layers;
			u64 gpu_virtual_address;
			u32 flags;
			u32 usage;
			u32 reserved;
		};

		struct extent
		{
			u32 top;
			u32 left;
			u32 width;
			u32 height;
		};

		struct resource_tag
		{
			void* next;
			struct_type type;
			size_t version;
			resource* resource_ptr;
			u32 buffer_type;
			u32 lifecycle;
			streamline_dlss::extent extent;
		};

		struct mat4
		{
			float values[16];
		};

		struct constants
		{
			void* next;
			struct_type type;
			size_t version;
			mat4 camera_view_to_clip;
			mat4 clip_to_camera_view;
			mat4 clip_to_lens_clip;
			mat4 clip_to_prev_clip;
			mat4 prev_clip_to_clip;
			float jitter_offset_x;
			float jitter_offset_y;
			float mvec_scale_x;
			float mvec_scale_y;
			float camera_pinhole_offset_x;
			float camera_pinhole_offset_y;
			float camera_pos_x;
			float camera_pos_y;
			float camera_pos_z;
			float camera_up_x;
			float camera_up_y;
			float camera_up_z;
			float camera_right_x;
			float camera_right_y;
			float camera_right_z;
			float camera_fwd_x;
			float camera_fwd_y;
			float camera_fwd_z;
			float camera_near;
			float camera_far;
			float camera_fov;
			float camera_aspect_ratio;
			float motion_vectors_invalid_value;
			u8 depth_inverted;
			u8 camera_motion_included;
			u8 motion_vectors_3d;
			u8 reset;
			u8 orthographic_projection;
			u8 motion_vectors_dilated;
			u8 motion_vectors_jittered;
			float min_relative_linear_depth_object_separation;
		};

		struct dlss_options
		{
			void* next;
			struct_type type;
			size_t version;
			u32 mode;
			u32 output_width;
			u32 output_height;
			float sharpness;
			float pre_exposure;
			float exposure_scale;
			u8 color_buffers_hdr;
			u8 indicator_invert_axis_x;
			u8 indicator_invert_axis_y;
			u32 dlaa_preset;
			u32 quality_preset;
			u32 balanced_preset;
			u32 performance_preset;
			u32 ultra_performance_preset;
			u32 ultra_quality_preset;
			u8 use_auto_exposure;
			u8 alpha_upscaling_enabled;
		};

		struct dlss_optimal_settings
		{
			void* next;
			struct_type type;
			size_t version;
			u32 optimal_render_width;
			u32 optimal_render_height;
			float optimal_sharpness;
			u32 render_width_min;
			u32 render_height_min;
			u32 render_width_max;
			u32 render_height_max;
		};

		struct reflex_options
		{
			void* next;
			struct_type type;
			size_t version;
			s32 mode;
			u32 frame_limit_us;
			u8 use_markers_to_optimize;
			u16 virtual_key;
			u32 id_thread;
		};

		struct dlssg_options
		{
			void* next;
			struct_type type;
			size_t version;
			u32 mode;
			u32 num_frames_to_generate;
			u32 flags;
			u32 dynamic_res_width;
			u32 dynamic_res_height;
			u32 num_back_buffers;
			u32 mvec_depth_width;
			u32 mvec_depth_height;
			u32 color_width;
			u32 color_height;
			VkFormat color_buffer_format;
			VkFormat mvec_buffer_format;
			VkFormat depth_buffer_format;
			VkFormat hud_less_buffer_format;
			VkFormat ui_buffer_format;
			void* on_error_callback;
			u8 reserved15;
			u32 queue_parallelism_mode;
			u8 enable_user_interface_recomposition;
			float dynamic_target_frame_rate;
		};

		struct dlssg_state
		{
			void* next;
			struct_type type;
			size_t version;
			u64 estimated_vram_usage;
			u32 status;
			u32 min_width_or_height;
			u32 num_frames_actually_presented;
			u32 num_frames_to_generate_max;
			u8 reserved4;
			u8 vsync_support_available;
			void* inputs_processing_completion_fence;
			u64 last_present_inputs_processing_completion_fence_value;
			u8 dynamic_mfg_supported;
		};

		struct preferences
		{
			void* next;
			struct_type type;
			size_t version;
			u8 show_console;
			u32 log_level;
			const wchar_t** paths_to_plugins;
			u32 num_paths_to_plugins;
			const wchar_t* path_to_logs_and_data;
			void* allocate_callback;
			void* release_callback;
			void* log_message_callback;
			u64 flags;
			const u32* features_to_load;
			u32 num_features_to_load;
			u32 application_id;
			u32 engine;
			const char* engine_version;
			const char* project_id;
			u32 render_api;
		};

		struct adapter_info
		{
			void* next;
			struct_type type;
			size_t version;
			void* device_luid;
			u32 device_luid_size;
			VkPhysicalDevice vk_physical_device;
		};

		struct vulkan_info
		{
			void* next;
			struct_type type;
			size_t version;
			VkDevice device;
			VkInstance instance;
			VkPhysicalDevice physical_device;
			u32 compute_queue_index;
			u32 compute_queue_family;
			u32 graphics_queue_index;
			u32 graphics_queue_family;
			u32 optical_flow_queue_index;
			u32 optical_flow_queue_family;
			u8 use_native_optical_flow_mode;
			u32 compute_queue_create_flags;
			u32 graphics_queue_create_flags;
			u32 optical_flow_queue_create_flags;
		};

		using sl_result = int;
		using sl_init_fn = sl_result (*)(const preferences&, u64);
		using sl_shutdown_fn = sl_result (*)();
		using sl_is_feature_supported_fn = sl_result (*)(u32, const adapter_info&);
		using sl_set_vulkan_info_fn = sl_result (*)(const vulkan_info&);
		using sl_get_new_frame_token_fn = sl_result (*)(void**, const u32&);
		using sl_set_constants_fn = sl_result (*)(const constants&, void*, const viewport_handle&);
		using sl_evaluate_feature_fn = sl_result (*)(u32, void*, void* const*, u32, VkCommandBuffer);
		using sl_get_feature_function_fn = sl_result (*)(u32, const char*, void**);
		using sl_set_tag_for_frame_fn = sl_result (*)(void*, const viewport_handle&, resource_tag*, u32, VkCommandBuffer);
		using sl_vk_get_device_proc_addr_fn = PFN_vkVoidFunction (*)(VkDevice, const char*);
		using sl_dlss_set_options_fn = sl_result (*)(const viewport_handle&, const dlss_options&);
		using sl_dlss_get_optimal_settings_fn = sl_result (*)(const dlss_options&, dlss_optimal_settings&);
		using sl_reflex_set_options_fn = sl_result (*)(const reflex_options&);
		using sl_reflex_sleep_fn = sl_result (*)(void*);
		using sl_pcl_set_marker_fn = sl_result (*)(u32, void*);
		using sl_dlssg_set_options_fn = sl_result (*)(const viewport_handle&, const dlssg_options&);
		using sl_dlssg_get_state_fn = sl_result (*)(const viewport_handle&, dlssg_state&, void*);
		using sl_hook_vk_present_fn = VkResult (*)(VkQueue, const VkPresentInfoKHR*, bool&);
		using sl_hook_vk_after_present_fn = VkResult (*)();

		utils::dynamic_library m_interposer;
		bool m_initialized = false;
		bool m_dlss_supported = false;
		bool m_functions_bound = false;
		bool m_frame_generation_requested = false;
		bool m_frame_generation_supported = false;
		bool m_frame_generation_proxy_armed = false;
		bool m_frame_generation_functions_bound = false;
		bool m_reflex_enabled = false;
		bool m_frame_generation_configured = false;
		bool m_frame_generation_frame_active = false;
		u32 m_fg_viewport_id = 0;
		u32 m_fg_color_width = 0;
		u32 m_fg_color_height = 0;
		u32 m_fg_motion_width = 0;
		u32 m_fg_motion_height = 0;
		u32 m_fg_depth_width = 0;
		u32 m_fg_depth_height = 0;
		u32 m_fg_backbuffer_count = 0;
		VkFormat m_fg_color_format = VK_FORMAT_UNDEFINED;
		VkFormat m_fg_motion_format = VK_FORMAT_UNDEFINED;
		VkFormat m_fg_depth_format = VK_FORMAT_UNDEFINED;
		u32 m_fg_frames_to_generate = 1;
		u32 m_fg_seen_color_width = 0;
		u32 m_fg_seen_color_height = 0;
		u32 m_fg_seen_motion_width = 0;
		u32 m_fg_seen_motion_height = 0;
		u32 m_fg_seen_depth_width = 0;
		u32 m_fg_seen_depth_height = 0;
		u32 m_fg_seen_backbuffer_count = 0;
		VkFormat m_fg_seen_color_format = VK_FORMAT_UNDEFINED;
		VkFormat m_fg_seen_motion_format = VK_FORMAT_UNDEFINED;
		VkFormat m_fg_seen_depth_format = VK_FORMAT_UNDEFINED;
		u32 m_fg_size_stable_frames = 0;
		dlssg_options m_last_fg_options{};
		bool m_fg_state_failed_logged = false;
		bool m_evaluate_success_logged = false;
		bool m_evaluate_failure_logged = false;
		void* m_last_frame_token = nullptr;
		bool m_options_set = false;
		u32 m_viewport_id = 0;
		mode m_mode = mode::balanced;
		u32 m_output_width = 0;
		u32 m_output_height = 0;
		bool m_hdr = false;
		u32 m_preset = 0;
		std::wstring m_log_path;
		std::vector<u32> m_features;

		sl_init_fn m_sl_init = nullptr;
		sl_shutdown_fn m_sl_shutdown = nullptr;
		sl_is_feature_supported_fn m_sl_is_feature_supported = nullptr;
		sl_set_vulkan_info_fn m_sl_set_vulkan_info = nullptr;
		sl_get_new_frame_token_fn m_sl_get_new_frame_token = nullptr;
		sl_set_constants_fn m_sl_set_constants = nullptr;
		sl_evaluate_feature_fn m_sl_evaluate_feature = nullptr;
		sl_get_feature_function_fn m_sl_get_feature_function = nullptr;
		sl_set_tag_for_frame_fn m_sl_set_tag_for_frame = nullptr;
		sl_vk_get_device_proc_addr_fn m_sl_vk_get_device_proc_addr = nullptr;
		sl_dlss_set_options_fn m_sl_dlss_set_options = nullptr;
		sl_dlss_get_optimal_settings_fn m_sl_dlss_get_optimal_settings = nullptr;
		sl_reflex_set_options_fn m_sl_reflex_set_options = nullptr;
		sl_reflex_sleep_fn m_sl_reflex_sleep = nullptr;
		sl_pcl_set_marker_fn m_sl_pcl_set_marker = nullptr;
		sl_dlssg_set_options_fn m_sl_dlssg_set_options = nullptr;
		sl_dlssg_get_state_fn m_sl_dlssg_get_state = nullptr;
		sl_hook_vk_present_fn m_sl_hook_vk_present = nullptr;
		sl_hook_vk_after_present_fn m_sl_hook_vk_after_present = nullptr;
		bool m_present_hook_failure_logged = false;

		static struct_type make_guid(u32 data1, u16 data2, u16 data3, std::initializer_list<u8> bytes);
		static mat4 identity_matrix();
		static resource make_resource(const texture& texture);
		static resource_tag make_tag(resource* resource, u32 buffer_type, u32 width, u32 height, u32 lifecycle = 2);
		static viewport_handle make_viewport(u32 viewport_id);
		static constants make_constants(u32 output_width, u32 output_height, u32 render_width, u32 render_height, bool reset, float jitter_x, float jitter_y);
		bool bind_feature_functions();
		bool bind_frame_generation_functions();
		bool load_interposer();

		static_assert(sizeof(struct_type) == 16);
		static_assert(sizeof(viewport_handle) == 40);
		static_assert(sizeof(resource) == 112);
		static_assert(sizeof(resource_tag) == 64);
		static_assert(sizeof(constants) == 456);
		static_assert(sizeof(dlss_options) == 88);
		static_assert(sizeof(dlss_optimal_settings) == 64);
		static_assert(sizeof(reflex_options) == 48);
		static_assert(sizeof(dlssg_options) == 120);
		static_assert(sizeof(dlssg_state) == 88);
		static_assert(sizeof(preferences) == 144);
		static_assert(sizeof(adapter_info) == 56);
		static_assert(sizeof(vulkan_info) == 96);
	};

	streamline_dlss& get_streamline_dlss();
}
