<script>import { createEventDispatcher } from "svelte";
import { Upload, ModifyUpload } from "@gradio/upload";
import { BlockLabel } from "@gradio/atoms";
import { Webcam } from "@gradio/image";
import { Video } from "@gradio/icons";
import { prettyBytes, playable } from "./utils";
import Player from "./Player.svelte";
import { SelectSource } from "@gradio/atoms";
export let value = null;
export let subtitle = null;
export let sources = ["webcam", "upload"];
export let label = void 0;
export let show_download_button = false;
export let show_label = true;
export let mirror_webcam = false;
export let include_audio;
export let autoplay;
export let root;
export let i18n;
export let active_source = "webcam";
export let handle_reset_value = () => {
};
export let max_file_size = null;
export let upload;
export let stream_handler;
export let loop;
export let uploading = false;
export let webcam_constraints = null;
let has_change_history = false;
const dispatch = createEventDispatcher();
function handle_load({ detail }) {
  value = detail;
  dispatch("change", detail);
  dispatch("upload", detail);
}
function handle_clear() {
  value = null;
  dispatch("change", null);
  dispatch("clear");
}
function handle_change(video) {
  has_change_history = true;
  dispatch("change", video);
}
function handle_capture({
  detail
}) {
  dispatch("change", detail);
}
let dragging = false;
$:
  dispatch("drag", dragging);
</script>

<BlockLabel {show_label} Icon={Video} label={label || "Video"} />
<div data-testid="video" class="video-container">
	{#if value === null || value.url === undefined}
		<div class="upload-container">
			{#if active_source === "upload"}
				<Upload
					bind:dragging
					bind:uploading
					filetype="video/x-m4v,video/*"
					on:load={handle_load}
					{max_file_size}
					on:error={({ detail }) => dispatch("error", detail)}
					{root}
					{upload}
					{stream_handler}
				>
					<slot />
				</Upload>
			{:else if active_source === "webcam"}
				<Webcam
					{root}
					{mirror_webcam}
					{include_audio}
					{webcam_constraints}
					mode="video"
					on:error
					on:capture={handle_capture}
					on:start_recording
					on:stop_recording
					{i18n}
					{upload}
					stream_every={1}
				/>
			{/if}
		</div>
	{:else if playable()}
		{#key value?.url}
			<Player
				{upload}
				{root}
				interactive
				{autoplay}
				src={value.url}
				subtitle={subtitle?.url}
				is_stream={false}
				on:play
				on:pause
				on:stop
				on:end
				mirror={mirror_webcam && active_source === "webcam"}
				{label}
				{handle_change}
				{handle_reset_value}
				{loop}
				{value}
				{i18n}
				{show_download_button}
				{handle_clear}
				{has_change_history}
			/>
		{/key}
	{:else if value.size}
		<div class="file-name">{value.orig_name || value.url}</div>
		<div class="file-size">
			{prettyBytes(value.size)}
		</div>
	{/if}

	<SelectSource {sources} bind:active_source {handle_clear} />
</div>

<style>
	.file-name {
		padding: var(--size-6);
		font-size: var(--text-xxl);
		word-break: break-all;
	}

	.file-size {
		padding: var(--size-2);
		font-size: var(--text-xl);
	}

	.upload-container {
		height: 100%;
		width: 100%;
	}

	.video-container {
		display: flex;
		height: 100%;
		flex-direction: column;
		justify-content: center;
		align-items: center;
	}
</style>
