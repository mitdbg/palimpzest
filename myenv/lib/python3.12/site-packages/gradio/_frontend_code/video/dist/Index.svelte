<svelte:options accessors={true} />

<script>import { Block, UploadText } from "@gradio/atoms";
import StaticVideo from "./shared/VideoPreview.svelte";
import Video from "./shared/InteractiveVideo.svelte";
import { StatusTracker } from "@gradio/statustracker";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = null;
let old_value = null;
export let label;
export let sources;
export let root;
export let show_label;
export let loading_status;
export let height;
export let width;
export let webcam_constraints = null;
export let container = false;
export let scale = null;
export let min_width = void 0;
export let autoplay = false;
export let show_share_button = true;
export let show_download_button;
export let gradio;
export let interactive;
export let mirror_webcam;
export let include_audio;
export let loop = false;
export let input_ready;
let uploading = false;
$:
  input_ready = !uploading;
let _video = null;
let _subtitle = null;
let active_source;
let initial_value = value;
$:
  if (value && initial_value === null) {
    initial_value = value;
  }
const handle_reset_value = () => {
  if (initial_value === null || value === initial_value) {
    return;
  }
  value = initial_value;
};
$:
  if (sources && !active_source) {
    active_source = sources[0];
  }
$: {
  if (value != null) {
    _video = value.video;
    _subtitle = value.subtitles;
  } else {
    _video = null;
    _subtitle = null;
  }
}
let dragging = false;
$: {
  if (JSON.stringify(value) !== JSON.stringify(old_value)) {
    old_value = value;
    gradio.dispatch("change");
  }
}
function handle_change({ detail }) {
  if (detail != null) {
    value = { video: detail, subtitles: null };
  } else {
    value = null;
  }
}
function handle_error({ detail }) {
  const [level, status] = detail.includes("Invalid file type") ? ["warning", "complete"] : ["error", "error"];
  loading_status = loading_status || {};
  loading_status.status = status;
  loading_status.message = detail;
  gradio.dispatch(level, detail);
}
</script>

{#if !interactive}
	<Block
		{visible}
		variant={value === null && active_source === "upload" ? "dashed" : "solid"}
		border_mode={dragging ? "focus" : "base"}
		padding={false}
		{elem_id}
		{elem_classes}
		{height}
		{width}
		{container}
		{scale}
		{min_width}
		allow_overflow={false}
	>
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>

		<StaticVideo
			value={_video}
			subtitle={_subtitle}
			{label}
			{show_label}
			{autoplay}
			{loop}
			{show_share_button}
			{show_download_button}
			on:play={() => gradio.dispatch("play")}
			on:pause={() => gradio.dispatch("pause")}
			on:stop={() => gradio.dispatch("stop")}
			on:end={() => gradio.dispatch("end")}
			on:share={({ detail }) => gradio.dispatch("share", detail)}
			on:error={({ detail }) => gradio.dispatch("error", detail)}
			i18n={gradio.i18n}
			upload={(...args) => gradio.client.upload(...args)}
		/>
	</Block>
{:else}
	<Block
		{visible}
		variant={value === null && active_source === "upload" ? "dashed" : "solid"}
		border_mode={dragging ? "focus" : "base"}
		padding={false}
		{elem_id}
		{elem_classes}
		{height}
		{width}
		{container}
		{scale}
		{min_width}
		allow_overflow={false}
	>
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>

		<Video
			value={_video}
			subtitle={_subtitle}
			on:change={handle_change}
			on:drag={({ detail }) => (dragging = detail)}
			on:error={handle_error}
			bind:uploading
			{label}
			{show_label}
			{show_download_button}
			{sources}
			{active_source}
			{mirror_webcam}
			{include_audio}
			{autoplay}
			{root}
			{loop}
			{webcam_constraints}
			{handle_reset_value}
			on:clear={() => gradio.dispatch("clear")}
			on:play={() => gradio.dispatch("play")}
			on:pause={() => gradio.dispatch("pause")}
			on:upload={() => gradio.dispatch("upload")}
			on:stop={() => gradio.dispatch("stop")}
			on:end={() => gradio.dispatch("end")}
			on:start_recording={() => gradio.dispatch("start_recording")}
			on:stop_recording={() => gradio.dispatch("stop_recording")}
			i18n={gradio.i18n}
			max_file_size={gradio.max_file_size}
			upload={(...args) => gradio.client.upload(...args)}
			stream_handler={(...args) => gradio.client.stream(...args)}
		>
			<UploadText i18n={gradio.i18n} type="video" />
		</Video>
	</Block>
{/if}
