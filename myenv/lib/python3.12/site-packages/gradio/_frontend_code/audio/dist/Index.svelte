<svelte:options accessors={true} />

<script>import { afterUpdate, onMount } from "svelte";
import StaticAudio from "./static/StaticAudio.svelte";
import InteractiveAudio from "./interactive/InteractiveAudio.svelte";
import { StatusTracker } from "@gradio/statustracker";
import { Block, UploadText } from "@gradio/atoms";
export let value_is_output = false;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let interactive;
export let value = null;
export let sources;
export let label;
export let root;
export let show_label;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let loading_status;
export let autoplay = false;
export let loop = false;
export let show_download_button;
export let show_share_button = false;
export let editable = true;
export let waveform_options = {
  show_recording_waveform: true
};
export let pending;
export let streaming;
export let stream_every;
export let input_ready;
export let recording = false;
let uploading = false;
$:
  input_ready = !uploading;
let stream_state = "closed";
let _modify_stream;
export function modify_stream_state(state) {
  stream_state = state;
  _modify_stream(state);
}
export const get_stream_state = () => stream_state;
export let set_time_limit;
export let gradio;
let old_value = null;
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
$: {
  if (JSON.stringify(value) !== JSON.stringify(old_value)) {
    old_value = value;
    gradio.dispatch("change");
    if (!value_is_output) {
      gradio.dispatch("input");
    }
  }
}
let dragging;
$:
  if (!active_source && sources) {
    active_source = sources[0];
  }
let waveform_settings;
let color_accent = "darkorange";
onMount(() => {
  color_accent = getComputedStyle(document?.documentElement).getPropertyValue(
    "--color-accent"
  );
  set_trim_region_colour();
  waveform_settings.waveColor = waveform_options.waveform_color || "#9ca3af";
  waveform_settings.progressColor = waveform_options.waveform_progress_color || color_accent;
  waveform_settings.mediaControls = waveform_options.show_controls;
  waveform_settings.sampleRate = waveform_options.sample_rate || 44100;
});
$:
  waveform_settings = {
    height: 50,
    barWidth: 2,
    barGap: 3,
    cursorWidth: 2,
    cursorColor: "#ddd5e9",
    autoplay,
    barRadius: 10,
    dragToSeek: true,
    normalize: true,
    minPxPerSec: 20
  };
const trim_region_settings = {
  color: waveform_options.trim_region_color,
  drag: true,
  resize: true
};
function set_trim_region_colour() {
  document.documentElement.style.setProperty(
    "--trim-region-color",
    trim_region_settings.color || color_accent
  );
}
function handle_error({ detail }) {
  const [level, status] = detail.includes("Invalid file type") ? ["warning", "complete"] : ["error", "error"];
  loading_status = loading_status || {};
  loading_status.status = status;
  loading_status.message = detail;
  gradio.dispatch(level, detail);
}
afterUpdate(() => {
  value_is_output = false;
});
</script>

{#if !interactive}
	<Block
		variant={"solid"}
		border_mode={dragging ? "focus" : "base"}
		padding={false}
		allow_overflow={false}
		{elem_id}
		{elem_classes}
		{visible}
		{container}
		{scale}
		{min_width}
	>
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>

		<StaticAudio
			i18n={gradio.i18n}
			{show_label}
			{show_download_button}
			{show_share_button}
			{value}
			{label}
			{loop}
			{waveform_settings}
			{waveform_options}
			{editable}
			on:share={(e) => gradio.dispatch("share", e.detail)}
			on:error={(e) => gradio.dispatch("error", e.detail)}
			on:play={() => gradio.dispatch("play")}
			on:pause={() => gradio.dispatch("pause")}
			on:stop={() => gradio.dispatch("stop")}
		/>
	</Block>
{:else}
	<Block
		variant={value === null && active_source === "upload" ? "dashed" : "solid"}
		border_mode={dragging ? "focus" : "base"}
		padding={false}
		allow_overflow={false}
		{elem_id}
		{elem_classes}
		{visible}
		{container}
		{scale}
		{min_width}
	>
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
		<InteractiveAudio
			{label}
			{show_label}
			{show_download_button}
			{value}
			on:change={({ detail }) => (value = detail)}
			on:stream={({ detail }) => {
				value = detail;
				gradio.dispatch("stream", value);
			}}
			on:drag={({ detail }) => (dragging = detail)}
			{root}
			{sources}
			{active_source}
			{pending}
			{streaming}
			bind:recording
			{loop}
			max_file_size={gradio.max_file_size}
			{handle_reset_value}
			{editable}
			bind:dragging
			bind:uploading
			on:edit={() => gradio.dispatch("edit")}
			on:play={() => gradio.dispatch("play")}
			on:pause={() => gradio.dispatch("pause")}
			on:stop={() => gradio.dispatch("stop")}
			on:start_recording={() => gradio.dispatch("start_recording")}
			on:pause_recording={() => gradio.dispatch("pause_recording")}
			on:stop_recording={(e) => gradio.dispatch("stop_recording")}
			on:upload={() => gradio.dispatch("upload")}
			on:clear={() => gradio.dispatch("clear")}
			on:error={handle_error}
			on:close_stream={() => gradio.dispatch("close_stream", "stream")}
			i18n={gradio.i18n}
			{waveform_settings}
			{waveform_options}
			{trim_region_settings}
			{stream_every}
			bind:modify_stream={_modify_stream}
			bind:set_time_limit
			upload={(...args) => gradio.client.upload(...args)}
			stream_handler={(...args) => gradio.client.stream(...args)}
		>
			<UploadText i18n={gradio.i18n} type="audio" />
		</InteractiveAudio>
	</Block>
{/if}
