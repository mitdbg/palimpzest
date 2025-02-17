<svelte:options accessors={true} />

<script context="module">export { default as BaseMultimodalTextbox } from "./shared/MultimodalTextbox.svelte";
export { default as BaseExample } from "./Example.svelte";
</script>

<script>import MultimodalTextbox from "./shared/MultimodalTextbox.svelte";
import { Block } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
import { onMount } from "svelte";
export let gradio;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = {
  text: "",
  files: []
};
export let file_types = null;
export let lines;
export let placeholder = "";
export let label = "MultimodalTextbox";
export let info = void 0;
export let show_label;
export let max_lines;
export let scale = null;
export let min_width = void 0;
export let submit_btn = null;
export let stop_btn = null;
export let loading_status = void 0;
export let value_is_output = false;
export let rtl = false;
export let text_align = void 0;
export let autofocus = false;
export let autoscroll = true;
export let interactive;
export let root;
export let file_count;
export let max_plain_text_length;
export let sources = ["upload"];
export let waveform_options = {};
let dragging;
let active_source = null;
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
    autoplay: false,
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
</script>

<Block
	{visible}
	{elem_id}
	elem_classes={[...elem_classes, "multimodal-textbox"]}
	{scale}
	{min_width}
	allow_overflow={false}
	padding={false}
	border_mode={dragging ? "focus" : "base"}
>
	{#if loading_status}
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
	{/if}

	<MultimodalTextbox
		bind:value
		bind:value_is_output
		bind:dragging
		bind:active_source
		{file_types}
		{root}
		{label}
		{info}
		{show_label}
		{lines}
		{rtl}
		{text_align}
		{waveform_settings}
		i18n={gradio.i18n}
		max_lines={!max_lines ? lines + 1 : max_lines}
		{placeholder}
		{submit_btn}
		{stop_btn}
		{autofocus}
		{autoscroll}
		{file_count}
		{sources}
		max_file_size={gradio.max_file_size}
		on:change={() => gradio.dispatch("change", value)}
		on:input={() => gradio.dispatch("input")}
		on:submit={() => gradio.dispatch("submit")}
		on:stop={() => gradio.dispatch("stop")}
		on:blur={() => gradio.dispatch("blur")}
		on:select={(e) => gradio.dispatch("select", e.detail)}
		on:focus={() => gradio.dispatch("focus")}
		on:error={({ detail }) => {
			gradio.dispatch("error", detail);
		}}
		on:start_recording={() => gradio.dispatch("start_recording")}
		on:pause_recording={() => gradio.dispatch("pause_recording")}
		on:stop_recording={() => gradio.dispatch("stop_recording")}
		on:upload={(e) => gradio.dispatch("upload", e.detail)}
		on:clear={() => gradio.dispatch("clear")}
		disabled={!interactive}
		upload={(...args) => gradio.client.upload(...args)}
		stream_handler={(...args) => gradio.client.stream(...args)}
		{max_plain_text_length}
	/>
</Block>
