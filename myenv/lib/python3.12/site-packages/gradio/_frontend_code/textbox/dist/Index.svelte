<svelte:options accessors={true} />

<script context="module">export { default as BaseTextbox } from "./shared/Textbox.svelte";
export { default as BaseExample } from "./Example.svelte";
</script>

<script>import TextBox from "./shared/Textbox.svelte";
import { Block } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
export let gradio;
export let label = "Textbox";
export let info = void 0;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = "";
export let lines;
export let placeholder = "";
export let show_label;
export let max_lines;
export let type = "text";
export let container = true;
export let scale = null;
export let min_width = void 0;
export let submit_btn = null;
export let stop_btn = null;
export let show_copy_button = false;
export let loading_status = void 0;
export let value_is_output = false;
export let rtl = false;
export let text_align = void 0;
export let autofocus = false;
export let autoscroll = true;
export let interactive;
export let root;
export let max_length = void 0;
</script>

<Block
	{visible}
	{elem_id}
	{elem_classes}
	{scale}
	{min_width}
	allow_overflow={false}
	padding={container}
>
	{#if loading_status}
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
	{/if}

	<TextBox
		bind:value
		bind:value_is_output
		{label}
		{info}
		{root}
		{show_label}
		{lines}
		{type}
		{rtl}
		{text_align}
		max_lines={!max_lines ? lines + 1 : max_lines}
		{placeholder}
		{submit_btn}
		{stop_btn}
		{show_copy_button}
		{autofocus}
		{container}
		{autoscroll}
		{max_length}
		on:change={() => gradio.dispatch("change", value)}
		on:input={() => gradio.dispatch("input")}
		on:submit={() => gradio.dispatch("submit")}
		on:blur={() => gradio.dispatch("blur")}
		on:select={(e) => gradio.dispatch("select", e.detail)}
		on:focus={() => gradio.dispatch("focus")}
		on:stop={() => gradio.dispatch("stop")}
		on:copy={(e) => gradio.dispatch("copy", e.detail)}
		disabled={!interactive}
	/>
</Block>
