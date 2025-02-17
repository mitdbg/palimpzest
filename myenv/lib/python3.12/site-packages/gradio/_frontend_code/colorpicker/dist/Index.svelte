<svelte:options accessors={true} />

<script context="module">export { default as BaseColorPicker } from "./shared/Colorpicker.svelte";
export { default as BaseExample } from "./Example.svelte";
</script>

<script>import Colorpicker from "./shared/Colorpicker.svelte";
import { Block } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
export let label = "ColorPicker";
export let info = void 0;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value;
export let value_is_output = false;
export let show_label;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let loading_status;
export let root;
export let gradio;
export let interactive;
export let disabled = false;
</script>

<Block {visible} {elem_id} {elem_classes} {container} {scale} {min_width}>
	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
		on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
	/>

	<Colorpicker
		bind:value
		bind:value_is_output
		{root}
		{label}
		{info}
		{show_label}
		disabled={!interactive || disabled}
		on:change={() => gradio.dispatch("change")}
		on:input={() => gradio.dispatch("input")}
		on:submit={() => gradio.dispatch("submit")}
		on:blur={() => gradio.dispatch("blur")}
		on:focus={() => gradio.dispatch("focus")}
	/>
</Block>
