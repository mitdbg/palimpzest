<script context="module">export { default as BaseCheckbox } from "./shared/Checkbox.svelte";
</script>

<script>import { Block, Info } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
import { afterUpdate } from "svelte";
import BaseCheckbox from "./shared/Checkbox.svelte";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = false;
export let value_is_output = false;
export let label = "Checkbox";
export let info = void 0;
export let root;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let loading_status;
export let gradio;
export let interactive;
function handle_change() {
  gradio.dispatch("change");
  if (!value_is_output) {
    gradio.dispatch("input");
  }
}
afterUpdate(() => {
  value_is_output = false;
});
</script>

<Block {visible} {elem_id} {elem_classes} {container} {scale} {min_width}>
	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
		on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
	/>

	{#if info}
		<Info {root} {info} />
	{/if}

	<BaseCheckbox
		bind:value
		{label}
		{interactive}
		on:change={handle_change}
		on:select={(e) => gradio.dispatch("select", e.detail)}
	/>
</Block>
