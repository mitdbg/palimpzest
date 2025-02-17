<script context="module">export { default as BaseRadio } from "./shared/Radio.svelte";
export { default as BaseExample } from "./Example.svelte";
</script>

<script>import { afterUpdate } from "svelte";
import { Block, BlockTitle } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
import BaseRadio from "./shared/Radio.svelte";
export let gradio;
export let label = gradio.i18n("radio.radio");
export let info = void 0;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = null;
export let choices = [];
export let show_label = true;
export let container = false;
export let scale = null;
export let min_width = void 0;
export let loading_status;
export let interactive = true;
export let root;
function handle_change() {
  gradio.dispatch("change");
}
let old_value = value;
$: {
  if (value !== old_value) {
    old_value = value;
    handle_change();
  }
}
$:
  disabled = !interactive;
</script>

<Block
	{visible}
	type="fieldset"
	{elem_id}
	{elem_classes}
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

	<BlockTitle {root} {show_label} {info}>{label}</BlockTitle>

	<div class="wrap">
		{#each choices as [display_value, internal_value], i (i)}
			<BaseRadio
				{display_value}
				{internal_value}
				bind:selected={value}
				{disabled}
				on:input={() => {
					gradio.dispatch("select", { value: internal_value, index: i });
					gradio.dispatch("input");
				}}
			/>
		{/each}
	</div>
</Block>

<style>
	.wrap {
		display: flex;
		flex-wrap: wrap;
		gap: var(--checkbox-label-gap);
	}
</style>
