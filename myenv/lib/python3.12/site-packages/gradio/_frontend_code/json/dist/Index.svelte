<script context="module">export { default as BaseJSON } from "./shared/JSON.svelte";
</script>

<script>import JSON from "./shared/JSON.svelte";
import { Block, BlockLabel } from "@gradio/atoms";
import { JSON as JSONIcon } from "@gradio/icons";
import { StatusTracker } from "@gradio/statustracker";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value;
let old_value;
export let loading_status;
export let label;
export let show_label;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let gradio;
export let open = false;
export let theme_mode;
export let show_indices;
export let height;
export let min_height;
export let max_height;
$: {
  if (value !== old_value) {
    old_value = value;
    gradio.dispatch("change");
  }
}
let label_height = 0;
</script>

<Block
	{visible}
	test_id="json"
	{elem_id}
	{elem_classes}
	{container}
	{scale}
	{min_width}
	padding={false}
	allow_overflow={true}
	overflow_behavior="auto"
	{height}
	{min_height}
	{max_height}
>
	<div bind:clientHeight={label_height}>
		{#if label}
			<BlockLabel
				Icon={JSONIcon}
				{show_label}
				{label}
				float={false}
				disable={container === false}
			/>
		{/if}
	</div>

	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
		on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
	/>

	<JSON {value} {open} {theme_mode} {show_indices} {label_height} />
</Block>
