<script context="module">export { default as BaseLabel } from "./shared/Label.svelte";
</script>

<script>import Label from "./shared/Label.svelte";
import { LineChart as LabelIcon } from "@gradio/icons";
import { Block, BlockLabel, Empty } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
export let gradio;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let color = void 0;
export let value = {};
let old_value = null;
export let label = gradio.i18n("label.label");
export let container = true;
export let scale = null;
export let min_width = void 0;
export let loading_status;
export let show_label = true;
export let _selectable = false;
export let show_heading = true;
$: {
  if (JSON.stringify(value) !== JSON.stringify(old_value)) {
    old_value = value;
    gradio.dispatch("change");
  }
}
$:
  _label = value.label;
</script>

<Block
	test_id="label"
	{visible}
	{elem_id}
	{elem_classes}
	{container}
	{scale}
	{min_width}
	padding={false}
>
	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
		on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
	/>
	{#if show_label}
		<BlockLabel
			Icon={LabelIcon}
			{label}
			disable={container === false}
			float={show_heading === true}
		/>
	{/if}
	{#if _label !== undefined && _label !== null}
		<Label
			on:select={({ detail }) => gradio.dispatch("select", detail)}
			selectable={_selectable}
			{value}
			{color}
			{show_heading}
		/>
	{:else}
		<Empty unpadded_box={true}><LabelIcon /></Empty>
	{/if}
</Block>
