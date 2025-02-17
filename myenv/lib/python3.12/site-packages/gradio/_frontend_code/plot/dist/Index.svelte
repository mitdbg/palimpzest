<script context="module">export { default as BasePlot } from "./shared/Plot.svelte";
</script>

<script>import Plot from "./shared/Plot.svelte";
import { Block, BlockLabel } from "@gradio/atoms";
import { Plot as PlotIcon } from "@gradio/icons";
import { StatusTracker } from "@gradio/statustracker";
export let value = null;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let loading_status;
export let label;
export let show_label;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let theme_mode;
export let caption;
export let bokeh_version;
export let gradio;
export let show_actions_button = false;
export let _selectable = false;
export let x_lim = null;
</script>

<Block
	padding={false}
	{elem_id}
	{elem_classes}
	{visible}
	{container}
	{scale}
	{min_width}
	allow_overflow={false}
>
	<BlockLabel
		{show_label}
		label={label || gradio.i18n("plot.plot")}
		Icon={PlotIcon}
	/>
	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
		on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
	/>
	<Plot
		{value}
		{theme_mode}
		{caption}
		{bokeh_version}
		{show_actions_button}
		{gradio}
		{show_label}
		{_selectable}
		{x_lim}
		on:change={() => gradio.dispatch("change")}
		on:select={(e) => gradio.dispatch("select", e.detail)}
	/>
</Block>
