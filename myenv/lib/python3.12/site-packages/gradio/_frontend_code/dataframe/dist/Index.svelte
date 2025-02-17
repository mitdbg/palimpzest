<svelte:options accessors={true} />

<script context="module">export { default as BaseDataFrame } from "./shared/Table.svelte";
export { default as BaseExample } from "./Example.svelte";
</script>

<script>import { Block } from "@gradio/atoms";
import Table from "./shared/Table.svelte";
import { StatusTracker } from "@gradio/statustracker";
export let headers = [];
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = {
  data: [["", "", ""]],
  headers: ["1", "2", "3"],
  metadata: null
};
export let value_is_output = false;
export let col_count;
export let row_count;
export let label = null;
export let show_label = true;
export let wrap;
export let datatype;
export let scale = null;
export let min_width = void 0;
export let root;
export let line_breaks = true;
export let column_widths = [];
export let gradio;
export let latex_delimiters;
export let max_height = void 0;
export let loading_status;
export let interactive;
export let show_fullscreen_button = false;
export let show_copy_button = false;
$:
  _headers = [...value.headers || headers];
$:
  cell_values = value.data ? [...value.data] : [];
$:
  display_value = value?.metadata?.display_value ? [...value?.metadata?.display_value] : null;
$:
  styling = !interactive && value?.metadata?.styling ? [...value?.metadata?.styling] : null;
</script>

<Block
	{visible}
	padding={false}
	{elem_id}
	{elem_classes}
	container={false}
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
	<Table
		{root}
		{label}
		{show_label}
		{row_count}
		{col_count}
		values={cell_values}
		{display_value}
		{styling}
		headers={_headers}
		on:change={(e) => {
			value = e.detail;
			gradio.dispatch("change");
		}}
		on:input={(e) => gradio.dispatch("input")}
		on:select={(e) => gradio.dispatch("select", e.detail)}
		{wrap}
		{datatype}
		{latex_delimiters}
		editable={interactive}
		{max_height}
		i18n={gradio.i18n}
		{line_breaks}
		{column_widths}
		upload={(...args) => gradio.client.upload(...args)}
		stream_handler={(...args) => gradio.client.stream(...args)}
		bind:value_is_output
		{show_fullscreen_button}
		{show_copy_button}
	/>
</Block>
