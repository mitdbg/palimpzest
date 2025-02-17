<svelte:options accessors={true} />

<script context="module">export { default as FilePreview } from "./shared/FilePreview.svelte";
export { default as BaseFileUpload } from "./shared/FileUpload.svelte";
export { default as BaseFile } from "./shared/File.svelte";
export { default as BaseExample } from "./Example.svelte";
</script>

<script>import File from "./shared/File.svelte";
import FileUpload from "./shared/FileUpload.svelte";
import { Block, UploadText } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value;
export let interactive;
export let root;
export let label;
export let show_label;
export let height = void 0;
export let _selectable = false;
export let loading_status;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let gradio;
export let file_count;
export let file_types = ["file"];
export let input_ready;
export let allow_reordering = false;
let uploading = false;
$:
  input_ready = !uploading;
let old_value = value;
$:
  if (JSON.stringify(old_value) !== JSON.stringify(value)) {
    gradio.dispatch("change");
    old_value = value;
  }
let dragging = false;
let pending_upload = false;
</script>

<Block
	{visible}
	variant={value ? "solid" : "dashed"}
	border_mode={dragging ? "focus" : "base"}
	padding={false}
	{elem_id}
	{elem_classes}
	{container}
	{scale}
	{min_width}
	allow_overflow={false}
>
	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
		status={pending_upload
			? "generating"
			: loading_status?.status || "complete"}
		on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
	/>
	{#if !interactive}
		<File
			on:select={({ detail }) => gradio.dispatch("select", detail)}
			on:download={({ detail }) => gradio.dispatch("download", detail)}
			selectable={_selectable}
			{value}
			{label}
			{show_label}
			{height}
			i18n={gradio.i18n}
		/>
	{:else}
		<FileUpload
			upload={(...args) => gradio.client.upload(...args)}
			stream_handler={(...args) => gradio.client.stream(...args)}
			{label}
			{show_label}
			{value}
			{file_count}
			{file_types}
			selectable={_selectable}
			{root}
			{height}
			{allow_reordering}
			bind:uploading
			max_file_size={gradio.max_file_size}
			on:change={({ detail }) => {
				value = detail;
			}}
			on:drag={({ detail }) => (dragging = detail)}
			on:clear={() => gradio.dispatch("clear")}
			on:select={({ detail }) => gradio.dispatch("select", detail)}
			on:upload={() => gradio.dispatch("upload")}
			on:error={({ detail }) => {
				loading_status = loading_status || {};
				loading_status.status = "error";
				gradio.dispatch("error", detail);
			}}
			on:delete={({ detail }) => {
				gradio.dispatch("delete", detail);
			}}
			i18n={gradio.i18n}
		>
			<UploadText i18n={gradio.i18n} type="file" />
		</FileUpload>
	{/if}
</Block>
