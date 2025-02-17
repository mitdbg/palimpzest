<svelte:options accessors={true} />

<script context="module">export { default as BaseImageUploader } from "./shared/ImageUploader.svelte";
export { default as BaseStaticImage } from "./shared/ImagePreview.svelte";
export { default as BaseExample } from "./Example.svelte";
</script>

<script>import ImagePreview from "./shared/ImagePreview.svelte";
import ImageUploader from "./shared/ImageUploader.svelte";
import { Block, UploadText } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = null;
export let label;
export let show_label;
export let show_download_button;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let loading_status;
export let interactive;
export let root;
export let placeholder = void 0;
export let gradio;
$:
  value, gradio.dispatch("change");
let dragging;
</script>

{#if !interactive}
	<Block
		{visible}
		variant={"solid"}
		border_mode={dragging ? "focus" : "base"}
		padding={false}
		{elem_id}
		{elem_classes}
		allow_overflow={false}
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
		<ImagePreview
			{value}
			{label}
			{show_label}
			{show_download_button}
			i18n={gradio.i18n}
		/>
	</Block>
{:else}
	<Block
		{visible}
		variant={value === null ? "dashed" : "solid"}
		border_mode={dragging ? "focus" : "base"}
		padding={false}
		{elem_id}
		{elem_classes}
		allow_overflow={false}
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

		<ImageUploader
			upload={(...args) => gradio.client.upload(...args)}
			stream_handler={(...args) => gradio.client.stream(...args)}
			bind:value
			{root}
			on:clear={() => gradio.dispatch("clear")}
			on:drag={({ detail }) => (dragging = detail)}
			on:upload={() => gradio.dispatch("upload")}
			{label}
			{show_label}
		>
			<UploadText i18n={gradio.i18n} type="image" {placeholder} />
		</ImageUploader>
	</Block>
{/if}
