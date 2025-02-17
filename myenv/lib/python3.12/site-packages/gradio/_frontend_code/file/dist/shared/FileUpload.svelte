<script>import { createEventDispatcher, tick } from "svelte";
import { Upload, ModifyUpload } from "@gradio/upload";
import { BlockLabel, IconButtonWrapper, IconButton } from "@gradio/atoms";
import { File, Clear, Upload as UploadIcon } from "@gradio/icons";
import FilePreview from "./FilePreview.svelte";
export let value;
export let label;
export let show_label = true;
export let file_count = "single";
export let file_types = null;
export let selectable = false;
export let root;
export let height = void 0;
export let i18n;
export let max_file_size = null;
export let upload;
export let stream_handler;
export let uploading = false;
export let allow_reordering = false;
async function handle_upload({
  detail
}) {
  if (Array.isArray(value)) {
    value = [...value, ...Array.isArray(detail) ? detail : [detail]];
  } else if (value) {
    value = [value, ...Array.isArray(detail) ? detail : [detail]];
  } else {
    value = detail;
  }
  await tick();
  dispatch("change", value);
  dispatch("upload", detail);
}
function handle_clear() {
  value = null;
  dispatch("change", null);
  dispatch("clear");
}
const dispatch = createEventDispatcher();
let dragging = false;
$:
  dispatch("drag", dragging);
</script>

<BlockLabel {show_label} Icon={File} float={!value} label={label || "File"} />

{#if value && (Array.isArray(value) ? value.length > 0 : true)}
	<IconButtonWrapper>
		{#if !(file_count === "single" && (Array.isArray(value) ? value.length > 0 : value !== null))}
			<IconButton Icon={UploadIcon} label={i18n("common.upload")}>
				<Upload
					icon_upload={true}
					on:load={handle_upload}
					filetype={file_types}
					{file_count}
					{max_file_size}
					{root}
					bind:dragging
					bind:uploading
					on:error
					{stream_handler}
					{upload}
				/>
			</IconButton>
		{/if}
		<IconButton
			Icon={Clear}
			label={i18n("common.clear")}
			on:click={(event) => {
				dispatch("clear");
				event.stopPropagation();
				handle_clear();
			}}
		/>
	</IconButtonWrapper>

	<FilePreview
		{i18n}
		on:select
		{selectable}
		{value}
		{height}
		on:change
		on:delete
		{allow_reordering}
	/>
{:else}
	<Upload
		on:load={handle_upload}
		filetype={file_types}
		{file_count}
		{max_file_size}
		{root}
		bind:dragging
		bind:uploading
		on:error
		{stream_handler}
		{upload}
		{height}
	>
		<slot />
	</Upload>
{/if}
