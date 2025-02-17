<svelte:options accessors={true} />

<script>import { File } from "@gradio/icons";
import { Block, BlockLabel } from "@gradio/atoms";
import DirectoryExplorer from "./shared/DirectoryExplorer.svelte";
import { StatusTracker } from "@gradio/statustracker";
import { _ } from "svelte-i18n";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value;
let old_value;
export let label;
export let show_label;
export let height;
export let min_height;
export let max_height;
export let file_count = "multiple";
export let root_dir;
export let glob;
export let ignore_glob;
export let loading_status;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let gradio;
export let server;
export let interactive;
$:
  rerender_key = [root_dir, glob, ignore_glob];
$:
  if (JSON.stringify(value) !== JSON.stringify(old_value)) {
    old_value = value;
    gradio.dispatch("change");
  }
</script>

<Block
	{visible}
	variant={value === null ? "dashed" : "solid"}
	border_mode={"base"}
	padding={false}
	{elem_id}
	{elem_classes}
	{container}
	{scale}
	{min_width}
	allow_overflow={true}
	overflow_behavior="auto"
	{height}
	{max_height}
	{min_height}
>
	<StatusTracker
		{...loading_status}
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
	/>
	<BlockLabel
		{show_label}
		Icon={File}
		label={label || "FileExplorer"}
		float={false}
	/>
	{#key rerender_key}
		<DirectoryExplorer
			bind:value
			{file_count}
			{interactive}
			ls_fn={server.ls}
		/>
	{/key}
</Block>
