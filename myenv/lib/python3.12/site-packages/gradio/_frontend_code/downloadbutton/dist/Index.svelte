<script context="module">export { default as BaseButton } from "./shared/DownloadButton.svelte";
</script>

<script>import {} from "@gradio/client";
import DownloadButton from "./shared/DownloadButton.svelte";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value;
export let variant = "secondary";
export let interactive;
export let size = "lg";
export let scale = null;
export let icon = null;
export let min_width = void 0;
export let label = null;
export let gradio;
</script>

<DownloadButton
	{value}
	{variant}
	{elem_id}
	{elem_classes}
	{size}
	{scale}
	{icon}
	{min_width}
	{visible}
	disabled={!interactive}
	on:click={() => gradio.dispatch("click")}
>
	{label ? gradio.i18n(label) : ""}
</DownloadButton>
