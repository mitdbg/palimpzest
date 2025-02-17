<script context="module">export { default as BaseButton } from "./shared/Button.svelte";
</script>

<script>import {} from "@gradio/client";
import Button from "./shared/Button.svelte";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value;
export let variant = "secondary";
export let interactive;
export let size = "lg";
export let scale = null;
export let icon = null;
export let link = null;
export let min_width = void 0;
export let gradio;
</script>

<Button
	{value}
	{variant}
	{elem_id}
	{elem_classes}
	{size}
	{scale}
	{link}
	{icon}
	{min_width}
	{visible}
	disabled={!interactive}
	on:click={() => gradio.dispatch("click")}
>
	{value ? gradio.i18n(value) : ""}
</Button>
