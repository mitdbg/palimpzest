<script context="module">export { default as BaseTabItem } from "./shared/TabItem.svelte";
</script>

<script>import TabItem from "./shared/TabItem.svelte";
export let elem_id = "";
export let elem_classes = [];
export let label;
export let id;
export let gradio;
export let visible = true;
export let interactive = true;
export let order;
export let scale;
</script>

<TabItem
	{elem_id}
	{elem_classes}
	{label}
	{visible}
	{interactive}
	{id}
	{order}
	{scale}
	on:select={({ detail }) => gradio?.dispatch("select", detail)}
>
	<slot />
</TabItem>
