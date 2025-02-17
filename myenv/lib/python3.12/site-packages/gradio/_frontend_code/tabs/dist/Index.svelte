<script context="module">export { default as BaseTabs, TABS } from "./shared/Tabs.svelte";
</script>

<script>import { createEventDispatcher } from "svelte";
import Tabs, {} from "./shared/Tabs.svelte";
const dispatch = createEventDispatcher();
export let visible = true;
export let elem_id = "";
export let elem_classes = [];
export let selected;
export let initial_tabs = [];
export let gradio;
$:
  dispatch("prop_change", { selected });
</script>

<Tabs
	{visible}
	{elem_id}
	{elem_classes}
	bind:selected
	on:change={() => gradio?.dispatch("change")}
	on:select={(e) => gradio?.dispatch("select", e.detail)}
	{initial_tabs}
>
	<slot />
</Tabs>
