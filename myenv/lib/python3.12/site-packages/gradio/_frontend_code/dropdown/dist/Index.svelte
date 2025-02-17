<script context="module">export { default as BaseDropdown } from "./shared/Dropdown.svelte";
export { default as BaseMultiselect } from "./shared/Multiselect.svelte";
export { default as BaseExample } from "./Example.svelte";
</script>

<script>import Multiselect from "./shared/Multiselect.svelte";
import Dropdown from "./shared/Dropdown.svelte";
import { Block } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
export let label = "Dropdown";
export let info = void 0;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let multiselect = false;
export let value = multiselect ? [] : void 0;
export let value_is_output = false;
export let max_choices = null;
export let choices;
export let show_label;
export let filterable;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let loading_status;
export let allow_custom_value = false;
export let root;
export let gradio;
export let interactive;
</script>

<Block
	{visible}
	{elem_id}
	{elem_classes}
	padding={container}
	allow_overflow={false}
	{scale}
	{min_width}
>
	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
		on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
	/>

	{#if multiselect}
		<Multiselect
			bind:value
			bind:value_is_output
			{choices}
			{max_choices}
			{root}
			{label}
			{info}
			{show_label}
			{allow_custom_value}
			{filterable}
			{container}
			i18n={gradio.i18n}
			on:change={() => gradio.dispatch("change")}
			on:input={() => gradio.dispatch("input")}
			on:select={(e) => gradio.dispatch("select", e.detail)}
			on:blur={() => gradio.dispatch("blur")}
			on:focus={() => gradio.dispatch("focus")}
			on:key_up={() => gradio.dispatch("key_up")}
			disabled={!interactive}
		/>
	{:else}
		<Dropdown
			bind:value
			bind:value_is_output
			{choices}
			{label}
			{root}
			{info}
			{show_label}
			{filterable}
			{allow_custom_value}
			{container}
			on:change={() => gradio.dispatch("change")}
			on:input={() => gradio.dispatch("input")}
			on:select={(e) => gradio.dispatch("select", e.detail)}
			on:blur={() => gradio.dispatch("blur")}
			on:focus={() => gradio.dispatch("focus")}
			on:key_up={(e) => gradio.dispatch("key_up", e.detail)}
			disabled={!interactive}
		/>
	{/if}
</Block>
