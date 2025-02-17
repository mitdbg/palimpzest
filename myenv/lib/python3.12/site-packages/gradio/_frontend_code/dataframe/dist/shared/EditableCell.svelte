<script>import { createEventDispatcher } from "svelte";
import { MarkdownCode } from "@gradio/markdown-code";
export let edit;
export let value = "";
export let display_value = null;
export let styling = "";
export let header = false;
export let datatype = "str";
export let latex_delimiters;
export let clear_on_focus = false;
export let line_breaks = true;
export let editable = true;
export let root;
const dispatch = createEventDispatcher();
export let el;
$:
  _value = value;
function use_focus(node) {
  if (clear_on_focus) {
    _value = "";
  }
  requestAnimationFrame(() => {
    node.focus();
  });
  return {};
}
function handle_blur({
  currentTarget
}) {
  value = currentTarget.value;
  dispatch("blur");
}
function handle_keydown(event) {
  if (event.key === "Enter") {
    value = _value;
    dispatch("blur");
  }
  dispatch("keydown", event);
}
</script>

{#if edit}
	<input
		role="textbox"
		bind:this={el}
		bind:value={_value}
		class:header
		tabindex="-1"
		on:blur={handle_blur}
		on:mousedown|stopPropagation
		on:mouseup|stopPropagation
		on:click|stopPropagation
		use:use_focus
		on:keydown={handle_keydown}
	/>
{/if}

<span
	on:dblclick
	tabindex="-1"
	role="button"
	class:edit
	on:focus|preventDefault
	style={styling}
	class="table-cell-text"
	placeholder=" "
>
	{#if datatype === "html"}
		{@html value}
	{:else if datatype === "markdown"}
		<MarkdownCode
			message={value.toLocaleString()}
			{latex_delimiters}
			{line_breaks}
			chatbot={false}
			{root}
		/>
	{:else}
		{editable ? value : display_value || value}
	{/if}
</span>

<style>
	input {
		position: absolute;
		top: var(--size-2);
		right: var(--size-2);
		bottom: var(--size-2);
		left: var(--size-2);
		flex: 1 1 0%;
		transform: translateX(-0.1px);
		outline: none;
		border: none;
		background: transparent;
		cursor: text;
	}

	span {
		flex: 1 1 0%;
		outline: none;
		padding: var(--size-2);
		-webkit-user-select: text;
		-moz-user-select: text;
		-ms-user-select: text;
		user-select: text;
		cursor: text;
	}

	.header {
		transform: translateX(0);
		font-weight: var(--weight-bold);
	}

	.edit {
		opacity: 0;
		pointer-events: none;
	}
</style>
