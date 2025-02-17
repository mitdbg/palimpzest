<svelte:options accessors={true} />

<script>import { BlockTitle } from "@gradio/atoms";
import { Block } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
import { tick } from "svelte";
export let gradio;
export let label = "Textbox";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = "";
export let placeholder = "";
export let show_label;
export let scale = null;
export let min_width = void 0;
export let loading_status = void 0;
export let value_is_output = false;
export let interactive;
export let rtl = false;
export let root;
let el;
const container = true;
function handle_change() {
  gradio.dispatch("change");
  if (!value_is_output) {
    gradio.dispatch("input");
  }
}
async function handle_keypress(e) {
  await tick();
  if (e.key === "Enter") {
    e.preventDefault();
    gradio.dispatch("submit");
  }
}
$:
  if (value === null)
    value = "";
$:
  value, handle_change();
</script>

<Block
	{visible}
	{elem_id}
	{elem_classes}
	{scale}
	{min_width}
	allow_overflow={false}
	padding={true}
>
	{#if loading_status}
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
	{/if}

	<label class:container>
		<BlockTitle {root} {show_label} info={undefined}>{label}</BlockTitle>

		<input
			data-testid="textbox"
			type="text"
			class="scroll-hide"
			bind:value
			bind:this={el}
			{placeholder}
			disabled={!interactive}
			dir={rtl ? "rtl" : "ltr"}
			on:keypress={handle_keypress}
		/>
	</label>
</Block>

<style>
	label {
		display: block;
		width: 100%;
	}

	input {
		display: block;
		position: relative;
		outline: none !important;
		box-shadow: var(--input-shadow);
		background: var(--input-background-fill);
		padding: var(--input-padding);
		width: 100%;
		color: var(--body-text-color);
		font-weight: var(--input-text-weight);
		font-size: var(--input-text-size);
		line-height: var(--line-sm);
		border: none;
	}
	.container > input {
		border: var(--input-border-width) solid var(--input-border-color);
		border-radius: var(--input-radius);
	}
	input:disabled {
		-webkit-text-fill-color: var(--body-text-color);
		-webkit-opacity: 1;
		opacity: 1;
	}

	input:focus {
		box-shadow: var(--input-shadow-focus);
		border-color: var(--input-border-color-focus);
	}

	input::placeholder {
		color: var(--input-placeholder-color);
	}
</style>
