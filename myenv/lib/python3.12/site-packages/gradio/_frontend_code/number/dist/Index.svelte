<script>import { Block, BlockTitle } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
import { afterUpdate, tick } from "svelte";
export let gradio;
export let label = gradio.i18n("number.number");
export let info = void 0;
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let value = 0;
export let show_label;
export let minimum = void 0;
export let maximum = void 0;
export let loading_status;
export let value_is_output = false;
export let step = null;
export let interactive;
export let root;
function handle_change() {
  if (!isNaN(value) && value !== null) {
    gradio.dispatch("change");
    if (!value_is_output) {
      gradio.dispatch("input");
    }
  }
}
afterUpdate(() => {
  value_is_output = false;
});
async function handle_keypress(e) {
  await tick();
  if (e.key === "Enter") {
    e.preventDefault();
    gradio.dispatch("submit");
  }
}
$:
  value, handle_change();
$:
  disabled = !interactive;
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
	<label class="block" class:container>
		<BlockTitle {root} {show_label} {info}>{label}</BlockTitle>
		<input
			aria-label={label}
			type="number"
			bind:value
			min={minimum}
			max={maximum}
			{step}
			on:keypress={handle_keypress}
			on:blur={() => gradio.dispatch("blur")}
			on:focus={() => gradio.dispatch("focus")}
			{disabled}
		/>
	</label>
</Block>

<style>
	label:not(.container),
	label:not(.container) > input {
		height: 100%;
		border: none;
	}
	.container > input {
		border: var(--input-border-width) solid var(--input-border-color);
		border-radius: var(--input-radius);
	}
	input[type="number"] {
		display: block;
		position: relative;
		outline: none !important;
		box-shadow: var(--input-shadow);
		background: var(--input-background-fill);
		padding: var(--input-padding);
		width: 100%;
		color: var(--body-text-color);
		font-size: var(--input-text-size);
		line-height: var(--line-sm);
	}
	input:disabled {
		-webkit-text-fill-color: var(--body-text-color);
		-webkit-opacity: 1;
		opacity: 1;
	}

	input:focus {
		box-shadow: var(--input-shadow-focus);
		border-color: var(--input-border-color-focus);
		background: var(--input-background-fill-focus);
	}

	input::placeholder {
		color: var(--input-placeholder-color);
	}

	input:out-of-range {
		border: var(--input-border-width) solid var(--error-border-color);
	}
</style>
