<script>import { Block, BlockTitle } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
export let label = "Dropdown";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value;
export let value_is_output = false;
export let choices;
export let show_label;
export let scale = null;
export let min_width = void 0;
export let loading_status;
export let root;
export let gradio;
export let interactive;
const container = true;
let display_value;
let candidate;
function handle_change() {
  gradio.dispatch("change");
  if (!value_is_output) {
    gradio.dispatch("input");
  }
}
$:
  if (display_value) {
    candidate = choices.filter((choice) => choice[0] === display_value);
    value = candidate.length ? candidate[0][1] : "";
  }
$:
  value, handle_change();
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
		<select disabled={!interactive} bind:value={display_value}>
			{#each choices as choice}
				<option>{choice[0]}</option>
			{/each}
		</select>
	</label>
</Block>

<style>
	select {
		--ring-color: transparent;
		display: block;
		position: relative;
		outline: none !important;
		box-shadow:
			0 0 0 var(--shadow-spread) var(--ring-color),
			var(--shadow-inset);
		border: var(--input-border-width) solid var(--border-color-primary);
		border-radius: var(--radius-lg);
		background-color: var(--input-background-base);
		padding: var(--size-2-5);
		width: 100%;
		color: var(--color-text-body);
		font-size: var(--scale-00);
		line-height: var(--line-sm);
	}

	select:focus {
		--ring-color: var(--color-focus-ring);
		border-color: var(--input-border-color-focus);
	}

	select::placeholder {
		color: var(--color-text-placeholder);
	}

	select[disabled] {
		cursor: not-allowed;
		box-shadow: none;
	}
</style>
