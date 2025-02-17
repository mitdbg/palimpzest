<script>export let value;
export let category;
export let active;
export let labelToEdit;
export let indexOfLabel;
export let text;
export let handleValueChange;
export let isScoresMode = false;
export let _color_map;
let _input_value = category;
function handleInput(e) {
  let target = e.target;
  if (target) {
    _input_value = target.value;
  }
}
function updateLabelValue(e, elementIndex, text2) {
  let target = e.target;
  value = [
    ...value.slice(0, elementIndex),
    {
      token: text2,
      class_or_confidence: target.value === "" ? null : isScoresMode ? Number(target.value) : target.value
    },
    ...value.slice(elementIndex + 1)
  ];
  handleValueChange();
}
function clearPlaceHolderOnFocus(e) {
  let target = e.target;
  if (target && target.placeholder)
    target.placeholder = "";
}
</script>

<!-- svelte-ignore a11y-autofocus -->
<!-- autofocus should not be disorienting for a screen reader users
as input is only rendered once a new label is created -->
{#if !isScoresMode}
	<input
		class="label-input"
		autofocus
		id={`label-input-${indexOfLabel}`}
		type="text"
		placeholder="label"
		value={category}
		style:background-color={category === null || (active && active !== category)
			? ""
			: _color_map[category].primary}
		style:width={_input_value
			? _input_value.toString()?.length + 4 + "ch"
			: "8ch"}
		on:input={handleInput}
		on:blur={(e) => updateLabelValue(e, indexOfLabel, text)}
		on:keydown={(e) => {
			if (e.key === "Enter") {
				updateLabelValue(e, indexOfLabel, text);
				labelToEdit = -1;
			}
		}}
		on:focus={clearPlaceHolderOnFocus}
	/>
{:else}
	<input
		class="label-input"
		autofocus
		type="number"
		step="0.1"
		style={"background-color: rgba(" +
			(typeof category === "number" && category < 0
				? "128, 90, 213," + -category
				: "239, 68, 60," + category) +
			")"}
		value={category}
		style:width="7ch"
		on:input={handleInput}
		on:blur={(e) => updateLabelValue(e, indexOfLabel, text)}
		on:keydown={(e) => {
			if (e.key === "Enter") {
				updateLabelValue(e, indexOfLabel, text);
				labelToEdit = -1;
			}
		}}
	/>
{/if}

<style>
	.label-input {
		transition: 150ms;
		margin-top: 1px;
		margin-right: calc(var(--size-1));
		border-radius: var(--radius-xs);
		padding: 1px 5px;
		color: black;
		font-weight: var(--weight-bold);
		font-size: var(--text-sm);
		text-transform: uppercase;
		line-height: 1;
		color: white;
	}

	.label-input::placeholder {
		color: rgba(1, 1, 1, 0.5);
	}
</style>
