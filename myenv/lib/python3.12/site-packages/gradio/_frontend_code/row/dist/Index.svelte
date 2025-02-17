<script>import { StatusTracker } from "@gradio/statustracker";
export let equal_height = true;
export let elem_id;
export let elem_classes = [];
export let visible = true;
export let variant = "default";
export let loading_status = void 0;
export let gradio = void 0;
export let show_progress = false;
export let height;
export let min_height;
export let max_height;
export let scale = null;
const get_dimension = (dimension_value) => {
  if (dimension_value === void 0) {
    return void 0;
  }
  if (typeof dimension_value === "number") {
    return dimension_value + "px";
  } else if (typeof dimension_value === "string") {
    return dimension_value;
  }
};
</script>

<div
	class:compact={variant === "compact"}
	class:panel={variant === "panel"}
	class:unequal-height={equal_height === false}
	class:stretch={equal_height}
	class:hide={!visible}
	class:grow-children={scale && scale >= 1}
	style:height={get_dimension(height)}
	style:max-height={get_dimension(max_height)}
	style:min-height={get_dimension(min_height)}
	style:flex-grow={scale}
	id={elem_id}
	class="row {elem_classes.join(' ')}"
>
	{#if loading_status && show_progress && gradio}
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			status={loading_status
				? loading_status.status == "pending"
					? "generating"
					: loading_status.status
				: null}
		/>
	{/if}
	<slot />
</div>

<style>
	div {
		display: flex;
		flex-wrap: wrap;
		gap: var(--layout-gap);
		width: var(--size-full);
		position: relative;
	}

	.hide {
		display: none;
	}
	.compact > :global(*),
	.compact :global(.box) {
		border-radius: 0;
	}
	.compact,
	.panel {
		border-radius: var(--container-radius);
		background: var(--background-fill-secondary);
		padding: var(--size-2);
	}
	.unequal-height {
		align-items: flex-start;
	}

	.stretch {
		align-items: stretch;
	}

	.stretch > :global(.column > *),
	.stretch > :global(.column > .form > *) {
		flex-grow: 1;
		flex-shrink: 0;
	}

	div > :global(*),
	div > :global(.form > *) {
		flex: 1 1 0%;
		flex-wrap: wrap;
		min-width: min(160px, 100%);
	}

	.grow-children > :global(.column) {
		align-self: stretch;
	}
</style>
