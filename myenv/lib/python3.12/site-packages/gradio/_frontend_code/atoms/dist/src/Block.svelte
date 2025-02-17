<script>export let height = void 0;
export let min_height = void 0;
export let max_height = void 0;
export let width = void 0;
export let elem_id = "";
export let elem_classes = [];
export let variant = "solid";
export let border_mode = "base";
export let padding = true;
export let type = "normal";
export let test_id = void 0;
export let explicit_call = false;
export let container = true;
export let visible = true;
export let allow_overflow = true;
export let overflow_behavior = "auto";
export let scale = null;
export let min_width = 0;
export let flex = false;
export let resizeable = false;
let element;
let tag = type === "fieldset" ? "fieldset" : "div";
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
$:
  if (!visible) {
    flex = false;
  }
const resize = (e) => {
  let prevY = e.clientY;
  const onMouseMove = (e2) => {
    const dy = e2.clientY - prevY;
    prevY = e2.clientY;
    element.style.height = `${element.offsetHeight + dy}px`;
  };
  const onMouseUp = () => {
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
  };
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
};
</script>

<svelte:element
	this={tag}
	bind:this={element}
	data-testid={test_id}
	id={elem_id}
	class:hidden={visible === false}
	class="block {elem_classes?.join(' ') || ''}"
	class:padded={padding}
	class:flex
	class:border_focus={border_mode === "focus"}
	class:border_contrast={border_mode === "contrast"}
	class:hide-container={!explicit_call && !container}
	style:height={get_dimension(height)}
	style:min-height={get_dimension(min_height)}
	style:max-height={get_dimension(max_height)}
	style:width={typeof width === "number"
		? `calc(min(${width}px, 100%))`
		: get_dimension(width)}
	style:border-style={variant}
	style:overflow={allow_overflow ? overflow_behavior : "hidden"}
	style:flex-grow={scale}
	style:min-width={`calc(min(${min_width}px, 100%))`}
	style:border-width="var(--block-border-width)"
	class:auto-margin={scale === null}
>
	<slot />
	{#if resizeable}
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<svg
			class="resize-handle"
			xmlns="http://www.w3.org/2000/svg"
			viewBox="0 0 10 10"
			on:mousedown={resize}
		>
			<line x1="1" y1="9" x2="9" y2="1" stroke="gray" stroke-width="0.5" />
			<line x1="5" y1="9" x2="9" y2="5" stroke="gray" stroke-width="0.5" />
		</svg>
	{/if}
</svelte:element>

<style>
	.block {
		position: relative;
		margin: 0;
		box-shadow: var(--block-shadow);
		border-width: var(--block-border-width);
		border-color: var(--block-border-color);
		border-radius: var(--block-radius);
		background: var(--block-background-fill);
		width: 100%;
		line-height: var(--line-sm);
	}

	.auto-margin {
		margin-left: auto;
		margin-right: auto;
	}

	.block.border_focus {
		border-color: var(--color-accent);
	}

	.block.border_contrast {
		border-color: var(--body-text-color);
	}

	.padded {
		padding: var(--block-padding);
	}

	.hidden {
		display: none;
	}

	.flex {
		display: flex;
		flex-direction: column;
	}
	.hide-container {
		margin: 0;
		box-shadow: none;
		--block-border-width: 0;
		background: transparent;
		padding: 0;
		overflow: visible;
	}
	.resize-handle {
		position: absolute;
		bottom: 0;
		right: 0;
		width: 10px;
		height: 10px;
		fill: var(--block-border-color);
		cursor: nwse-resize;
	}
</style>
