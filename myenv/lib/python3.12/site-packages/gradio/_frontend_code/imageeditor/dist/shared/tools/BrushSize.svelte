<script>import { click_outside } from "../utils/events";
import { createEventDispatcher } from "svelte";
import { BrushSize } from "@gradio/icons";
export let selected_size;
export let min;
export let max;
const dispatch = createEventDispatcher();
let width = 0;
let height = 0;
let c_width = 0;
let c_height = 0;
let wrap_el;
let anchor_right = false;
let anchor_top = false;
$: {
  if (wrap_el && (width || height || c_height || c_width)) {
    const box = wrap_el.getBoundingClientRect();
    anchor_right = box.width + 30 > width / 2;
    anchor_top = box.y < 80;
  }
}
</script>

<svelte:window bind:innerHeight={height} bind:innerWidth={width} />

<div
	class="wrap"
	use:click_outside={() => dispatch("click_outside")}
	bind:this={wrap_el}
	bind:clientWidth={c_width}
	bind:clientHeight={c_height}
	class:right={anchor_right}
	class:top={anchor_top}
	class:bottom={!anchor_top}
>
	<span>
		<BrushSize />
	</span>
	<input type="range" bind:value={selected_size} {min} {max} step={1} />
</div>

<style>
	.wrap {
		width: 100%;
		display: flex;
		gap: var(--size-4);
		background: var(--background-fill-secondary);
		padding: 0 var(--size-4);
		cursor: default;
		padding-top: var(--size-2-5);
	}

	input {
		width: 100%;
	}
	span {
		width: 26px;
		color: var(--body-text-color);
	}
</style>
