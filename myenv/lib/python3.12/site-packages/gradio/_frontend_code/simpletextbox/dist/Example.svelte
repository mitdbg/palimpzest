<script>import { onMount } from "svelte";
export let value;
export let type;
export let selected = false;
let size;
let el;
function set_styles(element, el_width) {
  element.style.setProperty(
    "--local-text-width",
    `${el_width && el_width < 150 ? el_width : 200}px`
  );
  element.style.whiteSpace = "unset";
}
onMount(() => {
  set_styles(el, size);
});
</script>

<div
	bind:clientWidth={size}
	bind:this={el}
	class:table={type === "table"}
	class:gallery={type === "gallery"}
	class:selected
>
	{value ? value : ""}
</div>

<style>
	.gallery {
		padding: var(--size-1) var(--size-2);
	}

	div {
		overflow: hidden;
		min-width: var(--local-text-width);

		white-space: nowrap;
	}
</style>
