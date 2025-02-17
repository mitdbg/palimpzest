<script>export let value;
export let type;
export let selected = false;
</script>

<div
	class:table={type === "table"}
	class:gallery={type === "gallery"}
	class:selected
>
	{value !== null ? value : ""}
</div>

<style>
	.gallery {
		padding: var(--size-1) var(--size-2);
	}
</style>
