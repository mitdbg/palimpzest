<script>export let value;
export let type;
export let selected = false;
</script>

<div
	style="background-color: {value ? value : 'black'}"
	class:table={type === "table"}
	class:gallery={type === "gallery"}
	class:selected
/>

<style>
	div {
		width: var(--size-10);
		height: var(--size-10);
	}
	.table {
		margin: 0 auto;
	}
</style>
