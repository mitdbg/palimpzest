<script>export let value;
export let type;
export let selected = false;
</script>

<pre
	class:table={type === "table"}
	class:gallery={type === "gallery"}
	class:selected>{value ? value : ""}</pre>

<style>
	pre {
		text-align: left;
	}
	.gallery {
		padding: var(--size-1) var(--size-2);
	}
</style>
