<script>export let value;
export let type;
export let selected = false;
export let choices;
let names = value.map(
  (val) => choices.find((pair) => pair[1] === val)?.[0]
).filter((name) => name !== void 0);
let names_string = names.join(", ");
</script>

<div
	class:table={type === "table"}
	class:gallery={type === "gallery"}
	class:selected
>
	{names_string}
</div>

<style>
	.gallery {
		padding: var(--size-1) var(--size-2);
	}
</style>
