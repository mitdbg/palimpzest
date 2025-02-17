<script>export let value;
export let type;
export let selected = false;
export let choices;
let name_string;
if (value === null) {
  name_string = "";
} else {
  let name = choices.find((pair) => pair[1] === value);
  name_string = name ? name[0] : "";
}
</script>

<div
	class:table={type === "table"}
	class:gallery={type === "gallery"}
	class:selected
>
	{name_string}
</div>

<style>
	.gallery {
		padding: var(--size-1) var(--size-2);
	}
</style>
