<script>export let title;
export let x;
export let y;
</script>

{#if title}
	{title}
{:else}
	{x} x {y}
{/if}
