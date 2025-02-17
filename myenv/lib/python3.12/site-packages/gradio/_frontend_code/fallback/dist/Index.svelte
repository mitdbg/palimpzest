<script>import { JsonView } from "@zerodevx/svelte-json-view";
import { Block, Info } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = false;
export let container = true;
export let scale = null;
export let min_width = void 0;
export let loading_status;
export let gradio;
</script>

<Block {visible} {elem_id} {elem_classes} {container} {scale} {min_width}>
	{#if loading_status}
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
	{/if}

	<JsonView json={value} />
</Block>
