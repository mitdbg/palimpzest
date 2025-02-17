<script>import Accordion from "./shared/Accordion.svelte";
import { Block } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
import Column from "@gradio/column";
export let label;
export let elem_id;
export let elem_classes;
export let visible = true;
export let open = true;
export let loading_status;
export let gradio;
</script>

<Block {elem_id} {elem_classes} {visible}>
	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
	/>

	<Accordion
		{label}
		bind:open
		on:expand={() => gradio.dispatch("expand")}
		on:collapse={() => gradio.dispatch("collapse")}
	>
		<Column>
			<slot />
		</Column>
	</Accordion>
</Block>
