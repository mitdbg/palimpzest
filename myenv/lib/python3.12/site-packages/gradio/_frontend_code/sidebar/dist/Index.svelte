<script>import Sidebar from "./shared/Sidebar.svelte";
import { StatusTracker } from "@gradio/statustracker";
export let open = true;
export let loading_status;
export let gradio;
export let width;
</script>

<StatusTracker
	autoscroll={gradio.autoscroll}
	i18n={gradio.i18n}
	{...loading_status}
/>

<Sidebar
	bind:open
	{width}
	on:expand={() => gradio.dispatch("expand")}
	on:collapse={() => gradio.dispatch("collapse")}
>
	<slot />
</Sidebar>
