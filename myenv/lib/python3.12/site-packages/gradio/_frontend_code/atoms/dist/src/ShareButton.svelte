<script>import IconButton from "./IconButton.svelte";
import { Community } from "@gradio/icons";
import { createEventDispatcher } from "svelte";
import { ShareError } from "@gradio/utils";
const dispatch = createEventDispatcher();
export let formatter;
export let value;
export let i18n;
let pending = false;
</script>

<IconButton
	Icon={Community}
	label={i18n("common.share")}
	{pending}
	on:click={async () => {
		try {
			pending = true;
			const formatted = await formatter(value);
			dispatch("share", {
				description: formatted
			});
		} catch (e) {
			console.error(e);
			let message = e instanceof ShareError ? e.message : "Share failed.";
			dispatch("error", message);
		} finally {
			pending = false;
		}
	}}
/>
