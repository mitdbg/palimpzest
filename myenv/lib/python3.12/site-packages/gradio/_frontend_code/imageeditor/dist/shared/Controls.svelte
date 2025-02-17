<script>import { createEventDispatcher } from "svelte";
import { IconButton, IconButtonWrapper } from "@gradio/atoms";
import { Undo, Redo, Check, Trash } from "@gradio/icons";
export let can_undo = false;
export let can_redo = false;
export let can_save = false;
export let changeable = false;
const dispatch = createEventDispatcher();
</script>

<IconButtonWrapper>
	{#if changeable}
		<IconButton
			disabled={!can_save}
			Icon={Check}
			label="Save changes"
			on:click={(event) => {
				dispatch("save");
				event.stopPropagation();
			}}
			background={"var(--color-green-500)"}
			color={"#fff"}
		/>
	{/if}
	<IconButton
		disabled={!can_undo}
		Icon={Undo}
		label="Undo"
		on:click={(event) => {
			dispatch("undo");
			event.stopPropagation();
		}}
	/>
	<IconButton
		disabled={!can_redo}
		Icon={Redo}
		label="Redo"
		on:click={(event) => {
			dispatch("redo");
			event.stopPropagation();
		}}
	/>
	<IconButton
		Icon={Trash}
		label="Clear canvas"
		on:click={(event) => {
			dispatch("remove_image");
			event.stopPropagation();
		}}
	/>
</IconButtonWrapper>
