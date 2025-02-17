<script>import LikeDislike from "./LikeDislike.svelte";
import Copy from "./Copy.svelte";
import { Retry, Undo, Edit, Check, Clear } from "@gradio/icons";
import { IconButtonWrapper, IconButton } from "@gradio/atoms";
import { all_text, is_all_text } from "./utils";
export let likeable;
export let feedback_options;
export let show_retry;
export let show_undo;
export let show_edit;
export let in_edit_mode;
export let show_copy_button;
export let message;
export let position;
export let avatar;
export let generating;
export let current_feedback;
export let handle_action;
export let layout;
export let dispatch;
$:
  message_text = is_all_text(message) ? all_text(message) : "";
$:
  show_copy = show_copy_button && message && is_all_text(message);
</script>

{#if show_copy || show_retry || show_undo || show_edit || likeable}
	<div
		class="message-buttons-{position} {layout} message-buttons {avatar !==
			null && 'with-avatar'}"
	>
		<IconButtonWrapper top_panel={false}>
			{#if in_edit_mode}
				<IconButton
					label="Submit"
					Icon={Check}
					on:click={() => handle_action("edit_submit")}
					disabled={generating}
				/>
				<IconButton
					label="Cancel"
					Icon={Clear}
					on:click={() => handle_action("edit_cancel")}
					disabled={generating}
				/>
			{:else}
				{#if show_copy}
					<Copy
						value={message_text}
						on:copy={(e) => dispatch("copy", e.detail)}
					/>
				{/if}
				{#if show_retry}
					<IconButton
						Icon={Retry}
						label="Retry"
						on:click={() => handle_action("retry")}
						disabled={generating}
					/>
				{/if}
				{#if show_undo}
					<IconButton
						label="Undo"
						Icon={Undo}
						on:click={() => handle_action("undo")}
						disabled={generating}
					/>
				{/if}
				{#if show_edit}
					<IconButton
						label="Edit"
						Icon={Edit}
						on:click={() => handle_action("edit")}
						disabled={generating}
					/>
				{/if}
				{#if likeable}
					<LikeDislike
						{handle_action}
						{feedback_options}
						selected={current_feedback}
					/>
				{/if}
			{/if}
		</IconButtonWrapper>
	</div>
{/if}

<style>
	.bubble :global(.icon-button-wrapper) {
		margin: 0px calc(var(--spacing-xl) * 2);
	}

	.message-buttons {
		z-index: var(--layer-1);
	}
	.message-buttons-left {
		align-self: flex-start;
	}

	.bubble.message-buttons-right {
		align-self: flex-end;
	}

	.message-buttons-right :global(.icon-button-wrapper) {
		margin-left: auto;
	}

	.bubble.with-avatar {
		margin-left: calc(var(--spacing-xl) * 5);
		margin-right: calc(var(--spacing-xl) * 5);
	}

	.panel {
		display: flex;
		align-self: flex-start;
		z-index: var(--layer-1);
	}
</style>
