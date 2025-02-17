<script>import { Microphone, Upload, Webcam, ImagePaste } from "@gradio/icons";
export let sources;
export let active_source;
export let handle_clear = () => {
};
export let handle_select = () => {
};
$:
  unique_sources = [...new Set(sources)];
async function handle_select_source(source) {
  handle_clear();
  active_source = source;
  handle_select(source);
}
</script>

{#if unique_sources.length > 1}
	<span class="source-selection" data-testid="source-select">
		{#if sources.includes("upload")}
			<button
				class="icon"
				class:selected={active_source === "upload" || !active_source}
				aria-label="Upload file"
				on:click={() => handle_select_source("upload")}><Upload /></button
			>
		{/if}

		{#if sources.includes("microphone")}
			<button
				class="icon"
				class:selected={active_source === "microphone"}
				aria-label="Record audio"
				on:click={() => handle_select_source("microphone")}
				><Microphone /></button
			>
		{/if}

		{#if sources.includes("webcam")}
			<button
				class="icon"
				class:selected={active_source === "webcam"}
				aria-label="Capture from camera"
				on:click={() => handle_select_source("webcam")}><Webcam /></button
			>
		{/if}
		{#if sources.includes("clipboard")}
			<button
				class="icon"
				class:selected={active_source === "clipboard"}
				aria-label="Paste from clipboard"
				on:click={() => handle_select_source("clipboard")}
				><ImagePaste /></button
			>
		{/if}
	</span>
{/if}

<style>
	.source-selection {
		display: flex;
		align-items: center;
		justify-content: center;
		border-top: 1px solid var(--border-color-primary);
		width: 100%;
		margin-left: auto;
		margin-right: auto;
		height: var(--size-10);
	}

	.icon {
		width: 22px;
		height: 22px;
		margin: var(--spacing-lg) var(--spacing-xs);
		padding: var(--spacing-xs);
		color: var(--neutral-400);
		border-radius: var(--radius-md);
	}

	.selected {
		color: var(--color-accent);
	}

	.icon:hover,
	.icon:focus {
		color: var(--color-accent);
	}
</style>
