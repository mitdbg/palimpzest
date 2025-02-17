<script>import { File } from "@gradio/icons";
import Component from "./Component.svelte";
import { MarkdownCode as Markdown } from "@gradio/markdown-code";
export let latex_delimiters;
export let sanitize_html;
export let _fetch;
export let i18n;
export let line_breaks;
export let upload;
export let target;
export let root;
export let theme_mode;
export let _components;
export let render_markdown;
export let scroll;
export let allow_file_downloads;
export let display_consecutive_in_same_bubble;
export let thought_index;
export let message;
</script>

{#if message.type === "text"}
	<div class="message-content">
		<Markdown
			message={message.content}
			{latex_delimiters}
			{sanitize_html}
			{render_markdown}
			{line_breaks}
			on:load={scroll}
			{root}
		/>
	</div>
{:else if message.type === "component" && message.content.component in _components}
	<Component
		{target}
		{theme_mode}
		props={message.content.props}
		type={message.content.component}
		components={_components}
		value={message.content.value}
		display_icon_button_wrapper_top_corner={thought_index > 0 &&
			display_consecutive_in_same_bubble}
		{i18n}
		{upload}
		{_fetch}
		on:load={() => scroll()}
		{allow_file_downloads}
	/>
{:else if message.type === "component" && message.content.component === "file"}
	<div class="file-container">
		<div class="file-icon">
			<File />
		</div>
		<div class="file-info">
			<a
				data-testid="chatbot-file"
				class="file-link"
				href={message.content.value.url}
				target="_blank"
				download={window.__is_colab__
					? null
					: message.content.value?.orig_name ||
						message.content.value?.path.split("/").pop() ||
						"file"}
			>
				<span class="file-name"
					>{message.content.value?.orig_name ||
						message.content.value?.path.split("/").pop() ||
						"file"}</span
				>
			</a>
			<span class="file-type"
				>{(
					message.content.value?.orig_name ||
					message.content.value?.path ||
					""
				)
					.split(".")
					.pop()
					.toUpperCase()}</span
			>
		</div>
	</div>
{/if}

<style>
	.file-container {
		display: flex;
		align-items: center;
		gap: var(--spacing-lg);
		padding: var(--spacing-lg);
		border-radius: var(--radius-lg);
		width: fit-content;
		margin: var(--spacing-sm) 0;
	}

	.file-icon {
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--body-text-color);
	}

	.file-icon :global(svg) {
		width: var(--size-7);
		height: var(--size-7);
	}

	.file-info {
		display: flex;
		flex-direction: column;
	}

	.file-link {
		text-decoration: none;
		color: var(--body-text-color);
		display: flex;
		flex-direction: column;
		gap: var(--spacing-xs);
	}

	.file-name {
		font-family: var(--font);
		font-size: var(--text-md);
		font-weight: 500;
	}

	.file-type {
		font-family: var(--font);
		font-size: var(--text-sm);
		color: var(--body-text-color-subdued);
		text-transform: uppercase;
	}
</style>
