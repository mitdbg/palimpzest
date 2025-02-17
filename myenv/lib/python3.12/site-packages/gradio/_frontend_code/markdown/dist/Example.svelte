<script>import { MarkdownCode } from "@gradio/markdown-code";
export let value;
export let type;
export let selected = false;
export let sanitize_html;
export let line_breaks;
export let latex_delimiters;
export let root;
</script>

<div
	class:table={type === "table"}
	class:gallery={type === "gallery"}
	class:selected
	class="prose"
>
	<MarkdownCode
		message={value ? value : ""}
		{latex_delimiters}
		{sanitize_html}
		{line_breaks}
		chatbot={false}
		{root}
	/>
</div>

<style>
	.gallery {
		padding: var(--size-1) var(--size-2);
	}
</style>
