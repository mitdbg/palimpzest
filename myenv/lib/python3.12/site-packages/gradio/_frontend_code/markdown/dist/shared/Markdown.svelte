<script>import { createEventDispatcher } from "svelte";
import { copy, css_units } from "@gradio/utils";
import { Copy, Check } from "@gradio/icons";
import { IconButton, IconButtonWrapper } from "@gradio/atoms";
import { MarkdownCode } from "@gradio/markdown-code";
export let elem_classes = [];
export let visible = true;
export let value;
export let min_height = void 0;
export let rtl = false;
export let sanitize_html = true;
export let line_breaks = false;
export let latex_delimiters;
export let header_links = false;
export let height = void 0;
export let show_copy_button = false;
export let root;
export let loading_status = void 0;
let copied = false;
let timer;
const dispatch = createEventDispatcher();
$:
  value, dispatch("change");
async function handle_copy() {
  if ("clipboard" in navigator) {
    await navigator.clipboard.writeText(value);
    dispatch("copy", { value });
    copy_feedback();
  }
}
function copy_feedback() {
  copied = true;
  if (timer)
    clearTimeout(timer);
  timer = setTimeout(() => {
    copied = false;
  }, 1e3);
}
</script>

<div
	class="prose {elem_classes?.join(' ') || ''}"
	class:hide={!visible}
	data-testid="markdown"
	dir={rtl ? "rtl" : "ltr"}
	use:copy
	style={height ? `max-height: ${css_units(height)}; overflow-y: auto;` : ""}
	style:min-height={min_height && loading_status?.status !== "pending"
		? css_units(min_height)
		: undefined}
>
	{#if show_copy_button}
		<IconButtonWrapper>
			<IconButton
				Icon={copied ? Check : Copy}
				on:click={handle_copy}
				label={copied ? "Copied conversation" : "Copy conversation"}
			></IconButton>
		</IconButtonWrapper>
	{/if}
	<MarkdownCode
		message={value}
		{latex_delimiters}
		{sanitize_html}
		{line_breaks}
		chatbot={false}
		{header_links}
		{root}
	/>
</div>

<style>
	div :global(.math.inline) {
		fill: var(--body-text-color);
		display: inline-block;
		vertical-align: middle;
		padding: var(--size-1-5) -var(--size-1);
		color: var(--body-text-color);
	}

	div :global(.math.inline svg) {
		display: inline;
		margin-bottom: 0.22em;
	}

	div {
		max-width: 100%;
	}

	.hide {
		display: none;
	}
</style>
