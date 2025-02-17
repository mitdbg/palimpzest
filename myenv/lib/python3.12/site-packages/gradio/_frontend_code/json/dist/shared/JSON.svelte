<script>import { onDestroy } from "svelte";
import { JSON as JSONIcon } from "@gradio/icons";
import { Empty, IconButtonWrapper, IconButton } from "@gradio/atoms";
import JSONNode from "./JSONNode.svelte";
import { Copy, Check } from "@gradio/icons";
export let value = {};
export let open = false;
export let theme_mode = "system";
export let show_indices = false;
export let label_height;
$:
  json_max_height = `calc(100% - ${label_height}px)`;
let copied = false;
let timer;
function copy_feedback() {
  copied = true;
  if (timer)
    clearTimeout(timer);
  timer = setTimeout(() => {
    copied = false;
  }, 1e3);
}
async function handle_copy() {
  if ("clipboard" in navigator) {
    await navigator.clipboard.writeText(JSON.stringify(value, null, 2));
    copy_feedback();
  }
}
function is_empty(obj) {
  return obj && Object.keys(obj).length === 0 && Object.getPrototypeOf(obj) === Object.prototype && JSON.stringify(obj) === JSON.stringify({});
}
onDestroy(() => {
  if (timer)
    clearTimeout(timer);
});
</script>

{#if value && value !== '""' && !is_empty(value)}
	<IconButtonWrapper>
		<IconButton
			show_label={false}
			label={copied ? "Copied" : "Copy"}
			Icon={copied ? Check : Copy}
			on:click={() => handle_copy()}
		/>
	</IconButtonWrapper>
	<div class="json-holder" style:max-height={json_max_height}>
		<JSONNode
			{value}
			depth={0}
			is_root={true}
			{open}
			{theme_mode}
			{show_indices}
		/>
	</div>
{:else}
	<div class="empty-wrapper">
		<Empty>
			<JSONIcon />
		</Empty>
	</div>
{/if}

<style>
	:global(.copied svg) {
		animation: fade ease 300ms;
		animation-fill-mode: forwards;
	}

	@keyframes fade {
		0% {
			opacity: 0;
		}
		100% {
			opacity: 1;
		}
	}

	.json-holder {
		padding: var(--size-2);
		overflow-y: auto;
	}

	.empty-wrapper {
		min-height: calc(var(--size-32) - 20px);
		height: 100%;
	}
</style>
