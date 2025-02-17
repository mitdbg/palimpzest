<script>import { BlockLabel, Empty } from "@gradio/atoms";
import { File } from "@gradio/icons";
import FilePreview from "./FilePreview.svelte";
export let value = null;
export let label;
export let show_label = true;
export let selectable = false;
export let height = void 0;
export let i18n;
</script>

<BlockLabel
	{show_label}
	float={value === null}
	Icon={File}
	label={label || "File"}
/>

{#if value && (Array.isArray(value) ? value.length > 0 : true)}
	<FilePreview {i18n} {selectable} on:select on:download {value} {height} />
{:else}
	<Empty unpadded_box={true} size="large"><File /></Empty>
{/if}
