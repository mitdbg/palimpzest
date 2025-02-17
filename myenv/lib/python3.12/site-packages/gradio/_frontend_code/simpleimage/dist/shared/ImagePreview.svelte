<script>import {
  BlockLabel,
  Empty,
  IconButton,
  IconButtonWrapper
} from "@gradio/atoms";
import { Download } from "@gradio/icons";
import { DownloadLink } from "@gradio/wasm/svelte";
import { Image as ImageIcon } from "@gradio/icons";
import {} from "@gradio/client";
export let value;
export let label = void 0;
export let show_label;
export let show_download_button = true;
export let selectable = false;
export let i18n;
</script>

<BlockLabel
	{show_label}
	Icon={ImageIcon}
	label={label || i18n("image.image")}
/>
{#if value === null || !value.url}
	<Empty unpadded_box={true} size="large"><ImageIcon /></Empty>
{:else}
	<IconButtonWrapper>
		{#if show_download_button}
			<DownloadLink href={value.url} download={value.orig_name || "image"}>
				<IconButton Icon={Download} label={i18n("common.download")} />
			</DownloadLink>
		{/if}
	</IconButtonWrapper>
	<button>
		<div class:selectable class="image-container">
			<img src={value.url} alt="" loading="lazy" />
		</div>
	</button>
{/if}

<style>
	.image-container {
		height: 100%;
	}
	.image-container :global(img),
	button {
		width: var(--size-full);
		height: var(--size-full);
		object-fit: scale-down;
		display: block;
		border-radius: var(--radius-lg);
	}

	.selectable {
		cursor: crosshair;
	}
</style>
