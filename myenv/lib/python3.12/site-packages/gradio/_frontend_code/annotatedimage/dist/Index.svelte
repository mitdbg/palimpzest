<script>import { onMount } from "svelte";
import {
  Block,
  BlockLabel,
  Empty,
  IconButtonWrapper,
  FullscreenButton
} from "@gradio/atoms";
import { Image, Maximize, Minimize } from "@gradio/icons";
import { StatusTracker } from "@gradio/statustracker";
import {} from "@gradio/client";
import { resolve_wasm_src } from "@gradio/wasm/svelte";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let value = null;
let old_value = null;
let _value = null;
export let gradio;
export let label = gradio.i18n("annotated_image.annotated_image");
export let show_label = true;
export let show_legend = true;
export let height;
export let width;
export let color_map;
export let container = true;
export let scale = null;
export let min_width = void 0;
let active = null;
export let loading_status;
export let show_fullscreen_button = true;
let image_container;
let is_full_screen = false;
let latest_promise = null;
$: {
  if (value !== old_value) {
    old_value = value;
    gradio.dispatch("change");
  }
  if (value) {
    const normalized_value = {
      image: value.image,
      annotations: value.annotations.map((ann) => ({
        image: ann.image,
        label: ann.label
      }))
    };
    _value = normalized_value;
    const image_url_promise = resolve_wasm_src(normalized_value.image.url);
    const annotation_urls_promise = Promise.all(
      normalized_value.annotations.map(
        (ann) => resolve_wasm_src(ann.image.url)
      )
    );
    const current_promise = Promise.all([
      image_url_promise,
      annotation_urls_promise
    ]);
    latest_promise = current_promise;
    current_promise.then(([image_url, annotation_urls]) => {
      if (latest_promise !== current_promise) {
        return;
      }
      const async_resolved_value = {
        image: {
          ...normalized_value.image,
          url: image_url ?? void 0
        },
        annotations: normalized_value.annotations.map((ann, i) => ({
          ...ann,
          image: {
            ...ann.image,
            url: annotation_urls[i] ?? void 0
          }
        }))
      };
      _value = async_resolved_value;
    });
  } else {
    _value = null;
  }
}
function handle_mouseover(_label) {
  active = _label;
}
function handle_mouseout() {
  active = null;
}
function handle_click(i, value2) {
  gradio.dispatch("select", {
    value: label,
    index: i
  });
}
</script>

<Block
	{visible}
	{elem_id}
	{elem_classes}
	padding={false}
	{height}
	{width}
	allow_overflow={false}
	{container}
	{scale}
	{min_width}
>
	<StatusTracker
		autoscroll={gradio.autoscroll}
		i18n={gradio.i18n}
		{...loading_status}
	/>
	<BlockLabel
		{show_label}
		Icon={Image}
		label={label || gradio.i18n("image.image")}
	/>

	<div class="container">
		{#if _value == null}
			<Empty size="large" unpadded_box={true}><Image /></Empty>
		{:else}
			<div class="image-container" bind:this={image_container}>
				<IconButtonWrapper>
					{#if show_fullscreen_button}
						<FullscreenButton
							container={image_container}
							on:fullscreenchange={(e) => (is_full_screen = e.detail)}
						/>
					{/if}
				</IconButtonWrapper>

				<img
					class="base-image"
					class:fit-height={height && !is_full_screen}
					src={_value ? _value.image.url : null}
					alt="the base file that is annotated"
				/>
				{#each _value ? _value?.annotations : [] as ann, i}
					<img
						alt="segmentation mask identifying {label} within the uploaded file"
						class="mask fit-height"
						class:fit-height={!is_full_screen}
						class:active={active == ann.label}
						class:inactive={active != ann.label && active != null}
						src={ann.image.url}
						style={color_map && ann.label in color_map
							? null
							: `filter: hue-rotate(${Math.round(
									(i * 360) / _value?.annotations.length
								)}deg);`}
					/>
				{/each}
			</div>
			{#if show_legend && _value}
				<div class="legend">
					{#each _value.annotations as ann, i}
						<button
							class="legend-item"
							style="background-color: {color_map && ann.label in color_map
								? color_map[ann.label] + '88'
								: `hsla(${Math.round(
										(i * 360) / _value.annotations.length
									)}, 100%, 50%, 0.3)`}"
							on:mouseover={() => handle_mouseover(ann.label)}
							on:focus={() => handle_mouseover(ann.label)}
							on:mouseout={() => handle_mouseout()}
							on:blur={() => handle_mouseout()}
							on:click={() => handle_click(i, ann.label)}
						>
							{ann.label}
						</button>
					{/each}
				</div>
			{/if}
		{/if}
	</div>
</Block>

<style>
	.base-image {
		display: block;
		width: 100%;
		height: auto;
	}
	.container {
		display: flex;
		position: relative;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		width: var(--size-full);
		height: var(--size-full);
	}
	.image-container {
		position: relative;
		top: 0;
		left: 0;
		flex-grow: 1;
		width: 100%;
		overflow: hidden;
	}
	.fit-height {
		top: 0;
		left: 0;
		width: 100%;
		height: 100%;
		object-fit: contain;
	}
	.mask {
		opacity: 0.85;
		transition: all 0.2s ease-in-out;
		position: absolute;
	}
	.image-container:hover .mask {
		opacity: 0.3;
	}
	.mask.active {
		opacity: 1;
	}
	.mask.inactive {
		opacity: 0;
	}
	.legend {
		display: flex;
		flex-direction: row;
		flex-wrap: wrap;
		align-content: center;
		justify-content: center;
		align-items: center;
		gap: var(--spacing-sm);
		padding: var(--spacing-sm);
	}
	.legend-item {
		display: flex;
		flex-direction: row;
		align-items: center;
		cursor: pointer;
		border-radius: var(--radius-sm);
		padding: var(--spacing-sm);
	}
</style>
