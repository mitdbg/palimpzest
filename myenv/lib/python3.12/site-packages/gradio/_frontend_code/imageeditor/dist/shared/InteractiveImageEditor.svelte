<script context="module"></script>

<script>import { createEventDispatcher } from "svelte";
import {} from "@gradio/utils";
import { prepare_files } from "@gradio/client";
import {} from "./utils/commands";
import ImageEditor from "./ImageEditor.svelte";
import Layers from "./layers/Layers.svelte";
import {} from "./tools/Brush.svelte";
import {} from "./tools/Brush.svelte";
import { Tools, Crop, Brush, Sources } from "./tools";
import { BlockLabel } from "@gradio/atoms";
import { Image as ImageIcon } from "@gradio/icons";
import { inject } from "./utils/parse_placeholder";
export let brush;
export let eraser;
export let sources;
export let crop_size = null;
export let i18n;
export let root;
export let label = void 0;
export let show_label;
export let changeable = false;
export let value = {
  background: null,
  layers: [],
  composite: null
};
export let transforms = ["crop"];
export let layers;
export let accept_blobs;
export let status = "complete";
export let canvas_size;
export let fixed_canvas = false;
export let realtime;
export let upload;
export let stream_handler;
export let dragging;
export let placeholder = void 0;
export let dynamic_height = void 0;
export let height;
export let full_history = null;
const dispatch = createEventDispatcher();
let editor;
function is_not_null(o) {
  return !!o;
}
function is_file_data(o) {
  return !!o;
}
$:
  if (bg)
    dispatch("upload");
export async function get_data() {
  let blobs;
  try {
    blobs = await editor.get_blobs();
  } catch (e) {
    return { background: null, layers: [], composite: null };
  }
  const bg2 = blobs.background ? upload(
    await prepare_files([new File([blobs.background], "background.png")]),
    root
  ) : Promise.resolve(null);
  const layers2 = blobs.layers.filter(is_not_null).map(
    async (blob, i) => upload(await prepare_files([new File([blob], `layer_${i}.png`)]), root)
  );
  const composite = blobs.composite ? upload(
    await prepare_files([new File([blobs.composite], "composite.png")]),
    root
  ) : Promise.resolve(null);
  const [background, composite_, ...layers_] = await Promise.all([
    bg2,
    composite,
    ...layers2
  ]);
  return {
    background: Array.isArray(background) ? background[0] : background,
    layers: layers_.flatMap((layer) => Array.isArray(layer) ? layer : [layer]).filter(is_file_data),
    composite: Array.isArray(composite_) ? composite_[0] : composite_
  };
}
function handle_value(value2) {
  if (!editor)
    return;
  if (value2 == null) {
    editor.handle_remove();
    dispatch("receive_null");
  }
}
$:
  handle_value(value);
$:
  crop_constraint = crop_size;
let bg = false;
let history = false;
export let image_id = null;
$:
  editor && editor.set_tool && (sources && sources.length ? editor.set_tool("bg") : editor.set_tool("draw"));
function nextframe() {
  return new Promise((resolve) => setTimeout(() => resolve(), 30));
}
let uploading = false;
let pending = false;
async function handle_change(e) {
  if (!realtime)
    return;
  if (uploading) {
    pending = true;
    return;
  }
  uploading = true;
  await nextframe();
  const blobs = await editor.get_blobs();
  const images = [];
  let id = Math.random().toString(36).substring(2);
  if (blobs.background)
    images.push([
      id,
      "background",
      new File([blobs.background], "background.png"),
      null
    ]);
  if (blobs.composite)
    images.push([
      id,
      "composite",
      new File([blobs.composite], "composite.png"),
      null
    ]);
  blobs.layers.forEach((layer, i) => {
    if (layer)
      images.push([
        id,
        `layer`,
        new File([layer], `layer_${i}.png`),
        i
      ]);
  });
  await Promise.all(
    images.map(async ([image_id2, type, data, index]) => {
      return accept_blobs({
        binary: true,
        data: { file: data, id: image_id2, type, index }
      });
    })
  );
  image_id = id;
  dispatch("change");
  await nextframe();
  uploading = false;
  if (pending) {
    pending = false;
    uploading = false;
    handle_change(e);
  }
}
let active_mode = null;
let editor_height = height - 100;
let _dynamic_height;
$:
  dynamic_height = _dynamic_height;
$:
  [heading, paragraph] = placeholder ? inject(placeholder) : [false, false];
</script>

<BlockLabel
	{show_label}
	Icon={ImageIcon}
	label={label || i18n("image.image")}
/>
<ImageEditor
	on:history
	{canvas_size}
	crop_size={Array.isArray(crop_size) ? crop_size : undefined}
	bind:this={editor}
	bind:height={editor_height}
	bind:canvas_height={_dynamic_height}
	parent_height={height}
	{changeable}
	on:save
	on:change={handle_change}
	on:clear={() => dispatch("clear")}
	bind:history
	bind:bg
	{sources}
	crop_constraint={!!crop_constraint}
	{full_history}
>
	<Tools {i18n}>
		<Layers layer_files={value?.layers || null} enable_layers={layers} />

		<Sources
			bind:dragging
			{i18n}
			{root}
			{sources}
			{upload}
			{stream_handler}
			{canvas_size}
			bind:bg
			bind:active_mode
			background_file={value?.background || value?.composite || null}
			{fixed_canvas}
		></Sources>

		{#if transforms.includes("crop")}
			<Crop {crop_constraint} />
		{/if}
		{#if brush}
			<Brush
				color_mode={brush.color_mode}
				default_color={brush.default_color}
				default_size={brush.default_size}
				colors={brush.colors}
				mode="draw"
			/>
		{/if}

		{#if brush && eraser}
			<Brush default_size={eraser.default_size} mode="erase" />
		{/if}
	</Tools>

	{#if !bg && !history && active_mode !== "webcam" && status !== "error"}
		<div class="empty wrap" style:height={`${editor_height}px`}>
			{#if sources && sources.length}
				{#if heading || paragraph}
					{#if heading}
						<h2>{heading}</h2>
					{/if}
					{#if paragraph}
						<p>{paragraph}</p>
					{/if}
				{:else}
					<div>Upload an image</div>
				{/if}
			{/if}

			{#if sources && sources.length && brush && !placeholder}
				<div class="or">or</div>
			{/if}
			{#if brush && !placeholder}
				<div>select the draw tool to start</div>
			{/if}
		</div>
	{/if}
</ImageEditor>

<style>
	h2 {
		font-size: var(--text-xl);
	}

	p,
	h2 {
		white-space: pre-line;
	}

	.empty {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		position: absolute;
		height: 100%;
		width: 100%;
		left: 0;
		right: 0;
		margin: auto;
		z-index: var(--layer-top);
		text-align: center;
		color: var(--body-text-color);
		top: var(--size-8);
	}

	.wrap {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		color: var(--block-label-text-color);
		line-height: var(--line-md);
		font-size: var(--text-md);
		pointer-events: none;
	}

	.or {
		color: var(--body-text-color-subdued);
	}
</style>
