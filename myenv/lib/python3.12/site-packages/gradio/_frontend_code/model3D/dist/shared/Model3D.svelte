<script>import { BlockLabel, IconButton, IconButtonWrapper } from "@gradio/atoms";
import { File, Download, Undo } from "@gradio/icons";
import { dequal } from "dequal";
export let value;
export let display_mode = "solid";
export let clear_color = [0, 0, 0, 0];
export let label = "";
export let show_label;
export let i18n;
export let zoom_speed = 1;
export let pan_speed = 1;
export let camera_position = [
  null,
  null,
  null
];
export let has_change_history = false;
let current_settings = { camera_position, zoom_speed, pan_speed };
let use_3dgs = false;
let Canvas3DGSComponent;
let Canvas3DComponent;
async function loadCanvas3D() {
  const module = await import("./Canvas3D.svelte");
  return module.default;
}
async function loadCanvas3DGS() {
  const module = await import("./Canvas3DGS.svelte");
  return module.default;
}
$:
  if (value) {
    use_3dgs = value.path.endsWith(".splat") || value.path.endsWith(".ply");
    if (use_3dgs) {
      loadCanvas3DGS().then((component) => {
        Canvas3DGSComponent = component;
      });
    } else {
      loadCanvas3D().then((component) => {
        Canvas3DComponent = component;
      });
    }
  }
let canvas3d;
function handle_undo() {
  canvas3d?.reset_camera_position(camera_position, zoom_speed, pan_speed);
}
$: {
  if (!dequal(current_settings.camera_position, camera_position) || current_settings.zoom_speed !== zoom_speed || current_settings.pan_speed !== pan_speed) {
    canvas3d?.reset_camera_position(camera_position, zoom_speed, pan_speed);
    current_settings = { camera_position, zoom_speed, pan_speed };
  }
}
let resolved_url;
</script>

<BlockLabel
	{show_label}
	Icon={File}
	label={label || i18n("3D_model.3d_model")}
/>
{#if value}
	<div class="model3D">
		<IconButtonWrapper>
			{#if !use_3dgs}
				<!-- Canvas3DGS doesn't implement the undo method (reset_camera_position) -->
				<IconButton
					Icon={Undo}
					label="Undo"
					on:click={() => handle_undo()}
					disabled={!has_change_history}
				/>
			{/if}
			<a
				href={resolved_url}
				target={window.__is_colab__ ? "_blank" : null}
				download={window.__is_colab__ ? null : value.orig_name || value.path}
			>
				<IconButton Icon={Download} label={i18n("common.download")} />
			</a>
		</IconButtonWrapper>

		{#if use_3dgs}
			<svelte:component
				this={Canvas3DGSComponent}
				bind:resolved_url
				{value}
				{zoom_speed}
				{pan_speed}
			/>
		{:else}
			<svelte:component
				this={Canvas3DComponent}
				bind:this={canvas3d}
				bind:resolved_url
				{value}
				{display_mode}
				{clear_color}
				{camera_position}
				{zoom_speed}
				{pan_speed}
			/>
		{/if}
	</div>
{/if}

<style>
	.model3D {
		display: flex;
		position: relative;
		width: var(--size-full);
		height: var(--size-full);
		border-radius: var(--block-radius);
		overflow: hidden;
	}
	.model3D :global(canvas) {
		width: var(--size-full);
		height: var(--size-full);
		object-fit: contain;
		overflow: hidden;
	}
</style>
