<script context="module">import {} from "tinycolor2";
</script>

<script>import tinycolor from "tinycolor2";
import { clamp } from "../utils/pixi";
import { getContext, onMount, tick } from "svelte";
import { TOOL_KEY } from "./Tools.svelte";
import { EDITOR_KEY } from "../ImageEditor.svelte";
import { draw_path } from "./brush";
import BrushOptions from "./BrushOptions.svelte";
export let default_size;
export let default_color = void 0;
export let colors = void 0;
export let color_mode = void 0;
export let mode;
let processed_colors = [];
let old_colors = [];
if (colors && JSON.stringify(old_colors) !== JSON.stringify(colors)) {
  processed_colors = colors.map(process_color).filter((_, i) => i < 4);
  old_colors = processed_colors;
}
let selected_color = process_color(default_color || "#000000");
$:
  if (mode === "draw") {
    current_color.set(selected_color);
  }
let brush_options = false;
const {
  pixi,
  dimensions,
  current_layer,
  command_manager,
  register_context,
  editor_box,
  crop,
  toolbar_box
} = getContext(EDITOR_KEY);
const { active_tool, register_tool, current_color } = getContext(TOOL_KEY);
let drawing = false;
let draw;
function generate_sizes(x, y) {
  const min = clamp(Math.min(x, y), 500, 1e3);
  return Math.round(min * 2 / 100);
}
$:
  mode === "draw" && current_color.set(selected_color);
let selected_size = default_size === "auto" ? generate_sizes(...$dimensions) : default_size;
function pointer_down_handler(event) {
  if ($active_tool !== mode) {
    return;
  }
  drawing = true;
  if (!$pixi || !$current_layer) {
    return;
  }
  draw = draw_path(
    $pixi.renderer,
    $pixi.layer_container,
    $current_layer,
    mode
  );
  draw.start({
    x: event.screen.x,
    y: event.screen.y,
    color: selected_color || void 0,
    size: selected_size,
    opacity: 1
  });
}
function pointer_up_handler(event) {
  if (!$pixi || !$current_layer) {
    return;
  }
  if ($active_tool !== mode) {
    return;
  }
  draw.stop();
  command_manager.execute(draw);
  drawing = false;
}
function pointer_move_handler(event) {
  if ($active_tool !== mode) {
    return;
  }
  if (drawing) {
    draw.continue({
      x: event.screen.x,
      y: event.screen.y
    });
  }
  const x_bound = $crop[0] * $dimensions[0];
  const y_bound = $crop[1] * $dimensions[1];
  if (x_bound > event.screen.x || y_bound > event.screen.y || event.screen.x > x_bound + $crop[2] * $dimensions[0] || event.screen.y > y_bound + $crop[3] * $dimensions[1]) {
    brush_cursor = false;
    document.body.style.cursor = "auto";
  } else {
    brush_cursor = true;
    document.body.style.cursor = "none";
  }
  if (brush_cursor) {
    pos = {
      x: event.clientX - $editor_box.child_left,
      y: event.clientY - $editor_box.child_top
    };
  }
}
let brush_cursor = false;
async function toggle_listeners(on_off) {
  $pixi?.layer_container[on_off]("pointerdown", pointer_down_handler);
  $pixi?.layer_container[on_off]("pointerup", pointer_up_handler);
  $pixi?.layer_container[on_off]("pointermove", pointer_move_handler);
  $pixi?.layer_container[on_off](
    "pointerenter",
    (event) => {
      if ($active_tool === mode) {
        brush_cursor = true;
        document.body.style.cursor = "none";
      }
    }
  );
  $pixi?.layer_container[on_off](
    "pointerleave",
    () => (brush_cursor = false, document.body.style.cursor = "auto")
  );
}
register_context(mode, {
  init_fn: () => {
    toggle_listeners("on");
  },
  reset_fn: () => {
    toggle_listeners("off");
  }
});
const toggle_options = debounce_toggle();
const unregister = register_tool(mode, {
  cb: toggle_options
});
onMount(() => {
  return () => {
    unregister();
    toggle_listeners("off");
  };
});
let recent_colors = [null, null, null];
function process_color(color) {
  return tinycolor(color).toRgbString();
}
let pos = { x: 0, y: 0 };
$:
  brush_size = selected_size / $dimensions[0] * $editor_box.child_width * 2;
function debounce_toggle() {
  let timeout = null;
  return function executedFunction(should_close) {
    const later = () => {
      if (timeout) {
        clearTimeout(timeout);
      }
      if (should_close !== void 0) {
        brush_options = should_close;
        return;
      }
      brush_options = !brush_options;
    };
    if (timeout) {
      clearTimeout(timeout);
    }
    timeout = setTimeout(later, 100);
  };
}
</script>

<svelte:window
	on:keydown={({ key }) => key === "Escape" && toggle_options(false)}
/>

<span
	style:transform="translate({pos.x}px, {pos.y}px)"
	style:top="{$editor_box.child_top -
		$editor_box.parent_top -
		brush_size / 2}px"
	style:left="{$editor_box.child_left -
		$editor_box.parent_left -
		brush_size / 2}px"
	style:width="{brush_size}px"
	style:height="{brush_size}px"
	style:opacity={brush_cursor ? 1 : 0}
/>

{#if brush_options}
	<div>
		<BrushOptions
			show_swatch={mode === "draw"}
			on:click_outside={() => toggle_options()}
			colors={processed_colors}
			bind:selected_color
			{color_mode}
			bind:recent_colors
			bind:selected_size
			dimensions={$dimensions}
			parent_width={$editor_box.parent_width}
			parent_height={$editor_box.parent_height}
			parent_left={$editor_box.parent_left}
			toolbar_box={$toolbar_box}
		/>
	</div>
{/if}

<style>
	span {
		position: absolute;
		background: rgba(0, 0, 0, 0.5);
		pointer-events: none;
		border-radius: 50%;
	}
</style>
