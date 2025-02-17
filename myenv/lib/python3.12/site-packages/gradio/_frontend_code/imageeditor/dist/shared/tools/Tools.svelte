<script context="module">export const TOOL_KEY = Symbol("tool");
</script>

<script>import { Toolbar } from "@gradio/atoms";
import { default as IconButton } from "./IconButton.svelte";
import { getContext, setContext } from "svelte";
import { writable } from "svelte/store";
import { EDITOR_KEY } from "../ImageEditor.svelte";
import { Image, Crop, Brush, Erase } from "@gradio/icons";
import {} from "@gradio/utils";
const { active_tool, toolbar_box, editor_box } = getContext(EDITOR_KEY);
export let i18n;
let tools = [];
const cbs = {};
let current_color = writable("#000000");
const tool_context = {
  current_color,
  register_tool: (type, opts) => {
    tools = [...tools, type];
    if (opts?.cb) {
      cbs[type] = opts.cb;
    }
    return () => {
      tools = tools.filter((tool) => tool !== type);
    };
  },
  active_tool: {
    subscribe: active_tool.subscribe,
    set: active_tool.set
  }
};
setContext(TOOL_KEY, tool_context);
const tools_meta = {
  crop: {
    order: 1,
    label: i18n("Transform"),
    icon: Crop
  },
  draw: {
    order: 2,
    label: i18n("Draw"),
    icon: Brush
  },
  erase: {
    order: 2,
    label: i18n("Erase"),
    icon: Erase
  },
  bg: {
    order: 0,
    label: i18n("Background"),
    icon: Image
  }
};
let toolbar_width;
let toolbar_wrap;
$:
  toolbar_width, $editor_box, get_dimensions();
function get_dimensions() {
  if (!toolbar_wrap)
    return;
  $toolbar_box = toolbar_wrap.getBoundingClientRect();
}
function handle_click(e, tool) {
  e.stopPropagation();
  $active_tool = tool;
  cbs[tool] && cbs[tool]();
}
</script>

<slot />

<div
	class="toolbar-wrap"
	bind:clientWidth={toolbar_width}
	bind:this={toolbar_wrap}
>
	<Toolbar show_border={false}>
		{#each tools as tool (tool)}
			<IconButton
				highlight={$active_tool === tool}
				on:click={(e) => handle_click(e, tool)}
				Icon={tools_meta[tool].icon}
				size="medium"
				padded={false}
				label={tools_meta[tool].label + " button"}
				transparent={true}
				offset={tool === "draw" ? -2 : tool === "erase" ? -6 : 0}
			/>
		{/each}
	</Toolbar>
</div>

<style>
	.toolbar-wrap {
		display: flex;
		justify-content: center;
		align-items: center;
		margin-left: var(--spacing-xl);
		height: 100%;
	}
</style>
