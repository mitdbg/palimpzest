<script>import { set_config } from "./altair_utils";
import { onMount, onDestroy } from "svelte";
import vegaEmbed from "vega-embed";
export let value;
export let colors = [];
export let caption;
export let show_actions_button;
export let gradio;
let element;
let parent_element;
let view;
export let _selectable;
let computed_style = window.getComputedStyle(document.body);
let old_spec;
let spec_width;
$:
  plot = value?.plot;
$:
  spec = JSON.parse(plot);
$:
  if (spec && spec.params && !_selectable) {
    spec.params = spec.params.filter((param) => param.name !== "brush");
  }
$:
  if (old_spec !== spec) {
    old_spec = spec;
    spec_width = spec.width;
  }
$:
  if (value.chart) {
    spec = set_config(spec, computed_style, value.chart, colors);
  }
$:
  fit_width_to_parent = spec.encoding?.column?.field || spec.encoding?.row?.field || value.chart === void 0 ? false : true;
const get_width = () => {
  return Math.min(
    parent_element.offsetWidth,
    spec_width || parent_element.offsetWidth
  );
};
let resize_callback = () => {
};
const renderPlot = () => {
  if (fit_width_to_parent) {
    spec.width = get_width();
  }
  vegaEmbed(element, spec, { actions: show_actions_button }).then(
    function(result) {
      view = result.view;
      resize_callback = () => {
        view.signal("width", get_width()).run();
      };
      if (!_selectable)
        return;
      const callback = (event, item) => {
        const brushValue = view.signal("brush");
        if (brushValue) {
          if (Object.keys(brushValue).length === 0) {
            gradio.dispatch("select", {
              value: null,
              index: null,
              selected: false
            });
          } else {
            const key = Object.keys(brushValue)[0];
            let range = brushValue[key].map(
              (x) => x / 1e3
            );
            gradio.dispatch("select", {
              value: brushValue,
              index: range,
              selected: true
            });
          }
        }
      };
      view.addEventListener("mouseup", callback);
      view.addEventListener("touchup", callback);
    }
  );
};
let resizeObserver = new ResizeObserver(() => {
  if (fit_width_to_parent && spec.width !== parent_element.offsetWidth) {
    resize_callback();
  }
});
onMount(() => {
  renderPlot();
  resizeObserver.observe(parent_element);
});
onDestroy(() => {
  resizeObserver.disconnect();
});
</script>

<div data-testid={"altair"} class="altair layout" bind:this={parent_element}>
	<div bind:this={element}></div>
	{#if caption}
		<div class="caption layout">
			{caption}
		</div>
	{/if}
</div>

<style>
	.altair :global(canvas) {
		padding: 6px;
	}
	.altair :global(.vega-embed) {
		padding: 0px !important;
	}
	.altair :global(.vega-actions) {
		right: 0px !important;
	}
	.layout {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		width: var(--size-full);
		height: var(--size-full);
		color: var(--body-text-color);
	}
	.altair {
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		width: var(--size-full);
		height: var(--size-full);
	}
	.caption {
		font-size: var(--text-sm);
		margin-bottom: 6px;
	}
	:global(#vg-tooltip-element) {
		font-family: var(--font) !important;
		font-size: var(--text-xs) !important;
		box-shadow: none !important;
		background-color: var(--block-background-fill) !important;
		border: 1px solid var(--border-color-primary) !important;
		color: var(--body-text-color) !important;
	}
	:global(#vg-tooltip-element .key) {
		color: var(--body-text-color-subdued) !important;
	}
</style>
