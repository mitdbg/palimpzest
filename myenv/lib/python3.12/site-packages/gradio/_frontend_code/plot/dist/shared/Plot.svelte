<script>import { Plot as PlotIcon } from "@gradio/icons";
import { Empty } from "@gradio/atoms";
export let value;
let _value;
export let colors = [];
export let show_label;
export let theme_mode;
export let caption;
export let bokeh_version;
export let show_actions_button;
export let gradio;
export let x_lim = null;
export let _selectable;
let PlotComponent = null;
let _type = value?.type;
const plotTypeMapping = {
  plotly: () => import("./plot_types/PlotlyPlot.svelte"),
  bokeh: () => import("./plot_types/BokehPlot.svelte"),
  altair: () => import("./plot_types/AltairPlot.svelte"),
  matplotlib: () => import("./plot_types/MatplotlibPlot.svelte")
};
let loadedPlotTypeMapping = {};
const is_browser = typeof window !== "undefined";
let key = 0;
$:
  if (value !== _value) {
    key += 1;
    let type = value?.type;
    if (type !== _type) {
      PlotComponent = null;
    }
    if (type && type in plotTypeMapping && is_browser) {
      if (loadedPlotTypeMapping[type]) {
        PlotComponent = loadedPlotTypeMapping[type];
      } else {
        plotTypeMapping[type]().then((module) => {
          PlotComponent = module.default;
          loadedPlotTypeMapping[type] = PlotComponent;
        });
      }
    }
    _value = value;
    _type = type;
  }
</script>

{#if value && PlotComponent}
	{#key key}
		<svelte:component
			this={PlotComponent}
			{value}
			{colors}
			{theme_mode}
			{show_label}
			{caption}
			{bokeh_version}
			{show_actions_button}
			{gradio}
			{_selectable}
			{x_lim}
			on:load
			on:select
		/>
	{/key}
{:else}
	<Empty unpadded_box={true} size="large"><PlotIcon /></Empty>
{/if}
