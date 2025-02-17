<script>import Plotly from "plotly.js-dist-min";
import { afterUpdate, createEventDispatcher } from "svelte";
export let value;
export let show_label;
$:
  plot = value?.plot;
let plot_div;
let plotly_global_style;
const dispatch = createEventDispatcher();
function load_plotly_css() {
  if (!plotly_global_style) {
    plotly_global_style = document.getElementById("plotly.js-style-global");
    const plotly_style_clone = plotly_global_style.cloneNode();
    plot_div.appendChild(plotly_style_clone);
    for (const rule of plotly_global_style.sheet.cssRules) {
      plotly_style_clone.sheet.insertRule(rule.cssText);
    }
  }
}
afterUpdate(async () => {
  load_plotly_css();
  let plotObj = JSON.parse(plot);
  plotObj.config = plotObj.config || {};
  plotObj.config.responsive = true;
  plotObj.responsive = true;
  plotObj.layout.autosize = true;
  if (plotObj.layout.margin == void 0) {
    plotObj.layout.margin = {};
  }
  if (plotObj.layout.title && show_label) {
    plotObj.layout.margin.t = Math.max(100, plotObj.layout.margin.t || 0);
  }
  plotObj.layout.margin.autoexpand = true;
  Plotly.react(plot_div, plotObj.data, plotObj.layout, plotObj.config);
  Plotly.Plots.resize(plot_div);
  plot_div.on("plotly_afterplot", () => {
    dispatch("load");
  });
});
</script>

<div data-testid={"plotly"} bind:this={plot_div} />
