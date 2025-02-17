<script>import { BlockTitle } from "@gradio/atoms";
import { Block } from "@gradio/atoms";
import { StatusTracker } from "@gradio/statustracker";
import { onMount } from "svelte";
import { LineChart as LabelIcon } from "@gradio/icons";
import { Empty } from "@gradio/atoms";
export let value;
export let x;
export let y;
export let color = null;
export let root;
$:
  unique_colors = color && value && value.datatypes[color] === "nominal" ? Array.from(new Set(_data.map((d) => d[color]))) : [];
export let title = null;
export let x_title = null;
export let y_title = null;
export let color_title = null;
export let x_bin = null;
export let y_aggregate = void 0;
export let color_map = null;
export let x_lim = null;
export let y_lim = null;
export let x_label_angle = null;
export let y_label_angle = null;
export let x_axis_labels_visible = true;
export let caption = null;
export let sort = null;
export let tooltip = "axis";
function reformat_sort(_sort2) {
  if (_sort2 === "x") {
    return "ascending";
  } else if (_sort2 === "-x") {
    return "descending";
  } else if (_sort2 === "y") {
    return { field: y, order: "ascending" };
  } else if (_sort2 === "-y") {
    return { field: y, order: "descending" };
  } else if (_sort2 === null) {
    return void 0;
  } else if (Array.isArray(_sort2)) {
    return _sort2;
  }
}
$:
  _sort = reformat_sort(sort);
export let _selectable = false;
let _data;
export let gradio;
$:
  x_temporal = value && value.datatypes[x] === "temporal";
$:
  _x_lim = x_lim && x_temporal ? [x_lim[0] * 1e3, x_lim[1] * 1e3] : x_lim;
let _x_bin;
let mouse_down_on_chart = false;
const SUFFIX_DURATION = {
  s: 1,
  m: 60,
  h: 60 * 60,
  d: 24 * 60 * 60
};
$:
  _x_bin = x_bin ? typeof x_bin === "string" ? 1e3 * parseInt(x_bin.substring(0, x_bin.length - 1)) * SUFFIX_DURATION[x_bin[x_bin.length - 1]] : x_bin : void 0;
let _y_aggregate;
let aggregating;
$: {
  if (value) {
    if (value.mark === "point") {
      aggregating = _x_bin !== void 0;
      _y_aggregate = y_aggregate || aggregating ? "sum" : void 0;
    } else {
      aggregating = _x_bin !== void 0 || value.datatypes[x] === "nominal";
      _y_aggregate = y_aggregate ? y_aggregate : "sum";
    }
  }
}
function reformat_data(data) {
  if (tooltip == "all" || Array.isArray(tooltip)) {
    return data.data.map((row) => {
      const obj = {};
      data.columns.forEach((col, i) => {
        obj[col] = row[i];
      });
      return obj;
    });
  }
  let x_index = data.columns.indexOf(x);
  let y_index = data.columns.indexOf(y);
  let color_index = color ? data.columns.indexOf(color) : null;
  return data.data.map((row) => {
    const obj = {
      [x]: row[x_index],
      [y]: row[y_index]
    };
    if (color && color_index !== null) {
      obj[color] = row[color_index];
    }
    return obj;
  });
}
$:
  _data = value ? reformat_data(value) : [];
const is_browser = typeof window !== "undefined";
let chart_element;
$:
  computed_style = chart_element ? window.getComputedStyle(chart_element) : null;
let view;
let mounted = false;
let old_width;
let resizeObserver;
let vegaEmbed;
async function load_chart() {
  if (view) {
    view.finalize();
  }
  if (!value || !chart_element)
    return;
  old_width = chart_element.offsetWidth;
  const spec = create_vega_lite_spec();
  if (!spec)
    return;
  resizeObserver = new ResizeObserver((el) => {
    if (!el[0].target || !(el[0].target instanceof HTMLElement))
      return;
    if (old_width === 0 && chart_element.offsetWidth !== 0 && value.datatypes[x] === "nominal") {
      load_chart();
    } else {
      view.signal("width", el[0].target.offsetWidth).run();
    }
  });
  if (!vegaEmbed) {
    vegaEmbed = (await import("vega-embed")).default;
  }
  vegaEmbed(chart_element, spec, { actions: false }).then(function(result) {
    view = result.view;
    resizeObserver.observe(chart_element);
    var debounceTimeout;
    view.addEventListener("dblclick", () => {
      gradio.dispatch("double_click");
    });
    chart_element.addEventListener(
      "mousedown",
      function(e) {
        if (e.detail > 1) {
          e.preventDefault();
        }
      },
      false
    );
    if (_selectable) {
      view.addSignalListener("brush", function(_, value2) {
        if (Object.keys(value2).length === 0)
          return;
        clearTimeout(debounceTimeout);
        let range = value2[Object.keys(value2)[0]];
        if (x_temporal) {
          range = [range[0] / 1e3, range[1] / 1e3];
        }
        let callback = () => {
          gradio.dispatch("select", {
            value: range,
            index: range,
            selected: true
          });
        };
        if (mouse_down_on_chart) {
          release_callback = callback;
        } else {
          debounceTimeout = setTimeout(function() {
            gradio.dispatch("select", {
              value: range,
              index: range,
              selected: true
            });
          }, 250);
        }
      });
    }
  });
}
let release_callback = null;
onMount(() => {
  mounted = true;
  chart_element.addEventListener("mousedown", () => {
    mouse_down_on_chart = true;
  });
  chart_element.addEventListener("mouseup", () => {
    mouse_down_on_chart = false;
    if (release_callback) {
      release_callback();
      release_callback = null;
    }
  });
  return () => {
    mounted = false;
    if (view) {
      view.finalize();
    }
    if (resizeObserver) {
      resizeObserver.disconnect();
    }
  };
});
$:
  title, x_title, y_title, color_title, x, y, color, x_bin, _y_aggregate, color_map, x_lim, y_lim, caption, sort, value, mounted, chart_element, computed_style && requestAnimationFrame(load_chart);
function create_vega_lite_spec() {
  if (!value || !computed_style)
    return null;
  let accent_color = computed_style.getPropertyValue("--color-accent");
  let body_text_color = computed_style.getPropertyValue("--body-text-color");
  let borderColorPrimary = computed_style.getPropertyValue(
    "--border-color-primary"
  );
  let font_family = computed_style.fontFamily;
  let title_weight = computed_style.getPropertyValue(
    "--block-title-text-weight"
  );
  const font_to_px_val = (font) => {
    return font.endsWith("px") ? parseFloat(font.slice(0, -2)) : 12;
  };
  let text_size_md = font_to_px_val(
    computed_style.getPropertyValue("--text-md")
  );
  let text_size_sm = font_to_px_val(
    computed_style.getPropertyValue("--text-sm")
  );
  return {
    $schema: "https://vega.github.io/schema/vega-lite/v5.17.0.json",
    background: "transparent",
    config: {
      autosize: { type: "fit", contains: "padding" },
      axis: {
        labelFont: font_family,
        labelColor: body_text_color,
        titleFont: font_family,
        titleColor: body_text_color,
        titlePadding: 8,
        tickColor: borderColorPrimary,
        labelFontSize: text_size_sm,
        gridColor: borderColorPrimary,
        titleFontWeight: "normal",
        titleFontSize: text_size_sm,
        labelFontWeight: "normal",
        domain: false,
        labelAngle: 0
      },
      legend: {
        labelColor: body_text_color,
        labelFont: font_family,
        titleColor: body_text_color,
        titleFont: font_family,
        titleFontWeight: "normal",
        titleFontSize: text_size_sm,
        labelFontWeight: "normal",
        offset: 2
      },
      title: {
        color: body_text_color,
        font: font_family,
        fontSize: text_size_md,
        fontWeight: title_weight,
        anchor: "middle"
      },
      view: { stroke: borderColorPrimary },
      mark: {
        stroke: value.mark !== "bar" ? accent_color : void 0,
        fill: value.mark === "bar" ? accent_color : void 0,
        cursor: "crosshair"
      }
    },
    data: { name: "data" },
    datasets: {
      data: _data
    },
    layer: ["plot", ...value.mark === "line" ? ["hover"] : []].map(
      (mode) => {
        return {
          encoding: {
            size: value.mark === "line" ? mode == "plot" ? {
              condition: {
                empty: false,
                param: "hoverPlot",
                value: 3
              },
              value: 2
            } : {
              condition: { empty: false, param: "hover", value: 100 },
              value: 0
            } : void 0,
            opacity: mode === "plot" ? void 0 : {
              condition: { empty: false, param: "hover", value: 1 },
              value: 0
            },
            x: {
              axis: {
                ...x_label_angle !== null && { labelAngle: x_label_angle },
                labels: x_axis_labels_visible,
                ticks: x_axis_labels_visible
              },
              field: x,
              title: x_title || x,
              type: value.datatypes[x],
              scale: _x_lim ? { domain: _x_lim } : void 0,
              bin: _x_bin ? { step: _x_bin } : void 0,
              sort: _sort
            },
            y: {
              axis: y_label_angle ? { labelAngle: y_label_angle } : {},
              field: y,
              title: y_title || y,
              type: value.datatypes[y],
              scale: y_lim ? { domain: y_lim } : void 0,
              aggregate: aggregating ? _y_aggregate : void 0
            },
            color: color ? {
              field: color,
              legend: { orient: "bottom", title: color_title },
              scale: value.datatypes[color] === "nominal" ? {
                domain: unique_colors,
                range: color_map ? unique_colors.map((c) => color_map[c]) : void 0
              } : {
                range: [
                  100,
                  200,
                  300,
                  400,
                  500,
                  600,
                  700,
                  800,
                  900
                ].map(
                  (n) => computed_style.getPropertyValue("--primary-" + n)
                ),
                interpolate: "hsl"
              },
              type: value.datatypes[color]
            } : void 0,
            tooltip: tooltip == "none" ? void 0 : [
              {
                field: y,
                type: value.datatypes[y],
                aggregate: aggregating ? _y_aggregate : void 0,
                title: y_title || y
              },
              {
                field: x,
                type: value.datatypes[x],
                title: x_title || x,
                format: x_temporal ? "%Y-%m-%d %H:%M:%S" : void 0,
                bin: _x_bin ? { step: _x_bin } : void 0
              },
              ...color ? [
                {
                  field: color,
                  type: value.datatypes[color]
                }
              ] : [],
              ...tooltip === "axis" ? [] : value?.columns.filter(
                (col) => col !== x && col !== y && col !== color && (tooltip === "all" || tooltip.includes(col))
              ).map((column) => ({
                field: column,
                type: value.datatypes[column]
              }))
            ]
          },
          strokeDash: {},
          mark: { clip: true, type: mode === "hover" ? "point" : value.mark },
          name: mode
        };
      }
    ),
    // @ts-ignore
    params: [
      ...value.mark === "line" ? [
        {
          name: "hoverPlot",
          select: {
            clear: "mouseout",
            fields: color ? [color] : [],
            nearest: true,
            on: "mouseover",
            type: "point"
          },
          views: ["hover"]
        },
        {
          name: "hover",
          select: {
            clear: "mouseout",
            nearest: true,
            on: "mouseover",
            type: "point"
          },
          views: ["hover"]
        }
      ] : [],
      ..._selectable ? [
        {
          name: "brush",
          select: {
            encodings: ["x"],
            mark: { fill: "gray", fillOpacity: 0.3, stroke: "none" },
            type: "interval"
          },
          views: ["plot"]
        }
      ] : []
    ],
    width: chart_element.offsetWidth,
    title: title || void 0
  };
}
export let label = "Textbox";
export let elem_id = "";
export let elem_classes = [];
export let visible = true;
export let show_label;
export let scale = null;
export let min_width = void 0;
export let loading_status = void 0;
export let height = void 0;
</script>

<Block
	{visible}
	{elem_id}
	{elem_classes}
	{scale}
	{min_width}
	allow_overflow={false}
	padding={true}
	{height}
>
	{#if loading_status}
		<StatusTracker
			autoscroll={gradio.autoscroll}
			i18n={gradio.i18n}
			{...loading_status}
			on:clear_status={() => gradio.dispatch("clear_status", loading_status)}
		/>
	{/if}
	<BlockTitle {root} {show_label} info={undefined}>{label}</BlockTitle>
	{#if value && is_browser}
		<div bind:this={chart_element}></div>

		{#if caption}
			<p class="caption">{caption}</p>
		{/if}
	{:else}
		<Empty unpadded_box={true}><LabelIcon /></Empty>
	{/if}
</Block>

<style>
	div {
		width: 100%;
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
	.caption {
		padding: 0 4px;
		margin: 0;
		text-align: center;
	}
</style>
