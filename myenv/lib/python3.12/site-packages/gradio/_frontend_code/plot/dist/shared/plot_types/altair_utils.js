import { colors as color_palette } from "@gradio/theme";
import { get_next_color } from "@gradio/utils";
export function set_config(spec, computed_style, chart_type, colors) {
    let accentColor = computed_style.getPropertyValue("--color-accent");
    let bodyTextColor = computed_style.getPropertyValue("--body-text-color");
    let borderColorPrimary = computed_style.getPropertyValue("--border-color-primary");
    let fontFamily = computed_style.fontFamily;
    let titleWeight = computed_style.getPropertyValue("--block-title-text-weight");
    const fontToPxVal = (font) => {
        return font.endsWith("px") ? parseFloat(font.slice(0, -2)) : 12;
    };
    let textSizeMd = fontToPxVal(computed_style.getPropertyValue("--text-md"));
    let textSizeSm = fontToPxVal(computed_style.getPropertyValue("--text-sm"));
    let config = {
        autosize: { type: "fit", contains: "padding" },
        axis: {
            labelFont: fontFamily,
            labelColor: bodyTextColor,
            titleFont: fontFamily,
            titleColor: bodyTextColor,
            tickColor: borderColorPrimary,
            labelFontSize: textSizeSm,
            gridColor: borderColorPrimary,
            titleFontWeight: "normal",
            titleFontSize: textSizeSm,
            labelFontWeight: "normal",
            domain: false,
            labelAngle: 0
        },
        legend: {
            labelColor: bodyTextColor,
            labelFont: fontFamily,
            titleColor: bodyTextColor,
            titleFont: fontFamily,
            titleFontWeight: "normal",
            titleFontSize: textSizeSm,
            labelFontWeight: "normal",
            offset: 2
        },
        title: {
            color: bodyTextColor,
            font: fontFamily,
            fontSize: textSizeMd,
            fontWeight: titleWeight,
            anchor: "middle"
        },
        view: {
            stroke: borderColorPrimary
        }
    };
    spec.config = config;
    // @ts-ignore (unsure why the following are not typed in Spec)
    let encoding = spec.encoding;
    // @ts-ignore
    let layer = spec.layer;
    switch (chart_type) {
        case "scatter":
            spec.config.mark = { stroke: accentColor };
            if (encoding.color && encoding.color.type == "nominal") {
                encoding.color.scale.range = encoding.color.scale.range.map((_, i) => get_color(colors, i));
            }
            else if (encoding.color && encoding.color.type == "quantitative") {
                encoding.color.scale.range = ["#eff6ff", "#1e3a8a"];
                encoding.color.scale.range.interpolate = "hsl";
            }
            break;
        case "line":
            spec.config.mark = { stroke: accentColor, cursor: "crosshair" };
            layer.forEach((d) => {
                if (d.encoding.color) {
                    d.encoding.color.scale.range = d.encoding.color.scale.range.map((_, i) => get_color(colors, i));
                }
            });
            break;
        case "bar":
            spec.config.mark = { opacity: 0.8, fill: accentColor };
            if (encoding.color) {
                encoding.color.scale.range = encoding.color.scale.range.map((_, i) => get_color(colors, i));
            }
            break;
    }
    return spec;
}
function get_color(colors, index) {
    let current_color = colors[index % colors.length];
    if (current_color && current_color in color_palette) {
        return color_palette[current_color]?.primary;
    }
    else if (!current_color) {
        return color_palette[get_next_color(index)]
            .primary;
    }
    return current_color;
}
