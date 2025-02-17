import { c as create_ssr_component, v as validate_component, a as createEventDispatcher, d as add_styles, e as escape, f as each, b as add_attribute } from './ssr-fyTaU2Wq.js';
import { B as Block, S as Static, f as BlockLabel, av as LineChart, h as Empty } from './2-CnaXPGyd.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BeHry4b7.js';

const css = {
  code: ".container.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus{padding:var(--block-padding)}.output-class.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus{display:flex;justify-content:center;align-items:center;padding:var(--size-6) var(--size-4);color:var(--body-text-color);font-weight:var(--weight-bold);font-size:var(--text-xxl)}.confidence-set.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:var(--size-2);color:var(--body-text-color);line-height:var(--line-none);font-family:var(--font-mono);width:100%}.confidence-set.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus:last-child{margin-bottom:0}.inner-wrap.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus{flex:1 1 0%;display:flex;flex-direction:column}.bar.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus{appearance:none;-webkit-appearance:none;-moz-appearance:none;align-self:flex-start;margin-bottom:var(--size-1);border-radius:var(--radius-md);background:var(--stat-background-fill);height:var(--size-1);border:none}.bar.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus::-moz-meter-bar{border-radius:var(--radius-md);background:var(--stat-background-fill)}.bar.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus::-webkit-meter-bar{border-radius:var(--radius-md);background:var(--stat-background-fill);border:none}.bar.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus::-webkit-meter-optimum-value,.bar.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus::-webkit-meter-suboptimum-value,.bar.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus::-webkit-meter-even-less-good-value{border-radius:var(--radius-md);background:var(--stat-background-fill)}.bar.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus::-ms-fill{border-radius:var(--radius-md);background:var(--stat-background-fill);border:none}.label.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus{display:flex;align-items:baseline}.label.svelte-1mutzus>.svelte-1mutzus+.svelte-1mutzus{margin-left:var(--size-2)}.confidence-set.svelte-1mutzus:hover .label.svelte-1mutzus.svelte-1mutzus{color:var(--color-accent)}.confidence-set.svelte-1mutzus:focus .label.svelte-1mutzus.svelte-1mutzus{color:var(--color-accent)}.text.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus{line-height:var(--line-md);text-align:left}.line.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus{flex:1 1 0%;border:1px dashed var(--border-color-primary);padding-right:var(--size-4);padding-left:var(--size-4)}.confidence.svelte-1mutzus.svelte-1mutzus.svelte-1mutzus{margin-left:auto;text-align:right}",
  map: '{"version":3,"file":"Label.svelte","sources":["Label.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { createEventDispatcher } from \\"svelte\\";\\nexport let value;\\nconst dispatch = createEventDispatcher();\\nexport let color = void 0;\\nexport let selectable = false;\\nexport let show_heading = true;\\nfunction get_aria_referenceable_id(elem_id) {\\n    return elem_id.replace(/\\\\s/g, \\"-\\");\\n}\\n<\/script>\\n\\n<div class=\\"container\\">\\n\\t{#if show_heading || !value.confidences}\\n\\t\\t<h2\\n\\t\\t\\tclass=\\"output-class\\"\\n\\t\\t\\tdata-testid=\\"label-output-value\\"\\n\\t\\t\\tclass:no-confidence={!(\\"confidences\\" in value)}\\n\\t\\t\\tstyle:background-color={color || \\"transparent\\"}\\n\\t\\t>\\n\\t\\t\\t{value.label}\\n\\t\\t</h2>\\n\\t{/if}\\n\\n\\t{#if typeof value === \\"object\\" && value.confidences}\\n\\t\\t{#each value.confidences as confidence_set, i}\\n\\t\\t\\t<button\\n\\t\\t\\t\\tclass=\\"confidence-set group\\"\\n\\t\\t\\t\\tdata-testid={`${confidence_set.label}-confidence-set`}\\n\\t\\t\\t\\tclass:selectable\\n\\t\\t\\t\\ton:click={() => {\\n\\t\\t\\t\\t\\tdispatch(\\"select\\", { index: i, value: confidence_set.label });\\n\\t\\t\\t\\t}}\\n\\t\\t\\t>\\n\\t\\t\\t\\t<div class=\\"inner-wrap\\">\\n\\t\\t\\t\\t\\t<meter\\n\\t\\t\\t\\t\\t\\taria-labelledby={get_aria_referenceable_id(\\n\\t\\t\\t\\t\\t\\t\\t`meter-text-${confidence_set.label}`\\n\\t\\t\\t\\t\\t\\t)}\\n\\t\\t\\t\\t\\t\\taria-label={confidence_set.label}\\n\\t\\t\\t\\t\\t\\taria-valuenow={Math.round(confidence_set.confidence * 100)}\\n\\t\\t\\t\\t\\t\\taria-valuemin=\\"0\\"\\n\\t\\t\\t\\t\\t\\taria-valuemax=\\"100\\"\\n\\t\\t\\t\\t\\t\\tclass=\\"bar\\"\\n\\t\\t\\t\\t\\t\\tmin=\\"0\\"\\n\\t\\t\\t\\t\\t\\tmax=\\"1\\"\\n\\t\\t\\t\\t\\t\\tvalue={confidence_set.confidence}\\n\\t\\t\\t\\t\\t\\tstyle=\\"width: {confidence_set.confidence *\\n\\t\\t\\t\\t\\t\\t\\t100}%; background: var(--stat-background-fill);\\n\\t\\t\\t\\t\\t\\t\\"\\n\\t\\t\\t\\t\\t/>\\n\\n\\t\\t\\t\\t\\t<dl class=\\"label\\">\\n\\t\\t\\t\\t\\t\\t<dt\\n\\t\\t\\t\\t\\t\\t\\tid={get_aria_referenceable_id(\\n\\t\\t\\t\\t\\t\\t\\t\\t`meter-text-${confidence_set.label}`\\n\\t\\t\\t\\t\\t\\t\\t)}\\n\\t\\t\\t\\t\\t\\t\\tclass=\\"text\\"\\n\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t{confidence_set.label}\\n\\t\\t\\t\\t\\t\\t</dt>\\n\\t\\t\\t\\t\\t\\t<div class=\\"line\\" />\\n\\t\\t\\t\\t\\t\\t<dd class=\\"confidence\\">\\n\\t\\t\\t\\t\\t\\t\\t{Math.round(confidence_set.confidence * 100)}%\\n\\t\\t\\t\\t\\t\\t</dd>\\n\\t\\t\\t\\t\\t</dl>\\n\\t\\t\\t\\t</div>\\n\\t\\t\\t</button>\\n\\t\\t{/each}\\n\\t{/if}\\n</div>\\n\\n<style>\\n\\t.container {\\n\\t\\tpadding: var(--block-padding);\\n\\t}\\n\\t.output-class {\\n\\t\\tdisplay: flex;\\n\\t\\tjustify-content: center;\\n\\t\\talign-items: center;\\n\\t\\tpadding: var(--size-6) var(--size-4);\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tfont-weight: var(--weight-bold);\\n\\t\\tfont-size: var(--text-xxl);\\n\\t}\\n\\n\\t.confidence-set {\\n\\t\\tdisplay: flex;\\n\\t\\tjustify-content: space-between;\\n\\t\\talign-items: flex-start;\\n\\t\\tmargin-bottom: var(--size-2);\\n\\t\\tcolor: var(--body-text-color);\\n\\t\\tline-height: var(--line-none);\\n\\t\\tfont-family: var(--font-mono);\\n\\t\\twidth: 100%;\\n\\t}\\n\\n\\t.confidence-set:last-child {\\n\\t\\tmargin-bottom: 0;\\n\\t}\\n\\n\\t.inner-wrap {\\n\\t\\tflex: 1 1 0%;\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t}\\n\\n\\t.bar {\\n\\t\\tappearance: none;\\n\\t\\t-webkit-appearance: none;\\n\\t\\t-moz-appearance: none;\\n\\t\\talign-self: flex-start;\\n\\t\\tmargin-bottom: var(--size-1);\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\tbackground: var(--stat-background-fill);\\n\\t\\theight: var(--size-1);\\n\\t\\tborder: none;\\n\\t}\\n\\n\\t.bar::-moz-meter-bar {\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\tbackground: var(--stat-background-fill);\\n\\t}\\n\\n\\t.bar::-webkit-meter-bar {\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\tbackground: var(--stat-background-fill);\\n\\t\\tborder: none;\\n\\t}\\n\\n\\t.bar::-webkit-meter-optimum-value,\\n\\t.bar::-webkit-meter-suboptimum-value,\\n\\t.bar::-webkit-meter-even-less-good-value {\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\tbackground: var(--stat-background-fill);\\n\\t}\\n\\n\\t.bar::-ms-fill {\\n\\t\\tborder-radius: var(--radius-md);\\n\\t\\tbackground: var(--stat-background-fill);\\n\\t\\tborder: none;\\n\\t}\\n\\n\\t.label {\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: baseline;\\n\\t}\\n\\n\\t.label > * + * {\\n\\t\\tmargin-left: var(--size-2);\\n\\t}\\n\\n\\t.confidence-set:hover .label {\\n\\t\\tcolor: var(--color-accent);\\n\\t}\\n\\n\\t.confidence-set:focus .label {\\n\\t\\tcolor: var(--color-accent);\\n\\t}\\n\\n\\t.text {\\n\\t\\tline-height: var(--line-md);\\n\\t\\ttext-align: left;\\n\\t}\\n\\n\\t.line {\\n\\t\\tflex: 1 1 0%;\\n\\t\\tborder: 1px dashed var(--border-color-primary);\\n\\t\\tpadding-right: var(--size-4);\\n\\t\\tpadding-left: var(--size-4);\\n\\t}\\n\\n\\t.confidence {\\n\\t\\tmargin-left: auto;\\n\\t\\ttext-align: right;\\n\\t}</style>\\n"],"names":[],"mappings":"AAwEC,uDAAW,CACV,OAAO,CAAE,IAAI,eAAe,CAC7B,CACA,0DAAc,CACb,OAAO,CAAE,IAAI,CACb,eAAe,CAAE,MAAM,CACvB,WAAW,CAAE,MAAM,CACnB,OAAO,CAAE,IAAI,QAAQ,CAAC,CAAC,IAAI,QAAQ,CAAC,CACpC,KAAK,CAAE,IAAI,iBAAiB,CAAC,CAC7B,WAAW,CAAE,IAAI,aAAa,CAAC,CAC/B,SAAS,CAAE,IAAI,UAAU,CAC1B,CAEA,4DAAgB,CACf,OAAO,CAAE,IAAI,CACb,eAAe,CAAE,aAAa,CAC9B,WAAW,CAAE,UAAU,CACvB,aAAa,CAAE,IAAI,QAAQ,CAAC,CAC5B,KAAK,CAAE,IAAI,iBAAiB,CAAC,CAC7B,WAAW,CAAE,IAAI,WAAW,CAAC,CAC7B,WAAW,CAAE,IAAI,WAAW,CAAC,CAC7B,KAAK,CAAE,IACR,CAEA,4DAAe,WAAY,CAC1B,aAAa,CAAE,CAChB,CAEA,wDAAY,CACX,IAAI,CAAE,CAAC,CAAC,CAAC,CAAC,EAAE,CACZ,OAAO,CAAE,IAAI,CACb,cAAc,CAAE,MACjB,CAEA,iDAAK,CACJ,UAAU,CAAE,IAAI,CAChB,kBAAkB,CAAE,IAAI,CACxB,eAAe,CAAE,IAAI,CACrB,UAAU,CAAE,UAAU,CACtB,aAAa,CAAE,IAAI,QAAQ,CAAC,CAC5B,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,UAAU,CAAE,IAAI,sBAAsB,CAAC,CACvC,MAAM,CAAE,IAAI,QAAQ,CAAC,CACrB,MAAM,CAAE,IACT,CAEA,iDAAI,gBAAiB,CACpB,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,UAAU,CAAE,IAAI,sBAAsB,CACvC,CAEA,iDAAI,mBAAoB,CACvB,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,UAAU,CAAE,IAAI,sBAAsB,CAAC,CACvC,MAAM,CAAE,IACT,CAEA,iDAAI,6BAA6B,CACjC,iDAAI,gCAAgC,CACpC,iDAAI,oCAAqC,CACxC,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,UAAU,CAAE,IAAI,sBAAsB,CACvC,CAEA,iDAAI,UAAW,CACd,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,UAAU,CAAE,IAAI,sBAAsB,CAAC,CACvC,MAAM,CAAE,IACT,CAEA,mDAAO,CACN,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,QACd,CAEA,qBAAM,CAAG,eAAC,CAAG,eAAE,CACd,WAAW,CAAE,IAAI,QAAQ,CAC1B,CAEA,8BAAe,MAAM,CAAC,oCAAO,CAC5B,KAAK,CAAE,IAAI,cAAc,CAC1B,CAEA,8BAAe,MAAM,CAAC,oCAAO,CAC5B,KAAK,CAAE,IAAI,cAAc,CAC1B,CAEA,kDAAM,CACL,WAAW,CAAE,IAAI,SAAS,CAAC,CAC3B,UAAU,CAAE,IACb,CAEA,kDAAM,CACL,IAAI,CAAE,CAAC,CAAC,CAAC,CAAC,EAAE,CACZ,MAAM,CAAE,GAAG,CAAC,MAAM,CAAC,IAAI,sBAAsB,CAAC,CAC9C,aAAa,CAAE,IAAI,QAAQ,CAAC,CAC5B,YAAY,CAAE,IAAI,QAAQ,CAC3B,CAEA,wDAAY,CACX,WAAW,CAAE,IAAI,CACjB,UAAU,CAAE,KACb"}'
};
function get_aria_referenceable_id(elem_id) {
  return elem_id.replace(/\s/g, "-");
}
const Label = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value } = $$props;
  createEventDispatcher();
  let { color = void 0 } = $$props;
  let { selectable = false } = $$props;
  let { show_heading = true } = $$props;
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.color === void 0 && $$bindings.color && color !== void 0)
    $$bindings.color(color);
  if ($$props.selectable === void 0 && $$bindings.selectable && selectable !== void 0)
    $$bindings.selectable(selectable);
  if ($$props.show_heading === void 0 && $$bindings.show_heading && show_heading !== void 0)
    $$bindings.show_heading(show_heading);
  $$result.css.add(css);
  return `<div class="container svelte-1mutzus">${show_heading || !value.confidences ? `<h2 class="${[
    "output-class svelte-1mutzus",
    !("confidences" in value) ? "no-confidence" : ""
  ].join(" ").trim()}" data-testid="label-output-value"${add_styles({
    "background-color": color || "transparent"
  })}>${escape(value.label)}</h2>` : ``} ${typeof value === "object" && value.confidences ? `${each(value.confidences, (confidence_set, i) => {
    return `<button class="${["confidence-set group svelte-1mutzus", selectable ? "selectable" : ""].join(" ").trim()}"${add_attribute("data-testid", `${confidence_set.label}-confidence-set`, 0)}><div class="inner-wrap svelte-1mutzus"><meter${add_attribute("aria-labelledby", get_aria_referenceable_id(`meter-text-${confidence_set.label}`), 0)}${add_attribute("aria-label", confidence_set.label, 0)}${add_attribute("aria-valuenow", Math.round(confidence_set.confidence * 100), 0)} aria-valuemin="0" aria-valuemax="100" class="bar svelte-1mutzus" min="0" max="1"${add_attribute("value", confidence_set.confidence, 0)} style="${"width: " + escape(confidence_set.confidence * 100, true) + "%; background: var(--stat-background-fill);"}"></meter> <dl class="label svelte-1mutzus"><dt${add_attribute("id", get_aria_referenceable_id(`meter-text-${confidence_set.label}`), 0)} class="text svelte-1mutzus">${escape(confidence_set.label)} </dt><div class="line svelte-1mutzus"></div><dd class="confidence svelte-1mutzus">${escape(Math.round(confidence_set.confidence * 100))}%
						</dd></dl></div> </button>`;
  })}` : ``} </div>`;
});
const Label$1 = Label;
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let _label;
  let { gradio } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { color = void 0 } = $$props;
  let { value = {} } = $$props;
  let old_value = null;
  let { label = gradio.i18n("label.label") } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { loading_status } = $$props;
  let { show_label = true } = $$props;
  let { _selectable = false } = $$props;
  let { show_heading = true } = $$props;
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.color === void 0 && $$bindings.color && color !== void 0)
    $$bindings.color(color);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.container === void 0 && $$bindings.container && container !== void 0)
    $$bindings.container(container);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  if ($$props.min_width === void 0 && $$bindings.min_width && min_width !== void 0)
    $$bindings.min_width(min_width);
  if ($$props.loading_status === void 0 && $$bindings.loading_status && loading_status !== void 0)
    $$bindings.loading_status(loading_status);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props._selectable === void 0 && $$bindings._selectable && _selectable !== void 0)
    $$bindings._selectable(_selectable);
  if ($$props.show_heading === void 0 && $$bindings.show_heading && show_heading !== void 0)
    $$bindings.show_heading(show_heading);
  {
    {
      if (JSON.stringify(value) !== JSON.stringify(old_value)) {
        old_value = value;
        gradio.dispatch("change");
      }
    }
  }
  _label = value.label;
  return `${validate_component(Block, "Block").$$render(
    $$result,
    {
      test_id: "label",
      visible,
      elem_id,
      elem_classes,
      container,
      scale,
      min_width,
      padding: false
    },
    {},
    {
      default: () => {
        return `${validate_component(Static, "StatusTracker").$$render($$result, Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status), {}, {})} ${show_label ? `${validate_component(BlockLabel, "BlockLabel").$$render(
          $$result,
          {
            Icon: LineChart,
            label,
            disable: container === false,
            float: show_heading === true
          },
          {},
          {}
        )}` : ``} ${_label !== void 0 && _label !== null ? `${validate_component(Label$1, "Label").$$render(
          $$result,
          {
            selectable: _selectable,
            value,
            color,
            show_heading
          },
          {},
          {}
        )}` : `${validate_component(Empty, "Empty").$$render($$result, { unpadded_box: true }, {}, {
          default: () => {
            return `${validate_component(LineChart, "LabelIcon").$$render($$result, {}, {}, {})}`;
          }
        })}`}`;
      }
    }
  )}`;
});

export { Label$1 as BaseLabel, Index as default };
//# sourceMappingURL=Index38-7vqJwYlL.js.map
