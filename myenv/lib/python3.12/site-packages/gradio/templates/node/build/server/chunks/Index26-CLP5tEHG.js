import { c as create_ssr_component, b as add_attribute, e as escape, d as add_styles, v as validate_component } from './ssr-fyTaU2Wq.js';
import { S as Static } from './2-CnaXPGyd.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BeHry4b7.js';

const css = {
  code: "div.svelte-1xp0cw7{display:flex;flex-wrap:wrap;gap:var(--layout-gap);width:var(--size-full);position:relative}.hide.svelte-1xp0cw7{display:none}.compact.svelte-1xp0cw7>*,.compact.svelte-1xp0cw7 .box{border-radius:0}.compact.svelte-1xp0cw7,.panel.svelte-1xp0cw7{border-radius:var(--container-radius);background:var(--background-fill-secondary);padding:var(--size-2)}.unequal-height.svelte-1xp0cw7{align-items:flex-start}.stretch.svelte-1xp0cw7{align-items:stretch}.stretch.svelte-1xp0cw7>.column > *,.stretch.svelte-1xp0cw7>.column > .form > *{flex-grow:1;flex-shrink:0}div.svelte-1xp0cw7>*,div.svelte-1xp0cw7>.form > *{flex:1 1 0%;flex-wrap:wrap;min-width:min(160px, 100%)}.grow-children.svelte-1xp0cw7>.column{align-self:stretch}",
  map: `{"version":3,"file":"Index.svelte","sources":["Index.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { StatusTracker } from \\"@gradio/statustracker\\";\\nexport let equal_height = true;\\nexport let elem_id;\\nexport let elem_classes = [];\\nexport let visible = true;\\nexport let variant = \\"default\\";\\nexport let loading_status = void 0;\\nexport let gradio = void 0;\\nexport let show_progress = false;\\nexport let height;\\nexport let min_height;\\nexport let max_height;\\nexport let scale = null;\\nconst get_dimension = (dimension_value) => {\\n    if (dimension_value === void 0) {\\n        return void 0;\\n    }\\n    if (typeof dimension_value === \\"number\\") {\\n        return dimension_value + \\"px\\";\\n    }\\n    else if (typeof dimension_value === \\"string\\") {\\n        return dimension_value;\\n    }\\n};\\n<\/script>\\n\\n<div\\n\\tclass:compact={variant === \\"compact\\"}\\n\\tclass:panel={variant === \\"panel\\"}\\n\\tclass:unequal-height={equal_height === false}\\n\\tclass:stretch={equal_height}\\n\\tclass:hide={!visible}\\n\\tclass:grow-children={scale && scale >= 1}\\n\\tstyle:height={get_dimension(height)}\\n\\tstyle:max-height={get_dimension(max_height)}\\n\\tstyle:min-height={get_dimension(min_height)}\\n\\tstyle:flex-grow={scale}\\n\\tid={elem_id}\\n\\tclass=\\"row {elem_classes.join(' ')}\\"\\n>\\n\\t{#if loading_status && show_progress && gradio}\\n\\t\\t<StatusTracker\\n\\t\\t\\tautoscroll={gradio.autoscroll}\\n\\t\\t\\ti18n={gradio.i18n}\\n\\t\\t\\t{...loading_status}\\n\\t\\t\\tstatus={loading_status\\n\\t\\t\\t\\t? loading_status.status == \\"pending\\"\\n\\t\\t\\t\\t\\t? \\"generating\\"\\n\\t\\t\\t\\t\\t: loading_status.status\\n\\t\\t\\t\\t: null}\\n\\t\\t/>\\n\\t{/if}\\n\\t<slot />\\n</div>\\n\\n<style>\\n\\tdiv {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-wrap: wrap;\\n\\t\\tgap: var(--layout-gap);\\n\\t\\twidth: var(--size-full);\\n\\t\\tposition: relative;\\n\\t}\\n\\n\\t.hide {\\n\\t\\tdisplay: none;\\n\\t}\\n\\t.compact > :global(*),\\n\\t.compact :global(.box) {\\n\\t\\tborder-radius: 0;\\n\\t}\\n\\t.compact,\\n\\t.panel {\\n\\t\\tborder-radius: var(--container-radius);\\n\\t\\tbackground: var(--background-fill-secondary);\\n\\t\\tpadding: var(--size-2);\\n\\t}\\n\\t.unequal-height {\\n\\t\\talign-items: flex-start;\\n\\t}\\n\\n\\t.stretch {\\n\\t\\talign-items: stretch;\\n\\t}\\n\\n\\t.stretch > :global(.column > *),\\n\\t.stretch > :global(.column > .form > *) {\\n\\t\\tflex-grow: 1;\\n\\t\\tflex-shrink: 0;\\n\\t}\\n\\n\\tdiv > :global(*),\\n\\tdiv > :global(.form > *) {\\n\\t\\tflex: 1 1 0%;\\n\\t\\tflex-wrap: wrap;\\n\\t\\tmin-width: min(160px, 100%);\\n\\t}\\n\\n\\t.grow-children > :global(.column) {\\n\\t\\talign-self: stretch;\\n\\t}</style>\\n"],"names":[],"mappings":"AAwDC,kBAAI,CACH,OAAO,CAAE,IAAI,CACb,SAAS,CAAE,IAAI,CACf,GAAG,CAAE,IAAI,YAAY,CAAC,CACtB,KAAK,CAAE,IAAI,WAAW,CAAC,CACvB,QAAQ,CAAE,QACX,CAEA,oBAAM,CACL,OAAO,CAAE,IACV,CACA,uBAAQ,CAAW,CAAE,CACrB,uBAAQ,CAAS,IAAM,CACtB,aAAa,CAAE,CAChB,CACA,uBAAQ,CACR,qBAAO,CACN,aAAa,CAAE,IAAI,kBAAkB,CAAC,CACtC,UAAU,CAAE,IAAI,2BAA2B,CAAC,CAC5C,OAAO,CAAE,IAAI,QAAQ,CACtB,CACA,8BAAgB,CACf,WAAW,CAAE,UACd,CAEA,uBAAS,CACR,WAAW,CAAE,OACd,CAEA,uBAAQ,CAAW,WAAY,CAC/B,uBAAQ,CAAW,mBAAqB,CACvC,SAAS,CAAE,CAAC,CACZ,WAAW,CAAE,CACd,CAEA,kBAAG,CAAW,CAAE,CAChB,kBAAG,CAAW,SAAW,CACxB,IAAI,CAAE,CAAC,CAAC,CAAC,CAAC,EAAE,CACZ,SAAS,CAAE,IAAI,CACf,SAAS,CAAE,IAAI,KAAK,CAAC,CAAC,IAAI,CAC3B,CAEA,6BAAc,CAAW,OAAS,CACjC,UAAU,CAAE,OACb"}`
};
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { equal_height = true } = $$props;
  let { elem_id } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { variant = "default" } = $$props;
  let { loading_status = void 0 } = $$props;
  let { gradio = void 0 } = $$props;
  let { show_progress = false } = $$props;
  let { height } = $$props;
  let { min_height } = $$props;
  let { max_height } = $$props;
  let { scale = null } = $$props;
  const get_dimension = (dimension_value) => {
    if (dimension_value === void 0) {
      return void 0;
    }
    if (typeof dimension_value === "number") {
      return dimension_value + "px";
    } else if (typeof dimension_value === "string") {
      return dimension_value;
    }
  };
  if ($$props.equal_height === void 0 && $$bindings.equal_height && equal_height !== void 0)
    $$bindings.equal_height(equal_height);
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.variant === void 0 && $$bindings.variant && variant !== void 0)
    $$bindings.variant(variant);
  if ($$props.loading_status === void 0 && $$bindings.loading_status && loading_status !== void 0)
    $$bindings.loading_status(loading_status);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.show_progress === void 0 && $$bindings.show_progress && show_progress !== void 0)
    $$bindings.show_progress(show_progress);
  if ($$props.height === void 0 && $$bindings.height && height !== void 0)
    $$bindings.height(height);
  if ($$props.min_height === void 0 && $$bindings.min_height && min_height !== void 0)
    $$bindings.min_height(min_height);
  if ($$props.max_height === void 0 && $$bindings.max_height && max_height !== void 0)
    $$bindings.max_height(max_height);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  $$result.css.add(css);
  return `<div${add_attribute("id", elem_id, 0)} class="${[
    "row " + escape(elem_classes.join(" "), true) + " svelte-1xp0cw7",
    (variant === "compact" ? "compact" : "") + " " + (variant === "panel" ? "panel" : "") + " " + (equal_height === false ? "unequal-height" : "") + " " + (equal_height ? "stretch" : "") + " " + (!visible ? "hide" : "") + " " + (scale && scale >= 1 ? "grow-children" : "")
  ].join(" ").trim()}"${add_styles({
    "height": get_dimension(height),
    "max-height": get_dimension(max_height),
    "min-height": get_dimension(min_height),
    "flex-grow": scale
  })}>${loading_status && show_progress && gradio ? `${validate_component(Static, "StatusTracker").$$render(
    $$result,
    Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status, {
      status: loading_status ? loading_status.status == "pending" ? "generating" : loading_status.status : null
    }),
    {},
    {}
  )}` : ``} ${slots.default ? slots.default({}) : ``} </div>`;
});

export { Index as default };
//# sourceMappingURL=Index26-CLP5tEHG.js.map
