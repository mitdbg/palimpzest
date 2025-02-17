import { c as create_ssr_component, v as validate_component, a as createEventDispatcher, e as escape, b as add_attribute } from './ssr-fyTaU2Wq.js';
import { S as Static } from './2-CnaXPGyd.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BeHry4b7.js';

const css = {
  code: ".sidebar-parent{display:flex !important;padding-left:0;transition:padding-left 0.3s ease-in-out}.sidebar-parent:has(.sidebar.open){padding-left:var(--overlap-amount)}.sidebar.svelte-1qwdacm.svelte-1qwdacm{display:flex;flex-direction:column;position:fixed;top:0;height:100%;background-color:var(--background-fill-secondary);box-shadow:var(--size-1) 0 var(--size-2) rgba(100, 89, 89, 0.1);transform:translateX(0%);transition:transform 0.3s ease-in-out;z-index:1000}.sidebar.open.svelte-1qwdacm.svelte-1qwdacm{transform:translateX(100%)}.toggle-button.svelte-1qwdacm.svelte-1qwdacm{position:absolute;top:var(--size-4);right:calc(var(--size-8) * -1);background:none;border:none;cursor:pointer;padding:var(--size-2);display:flex;align-items:center;justify-content:center;transition:right 0.3s ease-in-out;width:var(--size-8);height:var(--size-8);z-index:1001}.open.svelte-1qwdacm .toggle-button.svelte-1qwdacm{right:var(--size-2-5);transform:rotate(180deg)}.chevron.svelte-1qwdacm.svelte-1qwdacm{width:100%;height:100%;position:relative;display:flex;align-items:center;justify-content:center}.chevron-left.svelte-1qwdacm.svelte-1qwdacm{position:relative;width:var(--size-3);height:var(--size-3);border-top:var(--size-0-5) solid var(--button-secondary-text-color);border-right:var(--size-0-5) solid var(--button-secondary-text-color);transform:rotate(45deg)}.sidebar-content.svelte-1qwdacm.svelte-1qwdacm{padding:var(--size-5);overflow-y:auto}",
  map: '{"version":3,"file":"Sidebar.svelte","sources":["Sidebar.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { createEventDispatcher, onMount } from \\"svelte\\";\\nconst dispatch = createEventDispatcher();\\nexport let open = true;\\nexport let width;\\nlet _open = false;\\nlet sidebar_div;\\nlet overlap_amount = 0;\\nlet width_css = typeof width === \\"number\\" ? `${width}px` : width;\\nfunction check_overlap() {\\n    if (!sidebar_div?.parentElement)\\n        return;\\n    const parent_rect = sidebar_div.parentElement.getBoundingClientRect();\\n    const sidebar_rect = sidebar_div.getBoundingClientRect();\\n    const available_space = parent_rect.left;\\n    overlap_amount = Math.max(0, sidebar_rect.width - available_space + 30);\\n}\\nonMount(() => {\\n    sidebar_div.parentElement?.classList.add(\\"sidebar-parent\\");\\n    check_overlap();\\n    window.addEventListener(\\"resize\\", check_overlap);\\n    const update_parent_overlap = () => {\\n        if (sidebar_div?.parentElement) {\\n            sidebar_div.parentElement.style.setProperty(\\"--overlap-amount\\", `${overlap_amount}px`);\\n        }\\n    };\\n    update_parent_overlap();\\n    _open = open;\\n    return () => window.removeEventListener(\\"resize\\", check_overlap);\\n});\\n<\/script>\\n\\n<div\\n\\tclass=\\"sidebar\\"\\n\\tclass:open={_open}\\n\\tbind:this={sidebar_div}\\n\\tstyle=\\"width: {width_css}; left: calc({width_css} * -1)\\"\\n>\\n\\t<button\\n\\t\\ton:click={() => {\\n\\t\\t\\t_open = !_open;\\n\\t\\t\\tif (_open) {\\n\\t\\t\\t\\tdispatch(\\"expand\\");\\n\\t\\t\\t} else {\\n\\t\\t\\t\\tdispatch(\\"collapse\\");\\n\\t\\t\\t}\\n\\t\\t}}\\n\\t\\tclass=\\"toggle-button\\"\\n\\t\\taria-label=\\"Toggle Sidebar\\"\\n\\t>\\n\\t\\t<div class=\\"chevron\\">\\n\\t\\t\\t<span class=\\"chevron-left\\"></span>\\n\\t\\t</div>\\n\\t</button>\\n\\t<div class=\\"sidebar-content\\">\\n\\t\\t<slot />\\n\\t</div>\\n</div>\\n\\n<style>\\n\\t:global(.sidebar-parent) {\\n\\t\\tdisplay: flex !important;\\n\\t\\tpadding-left: 0;\\n\\t\\ttransition: padding-left 0.3s ease-in-out;\\n\\t}\\n\\n\\t:global(.sidebar-parent:has(.sidebar.open)) {\\n\\t\\tpadding-left: var(--overlap-amount);\\n\\t}\\n\\n\\t.sidebar {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\tposition: fixed;\\n\\t\\ttop: 0;\\n\\t\\theight: 100%;\\n\\t\\tbackground-color: var(--background-fill-secondary);\\n\\t\\tbox-shadow: var(--size-1) 0 var(--size-2) rgba(100, 89, 89, 0.1);\\n\\t\\ttransform: translateX(0%);\\n\\t\\ttransition: transform 0.3s ease-in-out;\\n\\t\\tz-index: 1000;\\n\\t}\\n\\n\\t.sidebar.open {\\n\\t\\ttransform: translateX(100%);\\n\\t}\\n\\n\\t.toggle-button {\\n\\t\\tposition: absolute;\\n\\t\\ttop: var(--size-4);\\n\\t\\tright: calc(var(--size-8) * -1);\\n\\t\\tbackground: none;\\n\\t\\tborder: none;\\n\\t\\tcursor: pointer;\\n\\t\\tpadding: var(--size-2);\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t\\ttransition: right 0.3s ease-in-out;\\n\\t\\twidth: var(--size-8);\\n\\t\\theight: var(--size-8);\\n\\t\\tz-index: 1001;\\n\\t}\\n\\n\\t.open .toggle-button {\\n\\t\\tright: var(--size-2-5);\\n\\t\\ttransform: rotate(180deg);\\n\\t}\\n\\n\\t.chevron {\\n\\t\\twidth: 100%;\\n\\t\\theight: 100%;\\n\\t\\tposition: relative;\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t}\\n\\n\\t.chevron-left {\\n\\t\\tposition: relative;\\n\\t\\twidth: var(--size-3);\\n\\t\\theight: var(--size-3);\\n\\t\\tborder-top: var(--size-0-5) solid var(--button-secondary-text-color);\\n\\t\\tborder-right: var(--size-0-5) solid var(--button-secondary-text-color);\\n\\t\\ttransform: rotate(45deg);\\n\\t}\\n\\n\\t.sidebar-content {\\n\\t\\tpadding: var(--size-5);\\n\\t\\toverflow-y: auto;\\n\\t}</style>\\n"],"names":[],"mappings":"AA2DS,eAAiB,CACxB,OAAO,CAAE,IAAI,CAAC,UAAU,CACxB,YAAY,CAAE,CAAC,CACf,UAAU,CAAE,YAAY,CAAC,IAAI,CAAC,WAC/B,CAEQ,kCAAoC,CAC3C,YAAY,CAAE,IAAI,gBAAgB,CACnC,CAEA,sCAAS,CACR,OAAO,CAAE,IAAI,CACb,cAAc,CAAE,MAAM,CACtB,QAAQ,CAAE,KAAK,CACf,GAAG,CAAE,CAAC,CACN,MAAM,CAAE,IAAI,CACZ,gBAAgB,CAAE,IAAI,2BAA2B,CAAC,CAClD,UAAU,CAAE,IAAI,QAAQ,CAAC,CAAC,CAAC,CAAC,IAAI,QAAQ,CAAC,CAAC,KAAK,GAAG,CAAC,CAAC,EAAE,CAAC,CAAC,EAAE,CAAC,CAAC,GAAG,CAAC,CAChE,SAAS,CAAE,WAAW,EAAE,CAAC,CACzB,UAAU,CAAE,SAAS,CAAC,IAAI,CAAC,WAAW,CACtC,OAAO,CAAE,IACV,CAEA,QAAQ,mCAAM,CACb,SAAS,CAAE,WAAW,IAAI,CAC3B,CAEA,4CAAe,CACd,QAAQ,CAAE,QAAQ,CAClB,GAAG,CAAE,IAAI,QAAQ,CAAC,CAClB,KAAK,CAAE,KAAK,IAAI,QAAQ,CAAC,CAAC,CAAC,CAAC,EAAE,CAAC,CAC/B,UAAU,CAAE,IAAI,CAChB,MAAM,CAAE,IAAI,CACZ,MAAM,CAAE,OAAO,CACf,OAAO,CAAE,IAAI,QAAQ,CAAC,CACtB,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,eAAe,CAAE,MAAM,CACvB,UAAU,CAAE,KAAK,CAAC,IAAI,CAAC,WAAW,CAClC,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CAAC,CACrB,OAAO,CAAE,IACV,CAEA,oBAAK,CAAC,6BAAe,CACpB,KAAK,CAAE,IAAI,UAAU,CAAC,CACtB,SAAS,CAAE,OAAO,MAAM,CACzB,CAEA,sCAAS,CACR,KAAK,CAAE,IAAI,CACX,MAAM,CAAE,IAAI,CACZ,QAAQ,CAAE,QAAQ,CAClB,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,eAAe,CAAE,MAClB,CAEA,2CAAc,CACb,QAAQ,CAAE,QAAQ,CAClB,KAAK,CAAE,IAAI,QAAQ,CAAC,CACpB,MAAM,CAAE,IAAI,QAAQ,CAAC,CACrB,UAAU,CAAE,IAAI,UAAU,CAAC,CAAC,KAAK,CAAC,IAAI,6BAA6B,CAAC,CACpE,YAAY,CAAE,IAAI,UAAU,CAAC,CAAC,KAAK,CAAC,IAAI,6BAA6B,CAAC,CACtE,SAAS,CAAE,OAAO,KAAK,CACxB,CAEA,8CAAiB,CAChB,OAAO,CAAE,IAAI,QAAQ,CAAC,CACtB,UAAU,CAAE,IACb"}'
};
const Sidebar = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  createEventDispatcher();
  let { open = true } = $$props;
  let { width } = $$props;
  let sidebar_div;
  let width_css = typeof width === "number" ? `${width}px` : width;
  if ($$props.open === void 0 && $$bindings.open && open !== void 0)
    $$bindings.open(open);
  if ($$props.width === void 0 && $$bindings.width && width !== void 0)
    $$bindings.width(width);
  $$result.css.add(css);
  return `<div class="${["sidebar svelte-1qwdacm", ""].join(" ").trim()}" style="${"width: " + escape(width_css, true) + "; left: calc(" + escape(width_css, true) + " * -1)"}"${add_attribute("this", sidebar_div, 0)}><button class="toggle-button svelte-1qwdacm" aria-label="Toggle Sidebar" data-svelte-h="svelte-k78zcg"><div class="chevron svelte-1qwdacm"><span class="chevron-left svelte-1qwdacm"></span></div></button> <div class="sidebar-content svelte-1qwdacm">${slots.default ? slots.default({}) : ``}</div> </div>`;
});
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { open = true } = $$props;
  let { loading_status } = $$props;
  let { gradio } = $$props;
  let { width } = $$props;
  if ($$props.open === void 0 && $$bindings.open && open !== void 0)
    $$bindings.open(open);
  if ($$props.loading_status === void 0 && $$bindings.loading_status && loading_status !== void 0)
    $$bindings.loading_status(loading_status);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.width === void 0 && $$bindings.width && width !== void 0)
    $$bindings.width(width);
  let $$settled;
  let $$rendered;
  let previous_head = $$result.head;
  do {
    $$settled = true;
    $$result.head = previous_head;
    $$rendered = `${validate_component(Static, "StatusTracker").$$render($$result, Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status), {}, {})} ${validate_component(Sidebar, "Sidebar").$$render(
      $$result,
      { width, open },
      {
        open: ($$value) => {
          open = $$value;
          $$settled = false;
        }
      },
      {
        default: () => {
          return `${slots.default ? slots.default({}) : ``}`;
        }
      }
    )}`;
  } while (!$$settled);
  return $$rendered;
});

export { Index as default };
//# sourceMappingURL=Index40-GGH7EURP.js.map
