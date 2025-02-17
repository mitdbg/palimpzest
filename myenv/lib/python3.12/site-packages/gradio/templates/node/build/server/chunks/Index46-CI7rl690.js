import { c as create_ssr_component, v as validate_component, d as add_styles, a as createEventDispatcher, e as escape } from './ssr-fyTaU2Wq.js';
import { B as Block, f as BlockLabel, N as Code, S as Static, ad as css_units } from './2-CnaXPGyd.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BeHry4b7.js';

const css$1 = {
  code: ".hide.svelte-ydeks8{display:none}",
  map: `{"version":3,"file":"HTML.svelte","sources":["HTML.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { createEventDispatcher } from \\"svelte\\";\\nexport let elem_classes = [];\\nexport let value;\\nexport let visible = true;\\nconst dispatch = createEventDispatcher();\\n$: value, dispatch(\\"change\\");\\n<\/script>\\n\\n<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->\\n<div\\n\\tclass=\\"prose {elem_classes.join(' ')}\\"\\n\\tclass:hide={!visible}\\n\\ton:click={() => dispatch(\\"click\\")}\\n>\\n\\t{@html value}\\n</div>\\n\\n<style>\\n\\t.hide {\\n\\t\\tdisplay: none;\\n\\t}</style>\\n"],"names":[],"mappings":"AAkBC,mBAAM,CACL,OAAO,CAAE,IACV"}`
};
const HTML = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { elem_classes = [] } = $$props;
  let { value } = $$props;
  let { visible = true } = $$props;
  const dispatch = createEventDispatcher();
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  $$result.css.add(css$1);
  {
    dispatch("change");
  }
  return ` <div class="${[
    "prose " + escape(elem_classes.join(" "), true) + " svelte-ydeks8",
    !visible ? "hide" : ""
  ].join(" ").trim()}"><!-- HTML_TAG_START -->${value}<!-- HTML_TAG_END --> </div>`;
});
const css = {
  code: ".padding.svelte-phx28p{padding:var(--block-padding)}div.svelte-phx28p{transition:150ms}.pending.svelte-phx28p{opacity:0.2}",
  map: '{"version":3,"file":"Index.svelte","sources":["Index.svelte"],"sourcesContent":["<script lang=\\"ts\\">import HTML from \\"./shared/HTML.svelte\\";\\nimport { StatusTracker } from \\"@gradio/statustracker\\";\\nimport { Block, BlockLabel } from \\"@gradio/atoms\\";\\nimport { Code as CodeIcon } from \\"@gradio/icons\\";\\nimport { css_units } from \\"@gradio/utils\\";\\nexport let label = \\"HTML\\";\\nexport let elem_id = \\"\\";\\nexport let elem_classes = [];\\nexport let visible = true;\\nexport let value = \\"\\";\\nexport let loading_status;\\nexport let gradio;\\nexport let show_label = false;\\nexport let min_height = void 0;\\nexport let max_height = void 0;\\nexport let container = false;\\nexport let padding = true;\\n$: label, gradio.dispatch(\\"change\\");\\n<\/script>\\n\\n<Block {visible} {elem_id} {elem_classes} {container} padding={false}>\\n\\t{#if show_label}\\n\\t\\t<BlockLabel Icon={CodeIcon} {show_label} {label} float={false} />\\n\\t{/if}\\n\\n\\t<StatusTracker\\n\\t\\tautoscroll={gradio.autoscroll}\\n\\t\\ti18n={gradio.i18n}\\n\\t\\t{...loading_status}\\n\\t\\tvariant=\\"center\\"\\n\\t\\ton:clear_status={() => gradio.dispatch(\\"clear_status\\", loading_status)}\\n\\t/>\\n\\t<div\\n\\t\\tclass=\\"html-container\\"\\n\\t\\tclass:padding\\n\\t\\tclass:pending={loading_status?.status === \\"pending\\"}\\n\\t\\tstyle:min-height={min_height && loading_status?.status !== \\"pending\\"\\n\\t\\t\\t? css_units(min_height)\\n\\t\\t\\t: undefined}\\n\\t\\tstyle:max-height={max_height ? css_units(max_height) : undefined}\\n\\t>\\n\\t\\t<HTML\\n\\t\\t\\t{value}\\n\\t\\t\\t{elem_classes}\\n\\t\\t\\t{visible}\\n\\t\\t\\ton:change={() => gradio.dispatch(\\"change\\")}\\n\\t\\t\\ton:click={() => gradio.dispatch(\\"click\\")}\\n\\t\\t/>\\n\\t</div>\\n</Block>\\n\\n<style>\\n\\t.padding {\\n\\t\\tpadding: var(--block-padding);\\n\\t}\\n\\n\\tdiv {\\n\\t\\ttransition: 150ms;\\n\\t}\\n\\n\\t.pending {\\n\\t\\topacity: 0.2;\\n\\t}</style>\\n"],"names":[],"mappings":"AAoDC,sBAAS,CACR,OAAO,CAAE,IAAI,eAAe,CAC7B,CAEA,iBAAI,CACH,UAAU,CAAE,KACb,CAEA,sBAAS,CACR,OAAO,CAAE,GACV"}'
};
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { label = "HTML" } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = "" } = $$props;
  let { loading_status } = $$props;
  let { gradio } = $$props;
  let { show_label = false } = $$props;
  let { min_height = void 0 } = $$props;
  let { max_height = void 0 } = $$props;
  let { container = false } = $$props;
  let { padding = true } = $$props;
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.loading_status === void 0 && $$bindings.loading_status && loading_status !== void 0)
    $$bindings.loading_status(loading_status);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.min_height === void 0 && $$bindings.min_height && min_height !== void 0)
    $$bindings.min_height(min_height);
  if ($$props.max_height === void 0 && $$bindings.max_height && max_height !== void 0)
    $$bindings.max_height(max_height);
  if ($$props.container === void 0 && $$bindings.container && container !== void 0)
    $$bindings.container(container);
  if ($$props.padding === void 0 && $$bindings.padding && padding !== void 0)
    $$bindings.padding(padding);
  $$result.css.add(css);
  {
    gradio.dispatch("change");
  }
  return `${validate_component(Block, "Block").$$render(
    $$result,
    {
      visible,
      elem_id,
      elem_classes,
      container,
      padding: false
    },
    {},
    {
      default: () => {
        return `${show_label ? `${validate_component(BlockLabel, "BlockLabel").$$render(
          $$result,
          {
            Icon: Code,
            show_label,
            label,
            float: false
          },
          {},
          {}
        )}` : ``} ${validate_component(Static, "StatusTracker").$$render($$result, Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status, { variant: "center" }), {}, {})} <div class="${[
          "html-container svelte-phx28p",
          (padding ? "padding" : "") + " " + (loading_status?.status === "pending" ? "pending" : "")
        ].join(" ").trim()}"${add_styles({
          "min-height": min_height && loading_status?.status !== "pending" ? css_units(min_height) : void 0,
          "max-height": max_height ? css_units(max_height) : void 0
        })}>${validate_component(HTML, "HTML").$$render($$result, { value, elem_classes, visible }, {}, {})}</div>`;
      }
    }
  )}`;
});

export { Index as default };
//# sourceMappingURL=Index46-CI7rl690.js.map
