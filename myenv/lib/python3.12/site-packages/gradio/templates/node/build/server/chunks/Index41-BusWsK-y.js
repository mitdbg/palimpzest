import { c as create_ssr_component, v as validate_component, a as createEventDispatcher, g as getContext, s as subscribe, b as add_attribute, e as escape, d as add_styles } from './ssr-fyTaU2Wq.js';
import { t as tick } from './Component-BeHry4b7.js';
import { TABS } from './Index42-BJLHqo4_.js';
import { I as Index$1 } from './2-CnaXPGyd.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';

const css = {
  code: "div.svelte-wv8on1{display:flex;flex-direction:column;position:relative;border:none;border-radius:var(--radius-sm);padding:var(--block-padding);width:100%;box-sizing:border-box}.grow-children.svelte-wv8on1>.column > .column{flex-grow:1}",
  map: `{"version":3,"file":"TabItem.svelte","sources":["TabItem.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { getContext, onMount, createEventDispatcher, tick } from \\"svelte\\";\\nimport { TABS } from \\"@gradio/tabs\\";\\nimport Column from \\"@gradio/column\\";\\nexport let elem_id = \\"\\";\\nexport let elem_classes = [];\\nexport let label;\\nexport let id = {};\\nexport let visible;\\nexport let interactive;\\nexport let order;\\nexport let scale;\\nconst dispatch = createEventDispatcher();\\nconst { register_tab, unregister_tab, selected_tab, selected_tab_index } = getContext(TABS);\\nlet tab_index;\\n$: tab_index = register_tab({ label, id, elem_id, visible, interactive, scale }, order);\\nonMount(() => {\\n    return () => unregister_tab({ label, id, elem_id }, order);\\n});\\n$: $selected_tab_index === tab_index && tick().then(() => dispatch(\\"select\\", { value: label, index: tab_index }));\\n<\/script>\\n\\n<!-- {#if $selected_tab === id && visible} -->\\n<div\\n\\tid={elem_id}\\n\\tclass=\\"tabitem {elem_classes.join(' ')}\\"\\n\\tclass:grow-children={scale >= 1}\\n\\tstyle:display={$selected_tab === id && visible ? \\"flex\\" : \\"none\\"}\\n\\tstyle:flex-grow={scale}\\n\\trole=\\"tabpanel\\"\\n>\\n\\t<Column scale={scale >= 1 ? scale : null}>\\n\\t\\t<slot />\\n\\t</Column>\\n</div>\\n\\n<!-- {/if} -->\\n\\n<style>\\n\\tdiv {\\n\\t\\tdisplay: flex;\\n\\t\\tflex-direction: column;\\n\\t\\tposition: relative;\\n\\t\\tborder: none;\\n\\t\\tborder-radius: var(--radius-sm);\\n\\t\\tpadding: var(--block-padding);\\n\\t\\twidth: 100%;\\n\\t\\tbox-sizing: border-box;\\n\\t}\\n\\t.grow-children > :global(.column > .column) {\\n\\t\\tflex-grow: 1;\\n\\t}</style>\\n"],"names":[],"mappings":"AAsCC,iBAAI,CACH,OAAO,CAAE,IAAI,CACb,cAAc,CAAE,MAAM,CACtB,QAAQ,CAAE,QAAQ,CAClB,MAAM,CAAE,IAAI,CACZ,aAAa,CAAE,IAAI,WAAW,CAAC,CAC/B,OAAO,CAAE,IAAI,eAAe,CAAC,CAC7B,KAAK,CAAE,IAAI,CACX,UAAU,CAAE,UACb,CACA,4BAAc,CAAW,iBAAmB,CAC3C,SAAS,CAAE,CACZ"}`
};
const TabItem = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let $selected_tab_index, $$unsubscribe_selected_tab_index;
  let $selected_tab, $$unsubscribe_selected_tab;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { label } = $$props;
  let { id = {} } = $$props;
  let { visible } = $$props;
  let { interactive } = $$props;
  let { order } = $$props;
  let { scale } = $$props;
  const dispatch = createEventDispatcher();
  const { register_tab, unregister_tab, selected_tab, selected_tab_index } = getContext(TABS);
  $$unsubscribe_selected_tab = subscribe(selected_tab, (value) => $selected_tab = value);
  $$unsubscribe_selected_tab_index = subscribe(selected_tab_index, (value) => $selected_tab_index = value);
  let tab_index;
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.id === void 0 && $$bindings.id && id !== void 0)
    $$bindings.id(id);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.interactive === void 0 && $$bindings.interactive && interactive !== void 0)
    $$bindings.interactive(interactive);
  if ($$props.order === void 0 && $$bindings.order && order !== void 0)
    $$bindings.order(order);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  $$result.css.add(css);
  tab_index = register_tab(
    {
      label,
      id,
      elem_id,
      visible,
      interactive,
      scale
    },
    order
  );
  $selected_tab_index === tab_index && tick().then(() => dispatch("select", { value: label, index: tab_index }));
  $$unsubscribe_selected_tab_index();
  $$unsubscribe_selected_tab();
  return ` <div${add_attribute("id", elem_id, 0)} class="${[
    "tabitem " + escape(elem_classes.join(" "), true) + " svelte-wv8on1",
    scale >= 1 ? "grow-children" : ""
  ].join(" ").trim()}" role="tabpanel"${add_styles({
    "display": $selected_tab === id && visible ? "flex" : "none",
    "flex-grow": scale
  })}>${validate_component(Index$1, "Column").$$render($$result, { scale: scale >= 1 ? scale : null }, {}, {
    default: () => {
      return `${slots.default ? slots.default({}) : ``}`;
    }
  })}</div> `;
});
const TabItem$1 = TabItem;
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { label } = $$props;
  let { id } = $$props;
  let { gradio } = $$props;
  let { visible = true } = $$props;
  let { interactive = true } = $$props;
  let { order } = $$props;
  let { scale } = $$props;
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.id === void 0 && $$bindings.id && id !== void 0)
    $$bindings.id(id);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.interactive === void 0 && $$bindings.interactive && interactive !== void 0)
    $$bindings.interactive(interactive);
  if ($$props.order === void 0 && $$bindings.order && order !== void 0)
    $$bindings.order(order);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  return `${validate_component(TabItem$1, "TabItem").$$render(
    $$result,
    {
      elem_id,
      elem_classes,
      label,
      visible,
      interactive,
      id,
      order,
      scale
    },
    {},
    {
      default: () => {
        return `${slots.default ? slots.default({}) : ``}`;
      }
    }
  )}`;
});

export { TabItem$1 as BaseTabItem, Index as default };
//# sourceMappingURL=Index41-BusWsK-y.js.map
