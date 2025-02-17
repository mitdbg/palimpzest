import { c as create_ssr_component, v as validate_component } from './ssr-fyTaU2Wq.js';
import { B as Block } from './2-CnaXPGyd.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BeHry4b7.js';

const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { elem_id } = $$props;
  let { elem_classes } = $$props;
  let { visible = true } = $$props;
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  return `${validate_component(Block, "Block").$$render(
    $$result,
    {
      elem_id,
      elem_classes,
      visible,
      explicit_call: true
    },
    {},
    {
      default: () => {
        return `${slots.default ? slots.default({}) : ``}`;
      }
    }
  )}`;
});

export { Index as default };
//# sourceMappingURL=Index5-DHsA6G92.js.map
