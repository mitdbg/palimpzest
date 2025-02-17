import { c as create_ssr_component, e as escape } from './ssr-fyTaU2Wq.js';

const Example = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { title } = $$props;
  let { x } = $$props;
  let { y } = $$props;
  if ($$props.title === void 0 && $$bindings.title && title !== void 0)
    $$bindings.title(title);
  if ($$props.x === void 0 && $$bindings.x && x !== void 0)
    $$bindings.x(x);
  if ($$props.y === void 0 && $$bindings.y && y !== void 0)
    $$bindings.y(y);
  return `${title ? `${escape(title)}` : `${escape(x)} x ${escape(y)}`}`;
});

export { Example as default };
//# sourceMappingURL=Example2-DReoc0dN.js.map
