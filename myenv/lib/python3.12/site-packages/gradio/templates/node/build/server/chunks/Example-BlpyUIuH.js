import { c as create_ssr_component, e as escape } from './ssr-fyTaU2Wq.js';

const Example = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value } = $$props;
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  return `${escape(value || "")}`;
});

export { Example as default };
//# sourceMappingURL=Example-BlpyUIuH.js.map
