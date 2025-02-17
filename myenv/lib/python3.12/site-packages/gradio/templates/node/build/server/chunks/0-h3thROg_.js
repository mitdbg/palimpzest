import { r as redirect } from './index3-DyoisQP2.js';

function load({ url }) {
  if (url.pathname !== "/") {
    redirect(308, "/");
  }
}

var _layout_server_ts = /*#__PURE__*/Object.freeze({
  __proto__: null,
  load: load
});

const index = 0;
let component_cache;
const component = async () => component_cache ??= (await import('./_layout.svelte-Bx0S5hZC.js')).default;
const server_id = "src/routes/+layout.server.ts";
const imports = ["_app/immutable/nodes/0.BltGkWcQ.js"];
const stylesheets = ["_app/immutable/assets/0.D1RK28YN.css"];
const fonts = [];

export { component, fonts, imports, index, _layout_server_ts as server, server_id, stylesheets };
//# sourceMappingURL=0-h3thROg_.js.map
