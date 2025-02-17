import { c as create_ssr_component, e as escape, f as each } from './ssr-fyTaU2Wq.js';

const css = {
  code: "table.svelte-hn96gn.svelte-hn96gn{position:relative;border-collapse:collapse}td.svelte-hn96gn.svelte-hn96gn{border:1px solid var(--table-border-color);padding:var(--size-2);font-size:var(--text-sm);font-family:var(--font-mono)}.selected.svelte-hn96gn td.svelte-hn96gn{border-color:var(--border-color-accent)}.table.svelte-hn96gn.svelte-hn96gn{display:inline-block;margin:0 auto}.gallery.svelte-hn96gn td.svelte-hn96gn:first-child{border-left:none}.gallery.svelte-hn96gn tr:first-child td.svelte-hn96gn{border-top:none}.gallery.svelte-hn96gn td.svelte-hn96gn:last-child{border-right:none}.gallery.svelte-hn96gn tr:last-child td.svelte-hn96gn{border-bottom:none}.overlay.svelte-hn96gn.svelte-hn96gn{--gradient-to:transparent;position:absolute;bottom:0;background:linear-gradient(to bottom, transparent, var(--gradient-to));width:var(--size-full);height:50%}.odd.svelte-hn96gn.svelte-hn96gn{--gradient-to:var(--table-even-background-fill)}.even.svelte-hn96gn.svelte-hn96gn{--gradient-to:var(--table-odd-background-fill)}.button.svelte-hn96gn.svelte-hn96gn{--gradient-to:var(--background-fill-primary)}",
  map: `{"version":3,"file":"Example.svelte","sources":["Example.svelte"],"sourcesContent":["<script lang=\\"ts\\">export let value;\\nexport let type;\\nexport let selected = false;\\nexport let index;\\nlet hovered = false;\\nlet loaded = Array.isArray(value);\\n<\/script>\\n\\n{#if loaded}\\n\\t<!-- TODO: fix-->\\n\\t<!-- svelte-ignore a11y-no-static-element-interactions-->\\n\\t<div\\n\\t\\tclass:table={type === \\"table\\"}\\n\\t\\tclass:gallery={type === \\"gallery\\"}\\n\\t\\tclass:selected\\n\\t\\ton:mouseenter={() => (hovered = true)}\\n\\t\\ton:mouseleave={() => (hovered = false)}\\n\\t>\\n\\t\\t{#if typeof value === \\"string\\"}\\n\\t\\t\\t{value}\\n\\t\\t{:else}\\n\\t\\t\\t<table class=\\"\\">\\n\\t\\t\\t\\t{#each value.slice(0, 3) as row, i}\\n\\t\\t\\t\\t\\t<tr>\\n\\t\\t\\t\\t\\t\\t{#each row.slice(0, 3) as cell, j}\\n\\t\\t\\t\\t\\t\\t\\t<td>{cell}</td>\\n\\t\\t\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t\\t\\t{#if row.length > 3}\\n\\t\\t\\t\\t\\t\\t\\t<td>…</td>\\n\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t</tr>\\n\\t\\t\\t\\t{/each}\\n\\t\\t\\t\\t{#if value.length > 3}\\n\\t\\t\\t\\t\\t<div\\n\\t\\t\\t\\t\\t\\tclass=\\"overlay\\"\\n\\t\\t\\t\\t\\t\\tclass:odd={index % 2 != 0}\\n\\t\\t\\t\\t\\t\\tclass:even={index % 2 == 0}\\n\\t\\t\\t\\t\\t\\tclass:button={type === \\"gallery\\"}\\n\\t\\t\\t\\t\\t/>\\n\\t\\t\\t\\t{/if}\\n\\t\\t\\t</table>\\n\\t\\t{/if}\\n\\t</div>\\n{/if}\\n\\n<style>\\n\\ttable {\\n\\t\\tposition: relative;\\n\\t\\tborder-collapse: collapse;\\n\\t}\\n\\n\\ttd {\\n\\t\\tborder: 1px solid var(--table-border-color);\\n\\t\\tpadding: var(--size-2);\\n\\t\\tfont-size: var(--text-sm);\\n\\t\\tfont-family: var(--font-mono);\\n\\t}\\n\\n\\t.selected td {\\n\\t\\tborder-color: var(--border-color-accent);\\n\\t}\\n\\n\\t.table {\\n\\t\\tdisplay: inline-block;\\n\\t\\tmargin: 0 auto;\\n\\t}\\n\\n\\t.gallery td:first-child {\\n\\t\\tborder-left: none;\\n\\t}\\n\\n\\t.gallery tr:first-child td {\\n\\t\\tborder-top: none;\\n\\t}\\n\\n\\t.gallery td:last-child {\\n\\t\\tborder-right: none;\\n\\t}\\n\\n\\t.gallery tr:last-child td {\\n\\t\\tborder-bottom: none;\\n\\t}\\n\\n\\t.overlay {\\n\\t\\t--gradient-to: transparent;\\n\\t\\tposition: absolute;\\n\\t\\tbottom: 0;\\n\\t\\tbackground: linear-gradient(to bottom, transparent, var(--gradient-to));\\n\\t\\twidth: var(--size-full);\\n\\t\\theight: 50%;\\n\\t}\\n\\n\\t/* i dont know what i've done here but it is what it is */\\n\\t.odd {\\n\\t\\t--gradient-to: var(--table-even-background-fill);\\n\\t}\\n\\n\\t.even {\\n\\t\\t--gradient-to: var(--table-odd-background-fill);\\n\\t}\\n\\n\\t.button {\\n\\t\\t--gradient-to: var(--background-fill-primary);\\n\\t}</style>\\n"],"names":[],"mappings":"AA8CC,iCAAM,CACL,QAAQ,CAAE,QAAQ,CAClB,eAAe,CAAE,QAClB,CAEA,8BAAG,CACF,MAAM,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,oBAAoB,CAAC,CAC3C,OAAO,CAAE,IAAI,QAAQ,CAAC,CACtB,SAAS,CAAE,IAAI,SAAS,CAAC,CACzB,WAAW,CAAE,IAAI,WAAW,CAC7B,CAEA,uBAAS,CAAC,gBAAG,CACZ,YAAY,CAAE,IAAI,qBAAqB,CACxC,CAEA,kCAAO,CACN,OAAO,CAAE,YAAY,CACrB,MAAM,CAAE,CAAC,CAAC,IACX,CAEA,sBAAQ,CAAC,gBAAE,YAAa,CACvB,WAAW,CAAE,IACd,CAEA,sBAAQ,CAAC,EAAE,YAAY,CAAC,gBAAG,CAC1B,UAAU,CAAE,IACb,CAEA,sBAAQ,CAAC,gBAAE,WAAY,CACtB,YAAY,CAAE,IACf,CAEA,sBAAQ,CAAC,EAAE,WAAW,CAAC,gBAAG,CACzB,aAAa,CAAE,IAChB,CAEA,oCAAS,CACR,aAAa,CAAE,WAAW,CAC1B,QAAQ,CAAE,QAAQ,CAClB,MAAM,CAAE,CAAC,CACT,UAAU,CAAE,gBAAgB,EAAE,CAAC,MAAM,CAAC,CAAC,WAAW,CAAC,CAAC,IAAI,aAAa,CAAC,CAAC,CACvE,KAAK,CAAE,IAAI,WAAW,CAAC,CACvB,MAAM,CAAE,GACT,CAGA,gCAAK,CACJ,aAAa,CAAE,iCAChB,CAEA,iCAAM,CACL,aAAa,CAAE,gCAChB,CAEA,mCAAQ,CACP,aAAa,CAAE,8BAChB"}`
};
const Example = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value } = $$props;
  let { type } = $$props;
  let { selected = false } = $$props;
  let { index } = $$props;
  let loaded = Array.isArray(value);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.type === void 0 && $$bindings.type && type !== void 0)
    $$bindings.type(type);
  if ($$props.selected === void 0 && $$bindings.selected && selected !== void 0)
    $$bindings.selected(selected);
  if ($$props.index === void 0 && $$bindings.index && index !== void 0)
    $$bindings.index(index);
  $$result.css.add(css);
  return `${loaded ? `  <div class="${[
    "svelte-hn96gn",
    (type === "table" ? "table" : "") + " " + (type === "gallery" ? "gallery" : "") + " " + (selected ? "selected" : "")
  ].join(" ").trim()}">${typeof value === "string" ? `${escape(value)}` : `<table class=" svelte-hn96gn">${each(value.slice(0, 3), (row, i) => {
    return `<tr>${each(row.slice(0, 3), (cell, j) => {
      return `<td class="svelte-hn96gn">${escape(cell)}</td>`;
    })} ${row.length > 3 ? `<td class="svelte-hn96gn" data-svelte-h="svelte-1o35md4">…</td>` : ``} </tr>`;
  })} ${value.length > 3 ? `<div class="${[
    "overlay svelte-hn96gn",
    (index % 2 != 0 ? "odd" : "") + " " + (index % 2 == 0 ? "even" : "") + " " + (type === "gallery" ? "button" : "")
  ].join(" ").trim()}"></div>` : ``}</table>`}</div>` : ``}`;
});

export { Example as default };
//# sourceMappingURL=Example8-C9bAZQI2.js.map
