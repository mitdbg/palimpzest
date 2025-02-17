import { c as create_ssr_component, v as validate_component, a as createEventDispatcher, d as add_styles, f as each, b as add_attribute, e as escape } from './ssr-fyTaU2Wq.js';
import { B as Block, S as Static, U as UploadText, f as BlockLabel, v as File$1, h as Empty, i as IconButtonWrapper, j as IconButton, a8 as Upload, z as Clear } from './2-CnaXPGyd.js';
import { D as DownloadLink } from './DownloadLink-BKK_IWmU.js';
import { U as Upload$1 } from './ModifyUpload-DkxchCER.js';
export { default as BaseExample } from './Example11-BjE799Om.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BeHry4b7.js';

const prettyBytes = (bytes) => {
  let units = ["B", "KB", "MB", "GB", "PB"];
  let i = 0;
  while (bytes > 1024) {
    bytes /= 1024;
    i++;
  }
  let unit = units[i];
  return bytes.toFixed(1) + "&nbsp;" + unit;
};
const css = {
  code: '.label-clear-button.svelte-1rvzbk6.svelte-1rvzbk6{color:var(--body-text-color-subdued);position:relative;left:-3px}.label-clear-button.svelte-1rvzbk6.svelte-1rvzbk6:hover{color:var(--body-text-color)}.file-preview.svelte-1rvzbk6.svelte-1rvzbk6{table-layout:fixed;width:var(--size-full);max-height:var(--size-60);overflow-y:auto;margin-top:var(--size-1);color:var(--body-text-color)}.file-preview-holder.svelte-1rvzbk6.svelte-1rvzbk6{overflow:auto}.file.svelte-1rvzbk6.svelte-1rvzbk6{display:flex;width:var(--size-full)}.file.svelte-1rvzbk6>.svelte-1rvzbk6{padding:var(--size-1) var(--size-2-5)}.filename.svelte-1rvzbk6.svelte-1rvzbk6{flex-grow:1;display:flex;overflow:hidden}.filename.svelte-1rvzbk6 .stem.svelte-1rvzbk6{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.filename.svelte-1rvzbk6 .ext.svelte-1rvzbk6{white-space:nowrap}.download.svelte-1rvzbk6.svelte-1rvzbk6{min-width:8rem;width:10%;white-space:nowrap;text-align:right}.download.svelte-1rvzbk6.svelte-1rvzbk6:hover{text-decoration:underline}.download.svelte-1rvzbk6>a{color:var(--link-text-color)}.download.svelte-1rvzbk6>a:hover{color:var(--link-text-color-hover)}.download.svelte-1rvzbk6>a:visited{color:var(--link-text-color-visited)}.download.svelte-1rvzbk6>a:active{color:var(--link-text-color-active)}.selectable.svelte-1rvzbk6.svelte-1rvzbk6{cursor:pointer}tbody.svelte-1rvzbk6>tr.svelte-1rvzbk6:nth-child(even){background:var(--block-background-fill)}tbody.svelte-1rvzbk6>tr.svelte-1rvzbk6:nth-child(odd){background:var(--table-odd-background-fill)}.drag-handle.svelte-1rvzbk6.svelte-1rvzbk6{cursor:grab;color:var(--body-text-color-subdued);padding-right:var(--size-2);user-select:none}.dragging.svelte-1rvzbk6.svelte-1rvzbk6{opacity:0.5;cursor:grabbing}.drop-target.svelte-1rvzbk6.svelte-1rvzbk6{border-top:2px solid var(--color-accent)}tr:last-child.drop-target[data-drop-target="before"].svelte-1rvzbk6.svelte-1rvzbk6{border-top:2px solid var(--color-accent);border-bottom:none}tr:last-child.drop-target[data-drop-target="after"].svelte-1rvzbk6.svelte-1rvzbk6{border-top:none;border-bottom:2px solid var(--color-accent)}',
  map: '{"version":3,"file":"FilePreview.svelte","sources":["FilePreview.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { prettyBytes } from \\"./utils\\";\\nimport { createEventDispatcher } from \\"svelte\\";\\nimport { DownloadLink } from \\"@gradio/wasm/svelte\\";\\nconst dispatch = createEventDispatcher();\\nexport let value;\\nexport let selectable = false;\\nexport let height = void 0;\\nexport let i18n;\\nexport let allow_reordering = false;\\nlet dragging_index = null;\\nlet drop_target_index = null;\\nfunction handle_drag_start(event, index) {\\n    dragging_index = index;\\n    if (event.dataTransfer) {\\n        event.dataTransfer.effectAllowed = \\"move\\";\\n        event.dataTransfer.setData(\\"text/plain\\", index.toString());\\n    }\\n}\\nfunction handle_drag_over(event, index) {\\n    event.preventDefault();\\n    if (index === normalized_files.length - 1) {\\n        const rect = event.currentTarget.getBoundingClientRect();\\n        const midY = rect.top + rect.height / 2;\\n        drop_target_index = event.clientY > midY ? normalized_files.length : index;\\n    }\\n    else {\\n        drop_target_index = index;\\n    }\\n    if (event.dataTransfer) {\\n        event.dataTransfer.dropEffect = \\"move\\";\\n    }\\n}\\nfunction handle_drag_end(event) {\\n    if (!event.dataTransfer?.dropEffect || event.dataTransfer.dropEffect === \\"none\\") {\\n        dragging_index = null;\\n        drop_target_index = null;\\n    }\\n}\\nfunction handle_drop(event, index) {\\n    event.preventDefault();\\n    if (dragging_index === null || dragging_index === index)\\n        return;\\n    const files = Array.isArray(value) ? [...value] : [value];\\n    const [removed] = files.splice(dragging_index, 1);\\n    files.splice(drop_target_index === normalized_files.length ? normalized_files.length : index, 0, removed);\\n    const new_value = Array.isArray(value) ? files : files[0];\\n    dispatch(\\"change\\", new_value);\\n    dragging_index = null;\\n    drop_target_index = null;\\n}\\nfunction split_filename(filename) {\\n    const last_dot = filename.lastIndexOf(\\".\\");\\n    if (last_dot === -1) {\\n        return [filename, \\"\\"];\\n    }\\n    return [filename.slice(0, last_dot), filename.slice(last_dot)];\\n}\\n$: normalized_files = (Array.isArray(value) ? value : [value]).map((file) => {\\n    const [filename_stem, filename_ext] = split_filename(file.orig_name ?? \\"\\");\\n    return {\\n        ...file,\\n        filename_stem,\\n        filename_ext\\n    };\\n});\\nfunction handle_row_click(event, index) {\\n    const tr = event.currentTarget;\\n    const should_select = event.target === tr || // Only select if the click is on the row itself\\n        tr && tr.firstElementChild && event.composedPath().includes(tr.firstElementChild);\\n    if (should_select) {\\n        dispatch(\\"select\\", { value: normalized_files[index].orig_name, index });\\n    }\\n}\\nfunction remove_file(index) {\\n    const removed = normalized_files.splice(index, 1);\\n    normalized_files = [...normalized_files];\\n    value = normalized_files;\\n    dispatch(\\"delete\\", removed[0]);\\n    dispatch(\\"change\\", normalized_files);\\n}\\nfunction handle_download(file) {\\n    dispatch(\\"download\\", file);\\n}\\nconst is_browser = typeof window !== \\"undefined\\";\\n<\/script>\\n\\n<div\\n\\tclass=\\"file-preview-holder\\"\\n\\tstyle:max-height={height\\n\\t\\t? typeof height === \\"number\\"\\n\\t\\t\\t? height + \\"px\\"\\n\\t\\t\\t: height\\n\\t\\t: \\"auto\\"}\\n>\\n\\t<table class=\\"file-preview\\">\\n\\t\\t<tbody>\\n\\t\\t\\t{#each normalized_files as file, i (file.url)}\\n\\t\\t\\t\\t<tr\\n\\t\\t\\t\\t\\tclass=\\"file\\"\\n\\t\\t\\t\\t\\tclass:selectable\\n\\t\\t\\t\\t\\tclass:dragging={dragging_index === i}\\n\\t\\t\\t\\t\\tclass:drop-target={drop_target_index === i ||\\n\\t\\t\\t\\t\\t\\t(i === normalized_files.length - 1 &&\\n\\t\\t\\t\\t\\t\\t\\tdrop_target_index === normalized_files.length)}\\n\\t\\t\\t\\t\\tdata-drop-target={drop_target_index === normalized_files.length &&\\n\\t\\t\\t\\t\\ti === normalized_files.length - 1\\n\\t\\t\\t\\t\\t\\t? \\"after\\"\\n\\t\\t\\t\\t\\t\\t: drop_target_index === i + 1\\n\\t\\t\\t\\t\\t\\t\\t? \\"after\\"\\n\\t\\t\\t\\t\\t\\t\\t: \\"before\\"}\\n\\t\\t\\t\\t\\tdraggable={allow_reordering && normalized_files.length > 1}\\n\\t\\t\\t\\t\\ton:click={(event) => {\\n\\t\\t\\t\\t\\t\\thandle_row_click(event, i);\\n\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\ton:dragstart={(event) => handle_drag_start(event, i)}\\n\\t\\t\\t\\t\\ton:dragenter|preventDefault\\n\\t\\t\\t\\t\\ton:dragover={(event) => handle_drag_over(event, i)}\\n\\t\\t\\t\\t\\ton:drop={(event) => handle_drop(event, i)}\\n\\t\\t\\t\\t\\ton:dragend={handle_drag_end}\\n\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t<td class=\\"filename\\" aria-label={file.orig_name}>\\n\\t\\t\\t\\t\\t\\t{#if allow_reordering && normalized_files.length > 1}\\n\\t\\t\\t\\t\\t\\t\\t<span class=\\"drag-handle\\">⋮⋮</span>\\n\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t\\t<span class=\\"stem\\">{file.filename_stem}</span>\\n\\t\\t\\t\\t\\t\\t<span class=\\"ext\\">{file.filename_ext}</span>\\n\\t\\t\\t\\t\\t</td>\\n\\n\\t\\t\\t\\t\\t<td class=\\"download\\">\\n\\t\\t\\t\\t\\t\\t{#if file.url}\\n\\t\\t\\t\\t\\t\\t\\t<DownloadLink\\n\\t\\t\\t\\t\\t\\t\\t\\thref={file.url}\\n\\t\\t\\t\\t\\t\\t\\t\\ton:click={() => handle_download(file)}\\n\\t\\t\\t\\t\\t\\t\\t\\tdownload={is_browser && window.__is_colab__\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t? null\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t: file.orig_name}\\n\\t\\t\\t\\t\\t\\t\\t>\\n\\t\\t\\t\\t\\t\\t\\t\\t{@html file.size != null\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t? prettyBytes(file.size)\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t: \\"(size unknown)\\"}&nbsp;&#8675;\\n\\t\\t\\t\\t\\t\\t\\t</DownloadLink>\\n\\t\\t\\t\\t\\t\\t{:else}\\n\\t\\t\\t\\t\\t\\t\\t{i18n(\\"file.uploading\\")}\\n\\t\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t\\t</td>\\n\\n\\t\\t\\t\\t\\t{#if normalized_files.length > 1}\\n\\t\\t\\t\\t\\t\\t<td>\\n\\t\\t\\t\\t\\t\\t\\t<button\\n\\t\\t\\t\\t\\t\\t\\t\\tclass=\\"label-clear-button\\"\\n\\t\\t\\t\\t\\t\\t\\t\\taria-label=\\"Remove this file\\"\\n\\t\\t\\t\\t\\t\\t\\t\\ton:click={() => {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tremove_file(i);\\n\\t\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\t\\ton:keydown={(event) => {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\tif (event.key === \\"Enter\\") {\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t\\tremove_file(i);\\n\\t\\t\\t\\t\\t\\t\\t\\t\\t}\\n\\t\\t\\t\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t\\t\\t\\t>×\\n\\t\\t\\t\\t\\t\\t\\t</button>\\n\\t\\t\\t\\t\\t\\t</td>\\n\\t\\t\\t\\t\\t{/if}\\n\\t\\t\\t\\t</tr>\\n\\t\\t\\t{/each}\\n\\t\\t</tbody>\\n\\t</table>\\n</div>\\n\\n<style>\\n\\t.label-clear-button {\\n\\t\\tcolor: var(--body-text-color-subdued);\\n\\t\\tposition: relative;\\n\\t\\tleft: -3px;\\n\\t}\\n\\n\\t.label-clear-button:hover {\\n\\t\\tcolor: var(--body-text-color);\\n\\t}\\n\\n\\t.file-preview {\\n\\t\\ttable-layout: fixed;\\n\\t\\twidth: var(--size-full);\\n\\t\\tmax-height: var(--size-60);\\n\\t\\toverflow-y: auto;\\n\\t\\tmargin-top: var(--size-1);\\n\\t\\tcolor: var(--body-text-color);\\n\\t}\\n\\n\\t.file-preview-holder {\\n\\t\\toverflow: auto;\\n\\t}\\n\\n\\t.file {\\n\\t\\tdisplay: flex;\\n\\t\\twidth: var(--size-full);\\n\\t}\\n\\n\\t.file > * {\\n\\t\\tpadding: var(--size-1) var(--size-2-5);\\n\\t}\\n\\n\\t.filename {\\n\\t\\tflex-grow: 1;\\n\\t\\tdisplay: flex;\\n\\t\\toverflow: hidden;\\n\\t}\\n\\t.filename .stem {\\n\\t\\toverflow: hidden;\\n\\t\\ttext-overflow: ellipsis;\\n\\t\\twhite-space: nowrap;\\n\\t}\\n\\t.filename .ext {\\n\\t\\twhite-space: nowrap;\\n\\t}\\n\\n\\t.download {\\n\\t\\tmin-width: 8rem;\\n\\t\\twidth: 10%;\\n\\t\\twhite-space: nowrap;\\n\\t\\ttext-align: right;\\n\\t}\\n\\t.download:hover {\\n\\t\\ttext-decoration: underline;\\n\\t}\\n\\t.download > :global(a) {\\n\\t\\tcolor: var(--link-text-color);\\n\\t}\\n\\n\\t.download > :global(a:hover) {\\n\\t\\tcolor: var(--link-text-color-hover);\\n\\t}\\n\\t.download > :global(a:visited) {\\n\\t\\tcolor: var(--link-text-color-visited);\\n\\t}\\n\\t.download > :global(a:active) {\\n\\t\\tcolor: var(--link-text-color-active);\\n\\t}\\n\\t.selectable {\\n\\t\\tcursor: pointer;\\n\\t}\\n\\n\\ttbody > tr:nth-child(even) {\\n\\t\\tbackground: var(--block-background-fill);\\n\\t}\\n\\n\\ttbody > tr:nth-child(odd) {\\n\\t\\tbackground: var(--table-odd-background-fill);\\n\\t}\\n\\n\\t.drag-handle {\\n\\t\\tcursor: grab;\\n\\t\\tcolor: var(--body-text-color-subdued);\\n\\t\\tpadding-right: var(--size-2);\\n\\t\\tuser-select: none;\\n\\t}\\n\\n\\t.dragging {\\n\\t\\topacity: 0.5;\\n\\t\\tcursor: grabbing;\\n\\t}\\n\\n\\t.drop-target {\\n\\t\\tborder-top: 2px solid var(--color-accent);\\n\\t}\\n\\n\\ttr:last-child.drop-target[data-drop-target=\\"before\\"] {\\n\\t\\tborder-top: 2px solid var(--color-accent);\\n\\t\\tborder-bottom: none;\\n\\t}\\n\\n\\ttr:last-child.drop-target[data-drop-target=\\"after\\"] {\\n\\t\\tborder-top: none;\\n\\t\\tborder-bottom: 2px solid var(--color-accent);\\n\\t}</style>\\n"],"names":[],"mappings":"AA0KC,iDAAoB,CACnB,KAAK,CAAE,IAAI,yBAAyB,CAAC,CACrC,QAAQ,CAAE,QAAQ,CAClB,IAAI,CAAE,IACP,CAEA,iDAAmB,MAAO,CACzB,KAAK,CAAE,IAAI,iBAAiB,CAC7B,CAEA,2CAAc,CACb,YAAY,CAAE,KAAK,CACnB,KAAK,CAAE,IAAI,WAAW,CAAC,CACvB,UAAU,CAAE,IAAI,SAAS,CAAC,CAC1B,UAAU,CAAE,IAAI,CAChB,UAAU,CAAE,IAAI,QAAQ,CAAC,CACzB,KAAK,CAAE,IAAI,iBAAiB,CAC7B,CAEA,kDAAqB,CACpB,QAAQ,CAAE,IACX,CAEA,mCAAM,CACL,OAAO,CAAE,IAAI,CACb,KAAK,CAAE,IAAI,WAAW,CACvB,CAEA,oBAAK,CAAG,eAAE,CACT,OAAO,CAAE,IAAI,QAAQ,CAAC,CAAC,IAAI,UAAU,CACtC,CAEA,uCAAU,CACT,SAAS,CAAE,CAAC,CACZ,OAAO,CAAE,IAAI,CACb,QAAQ,CAAE,MACX,CACA,wBAAS,CAAC,oBAAM,CACf,QAAQ,CAAE,MAAM,CAChB,aAAa,CAAE,QAAQ,CACvB,WAAW,CAAE,MACd,CACA,wBAAS,CAAC,mBAAK,CACd,WAAW,CAAE,MACd,CAEA,uCAAU,CACT,SAAS,CAAE,IAAI,CACf,KAAK,CAAE,GAAG,CACV,WAAW,CAAE,MAAM,CACnB,UAAU,CAAE,KACb,CACA,uCAAS,MAAO,CACf,eAAe,CAAE,SAClB,CACA,wBAAS,CAAW,CAAG,CACtB,KAAK,CAAE,IAAI,iBAAiB,CAC7B,CAEA,wBAAS,CAAW,OAAS,CAC5B,KAAK,CAAE,IAAI,uBAAuB,CACnC,CACA,wBAAS,CAAW,SAAW,CAC9B,KAAK,CAAE,IAAI,yBAAyB,CACrC,CACA,wBAAS,CAAW,QAAU,CAC7B,KAAK,CAAE,IAAI,wBAAwB,CACpC,CACA,yCAAY,CACX,MAAM,CAAE,OACT,CAEA,oBAAK,CAAG,iBAAE,WAAW,IAAI,CAAE,CAC1B,UAAU,CAAE,IAAI,uBAAuB,CACxC,CAEA,oBAAK,CAAG,iBAAE,WAAW,GAAG,CAAE,CACzB,UAAU,CAAE,IAAI,2BAA2B,CAC5C,CAEA,0CAAa,CACZ,MAAM,CAAE,IAAI,CACZ,KAAK,CAAE,IAAI,yBAAyB,CAAC,CACrC,aAAa,CAAE,IAAI,QAAQ,CAAC,CAC5B,WAAW,CAAE,IACd,CAEA,uCAAU,CACT,OAAO,CAAE,GAAG,CACZ,MAAM,CAAE,QACT,CAEA,0CAAa,CACZ,UAAU,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,cAAc,CACzC,CAEA,EAAE,WAAW,YAAY,CAAC,gBAAgB,CAAC,QAAQ,+BAAE,CACpD,UAAU,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,cAAc,CAAC,CACzC,aAAa,CAAE,IAChB,CAEA,EAAE,WAAW,YAAY,CAAC,gBAAgB,CAAC,OAAO,+BAAE,CACnD,UAAU,CAAE,IAAI,CAChB,aAAa,CAAE,GAAG,CAAC,KAAK,CAAC,IAAI,cAAc,CAC5C"}'
};
function split_filename(filename) {
  const last_dot = filename.lastIndexOf(".");
  if (last_dot === -1) {
    return [filename, ""];
  }
  return [filename.slice(0, last_dot), filename.slice(last_dot)];
}
const FilePreview = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let normalized_files;
  createEventDispatcher();
  let { value } = $$props;
  let { selectable = false } = $$props;
  let { height = void 0 } = $$props;
  let { i18n } = $$props;
  let { allow_reordering = false } = $$props;
  let dragging_index = null;
  let drop_target_index = null;
  const is_browser = typeof window !== "undefined";
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.selectable === void 0 && $$bindings.selectable && selectable !== void 0)
    $$bindings.selectable(selectable);
  if ($$props.height === void 0 && $$bindings.height && height !== void 0)
    $$bindings.height(height);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.allow_reordering === void 0 && $$bindings.allow_reordering && allow_reordering !== void 0)
    $$bindings.allow_reordering(allow_reordering);
  $$result.css.add(css);
  normalized_files = (Array.isArray(value) ? value : [value]).map((file) => {
    const [filename_stem, filename_ext] = split_filename(file.orig_name ?? "");
    return { ...file, filename_stem, filename_ext };
  });
  return `<div class="file-preview-holder svelte-1rvzbk6"${add_styles({
    "max-height": height ? typeof height === "number" ? height + "px" : height : "auto"
  })}><table class="file-preview svelte-1rvzbk6"><tbody class="svelte-1rvzbk6">${each(normalized_files, (file, i) => {
    return `<tr class="${[
      "file svelte-1rvzbk6",
      (selectable ? "selectable" : "") + " " + (dragging_index === i ? "dragging" : "") + " " + (drop_target_index === i || i === normalized_files.length - 1 && drop_target_index === normalized_files.length ? "drop-target" : "")
    ].join(" ").trim()}"${add_attribute(
      "data-drop-target",
      drop_target_index === normalized_files.length && i === normalized_files.length - 1 ? "after" : drop_target_index === i + 1 ? "after" : "before",
      0
    )}${add_attribute("draggable", allow_reordering && normalized_files.length > 1, 0)}><td class="filename svelte-1rvzbk6"${add_attribute("aria-label", file.orig_name, 0)}>${allow_reordering && normalized_files.length > 1 ? `<span class="drag-handle svelte-1rvzbk6" data-svelte-h="svelte-1u4up0a">⋮⋮</span>` : ``} <span class="stem svelte-1rvzbk6">${escape(file.filename_stem)}</span> <span class="ext svelte-1rvzbk6">${escape(file.filename_ext)}</span></td> <td class="download svelte-1rvzbk6">${file.url ? `${validate_component(DownloadLink, "DownloadLink").$$render(
      $$result,
      {
        href: file.url,
        download: is_browser && window.__is_colab__ ? null : file.orig_name
      },
      {},
      {
        default: () => {
          return `<!-- HTML_TAG_START -->${file.size != null ? prettyBytes(file.size) : "(size unknown)"}<!-- HTML_TAG_END --> ⇣
							`;
        }
      }
    )}` : `${escape(i18n("file.uploading"))}`}</td> ${normalized_files.length > 1 ? `<td class="svelte-1rvzbk6"><button class="label-clear-button svelte-1rvzbk6" aria-label="Remove this file" data-svelte-h="svelte-nhtord">×</button> </td>` : ``} </tr>`;
  })}</tbody></table> </div>`;
});
const FilePreview$1 = FilePreview;
const File_1 = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value = null } = $$props;
  let { label } = $$props;
  let { show_label = true } = $$props;
  let { selectable = false } = $$props;
  let { height = void 0 } = $$props;
  let { i18n } = $$props;
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.selectable === void 0 && $$bindings.selectable && selectable !== void 0)
    $$bindings.selectable(selectable);
  if ($$props.height === void 0 && $$bindings.height && height !== void 0)
    $$bindings.height(height);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  return `${validate_component(BlockLabel, "BlockLabel").$$render(
    $$result,
    {
      show_label,
      float: value === null,
      Icon: File$1,
      label: label || "File"
    },
    {},
    {}
  )} ${value && (Array.isArray(value) ? value.length > 0 : true) ? `${validate_component(FilePreview$1, "FilePreview").$$render($$result, { i18n, selectable, value, height }, {}, {})}` : `${validate_component(Empty, "Empty").$$render($$result, { unpadded_box: true, size: "large" }, {}, {
    default: () => {
      return `${validate_component(File$1, "File").$$render($$result, {}, {}, {})}`;
    }
  })}`}`;
});
const File = File_1;
const FileUpload = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value } = $$props;
  let { label } = $$props;
  let { show_label = true } = $$props;
  let { file_count = "single" } = $$props;
  let { file_types = null } = $$props;
  let { selectable = false } = $$props;
  let { root } = $$props;
  let { height = void 0 } = $$props;
  let { i18n } = $$props;
  let { max_file_size = null } = $$props;
  let { upload } = $$props;
  let { stream_handler } = $$props;
  let { uploading = false } = $$props;
  let { allow_reordering = false } = $$props;
  const dispatch = createEventDispatcher();
  let dragging = false;
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.file_count === void 0 && $$bindings.file_count && file_count !== void 0)
    $$bindings.file_count(file_count);
  if ($$props.file_types === void 0 && $$bindings.file_types && file_types !== void 0)
    $$bindings.file_types(file_types);
  if ($$props.selectable === void 0 && $$bindings.selectable && selectable !== void 0)
    $$bindings.selectable(selectable);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props.height === void 0 && $$bindings.height && height !== void 0)
    $$bindings.height(height);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.max_file_size === void 0 && $$bindings.max_file_size && max_file_size !== void 0)
    $$bindings.max_file_size(max_file_size);
  if ($$props.upload === void 0 && $$bindings.upload && upload !== void 0)
    $$bindings.upload(upload);
  if ($$props.stream_handler === void 0 && $$bindings.stream_handler && stream_handler !== void 0)
    $$bindings.stream_handler(stream_handler);
  if ($$props.uploading === void 0 && $$bindings.uploading && uploading !== void 0)
    $$bindings.uploading(uploading);
  if ($$props.allow_reordering === void 0 && $$bindings.allow_reordering && allow_reordering !== void 0)
    $$bindings.allow_reordering(allow_reordering);
  let $$settled;
  let $$rendered;
  let previous_head = $$result.head;
  do {
    $$settled = true;
    $$result.head = previous_head;
    {
      dispatch("drag", dragging);
    }
    $$rendered = `${validate_component(BlockLabel, "BlockLabel").$$render(
      $$result,
      {
        show_label,
        Icon: File$1,
        float: !value,
        label: label || "File"
      },
      {},
      {}
    )} ${value && (Array.isArray(value) ? value.length > 0 : true) ? `${validate_component(IconButtonWrapper, "IconButtonWrapper").$$render($$result, {}, {}, {
      default: () => {
        return `${!(file_count === "single" && (Array.isArray(value) ? value.length > 0 : value !== null)) ? `${validate_component(IconButton, "IconButton").$$render(
          $$result,
          {
            Icon: Upload,
            label: i18n("common.upload")
          },
          {},
          {
            default: () => {
              return `${validate_component(Upload$1, "Upload").$$render(
                $$result,
                {
                  icon_upload: true,
                  filetype: file_types,
                  file_count,
                  max_file_size,
                  root,
                  stream_handler,
                  upload,
                  dragging,
                  uploading
                },
                {
                  dragging: ($$value) => {
                    dragging = $$value;
                    $$settled = false;
                  },
                  uploading: ($$value) => {
                    uploading = $$value;
                    $$settled = false;
                  }
                },
                {}
              )}`;
            }
          }
        )}` : ``} ${validate_component(IconButton, "IconButton").$$render($$result, { Icon: Clear, label: i18n("common.clear") }, {}, {})}`;
      }
    })} ${validate_component(FilePreview$1, "FilePreview").$$render(
      $$result,
      {
        i18n,
        selectable,
        value,
        height,
        allow_reordering
      },
      {},
      {}
    )}` : `${validate_component(Upload$1, "Upload").$$render(
      $$result,
      {
        filetype: file_types,
        file_count,
        max_file_size,
        root,
        stream_handler,
        upload,
        height,
        dragging,
        uploading
      },
      {
        dragging: ($$value) => {
          dragging = $$value;
          $$settled = false;
        },
        uploading: ($$value) => {
          uploading = $$value;
          $$settled = false;
        }
      },
      {
        default: () => {
          return `${slots.default ? slots.default({}) : ``}`;
        }
      }
    )}`}`;
  } while (!$$settled);
  return $$rendered;
});
const BaseFileUpload = FileUpload;
const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value } = $$props;
  let { interactive } = $$props;
  let { root } = $$props;
  let { label } = $$props;
  let { show_label } = $$props;
  let { height = void 0 } = $$props;
  let { _selectable = false } = $$props;
  let { loading_status } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { gradio } = $$props;
  let { file_count } = $$props;
  let { file_types = ["file"] } = $$props;
  let { input_ready } = $$props;
  let { allow_reordering = false } = $$props;
  let uploading = false;
  let old_value = value;
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.interactive === void 0 && $$bindings.interactive && interactive !== void 0)
    $$bindings.interactive(interactive);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.height === void 0 && $$bindings.height && height !== void 0)
    $$bindings.height(height);
  if ($$props._selectable === void 0 && $$bindings._selectable && _selectable !== void 0)
    $$bindings._selectable(_selectable);
  if ($$props.loading_status === void 0 && $$bindings.loading_status && loading_status !== void 0)
    $$bindings.loading_status(loading_status);
  if ($$props.container === void 0 && $$bindings.container && container !== void 0)
    $$bindings.container(container);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  if ($$props.min_width === void 0 && $$bindings.min_width && min_width !== void 0)
    $$bindings.min_width(min_width);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  if ($$props.file_count === void 0 && $$bindings.file_count && file_count !== void 0)
    $$bindings.file_count(file_count);
  if ($$props.file_types === void 0 && $$bindings.file_types && file_types !== void 0)
    $$bindings.file_types(file_types);
  if ($$props.input_ready === void 0 && $$bindings.input_ready && input_ready !== void 0)
    $$bindings.input_ready(input_ready);
  if ($$props.allow_reordering === void 0 && $$bindings.allow_reordering && allow_reordering !== void 0)
    $$bindings.allow_reordering(allow_reordering);
  let $$settled;
  let $$rendered;
  let previous_head = $$result.head;
  do {
    $$settled = true;
    $$result.head = previous_head;
    input_ready = !uploading;
    {
      if (JSON.stringify(old_value) !== JSON.stringify(value)) {
        gradio.dispatch("change");
        old_value = value;
      }
    }
    $$rendered = `   ${validate_component(Block, "Block").$$render(
      $$result,
      {
        visible,
        variant: value ? "solid" : "dashed",
        border_mode: "base",
        padding: false,
        elem_id,
        elem_classes,
        container,
        scale,
        min_width,
        allow_overflow: false
      },
      {},
      {
        default: () => {
          return `${validate_component(Static, "StatusTracker").$$render(
            $$result,
            Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status, {
              status: loading_status?.status || "complete"
            }),
            {},
            {}
          )} ${!interactive ? `${validate_component(File, "File").$$render(
            $$result,
            {
              selectable: _selectable,
              value,
              label,
              show_label,
              height,
              i18n: gradio.i18n
            },
            {},
            {}
          )}` : `${validate_component(BaseFileUpload, "FileUpload").$$render(
            $$result,
            {
              upload: (...args) => gradio.client.upload(...args),
              stream_handler: (...args) => gradio.client.stream(...args),
              label,
              show_label,
              value,
              file_count,
              file_types,
              selectable: _selectable,
              root,
              height,
              allow_reordering,
              max_file_size: gradio.max_file_size,
              i18n: gradio.i18n,
              uploading
            },
            {
              uploading: ($$value) => {
                uploading = $$value;
                $$settled = false;
              }
            },
            {
              default: () => {
                return `${validate_component(UploadText, "UploadText").$$render($$result, { i18n: gradio.i18n, type: "file" }, {}, {})}`;
              }
            }
          )}`}`;
        }
      }
    )}`;
  } while (!$$settled);
  return $$rendered;
});

export { File as BaseFile, BaseFileUpload, FilePreview$1 as FilePreview, Index as default };
//# sourceMappingURL=Index8-BJaxilKd.js.map
