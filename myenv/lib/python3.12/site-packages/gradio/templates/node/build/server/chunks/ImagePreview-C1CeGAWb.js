import { c as create_ssr_component, a as createEventDispatcher, v as validate_component, b as add_attribute } from './ssr-fyTaU2Wq.js';
import { f as BlockLabel, g as Image, h as Empty, i as IconButtonWrapper, F as FullscreenButton, j as IconButton, D as Download, k as ShareButton, u as uploadToHuggingFace } from './2-CnaXPGyd.js';
import { I as Image$1 } from './Image-BwObd70i.js';
import { D as DownloadLink } from './DownloadLink-BKK_IWmU.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BeHry4b7.js';

const css = {
  code: ".image-container.svelte-dpdy90.svelte-dpdy90{height:100%;position:relative;min-width:var(--size-20)}.image-container.svelte-dpdy90 button.svelte-dpdy90{width:var(--size-full);height:var(--size-full);border-radius:var(--radius-lg);display:flex;align-items:center;justify-content:center}.image-frame.svelte-dpdy90.svelte-dpdy90{width:auto;height:100%;display:flex;align-items:center;justify-content:center}.image-frame.svelte-dpdy90 img{width:var(--size-full);height:var(--size-full);object-fit:scale-down}.selectable.svelte-dpdy90.svelte-dpdy90{cursor:crosshair}.fullscreen-controls svg{position:relative;top:0px}.image-container:fullscreen{background-color:black;display:flex;justify-content:center;align-items:center}.image-container:fullscreen img{max-width:90vw;max-height:90vh;object-fit:scale-down}",
  map: '{"version":3,"file":"ImagePreview.svelte","sources":["ImagePreview.svelte"],"sourcesContent":["<script lang=\\"ts\\">import { createEventDispatcher, onMount } from \\"svelte\\";\\nimport { uploadToHuggingFace } from \\"@gradio/utils\\";\\nimport { BlockLabel, Empty, IconButton, ShareButton, IconButtonWrapper, FullscreenButton } from \\"@gradio/atoms\\";\\nimport { Download, Image as ImageIcon } from \\"@gradio/icons\\";\\nimport { get_coordinates_of_clicked_image } from \\"./utils\\";\\nimport { Image } from \\"@gradio/image/shared\\";\\nimport { DownloadLink } from \\"@gradio/wasm/svelte\\";\\nexport let value;\\nexport let label = void 0;\\nexport let show_label;\\nexport let show_download_button = true;\\nexport let selectable = false;\\nexport let show_share_button = false;\\nexport let i18n;\\nexport let show_fullscreen_button = true;\\nexport let display_icon_button_wrapper_top_corner = false;\\nconst dispatch = createEventDispatcher();\\nconst handle_click = (evt) => {\\n    let coordinates = get_coordinates_of_clicked_image(evt);\\n    if (coordinates) {\\n        dispatch(\\"select\\", { index: coordinates, value: null });\\n    }\\n};\\nlet image_container;\\n<\/script>\\n\\n<BlockLabel\\n\\t{show_label}\\n\\tIcon={ImageIcon}\\n\\tlabel={!show_label ? \\"\\" : label || i18n(\\"image.image\\")}\\n/>\\n{#if value === null || !value.url}\\n\\t<Empty unpadded_box={true} size=\\"large\\"><ImageIcon /></Empty>\\n{:else}\\n\\t<div class=\\"image-container\\" bind:this={image_container}>\\n\\t\\t<IconButtonWrapper\\n\\t\\t\\tdisplay_top_corner={display_icon_button_wrapper_top_corner}\\n\\t\\t>\\n\\t\\t\\t{#if show_fullscreen_button}\\n\\t\\t\\t\\t<FullscreenButton container={image_container} />\\n\\t\\t\\t{/if}\\n\\n\\t\\t\\t{#if show_download_button}\\n\\t\\t\\t\\t<DownloadLink href={value.url} download={value.orig_name || \\"image\\"}>\\n\\t\\t\\t\\t\\t<IconButton Icon={Download} label={i18n(\\"common.download\\")} />\\n\\t\\t\\t\\t</DownloadLink>\\n\\t\\t\\t{/if}\\n\\t\\t\\t{#if show_share_button}\\n\\t\\t\\t\\t<ShareButton\\n\\t\\t\\t\\t\\t{i18n}\\n\\t\\t\\t\\t\\ton:share\\n\\t\\t\\t\\t\\ton:error\\n\\t\\t\\t\\t\\tformatter={async (value) => {\\n\\t\\t\\t\\t\\t\\tif (!value) return \\"\\";\\n\\t\\t\\t\\t\\t\\tlet url = await uploadToHuggingFace(value, \\"url\\");\\n\\t\\t\\t\\t\\t\\treturn `<img src=\\"${url}\\" />`;\\n\\t\\t\\t\\t\\t}}\\n\\t\\t\\t\\t\\t{value}\\n\\t\\t\\t\\t/>\\n\\t\\t\\t{/if}\\n\\t\\t</IconButtonWrapper>\\n\\t\\t<button on:click={handle_click}>\\n\\t\\t\\t<div class:selectable class=\\"image-frame\\">\\n\\t\\t\\t\\t<Image src={value.url} alt=\\"\\" loading=\\"lazy\\" on:load />\\n\\t\\t\\t</div>\\n\\t\\t</button>\\n\\t</div>\\n{/if}\\n\\n<style>\\n\\t.image-container {\\n\\t\\theight: 100%;\\n\\t\\tposition: relative;\\n\\t\\tmin-width: var(--size-20);\\n\\t}\\n\\n\\t.image-container button {\\n\\t\\twidth: var(--size-full);\\n\\t\\theight: var(--size-full);\\n\\t\\tborder-radius: var(--radius-lg);\\n\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t}\\n\\n\\t.image-frame {\\n\\t\\twidth: auto;\\n\\t\\theight: 100%;\\n\\t\\tdisplay: flex;\\n\\t\\talign-items: center;\\n\\t\\tjustify-content: center;\\n\\t}\\n\\t.image-frame :global(img) {\\n\\t\\twidth: var(--size-full);\\n\\t\\theight: var(--size-full);\\n\\t\\tobject-fit: scale-down;\\n\\t}\\n\\n\\t.selectable {\\n\\t\\tcursor: crosshair;\\n\\t}\\n\\n\\t:global(.fullscreen-controls svg) {\\n\\t\\tposition: relative;\\n\\t\\ttop: 0px;\\n\\t}\\n\\n\\t:global(.image-container:fullscreen) {\\n\\t\\tbackground-color: black;\\n\\t\\tdisplay: flex;\\n\\t\\tjustify-content: center;\\n\\t\\talign-items: center;\\n\\t}\\n\\n\\t:global(.image-container:fullscreen img) {\\n\\t\\tmax-width: 90vw;\\n\\t\\tmax-height: 90vh;\\n\\t\\tobject-fit: scale-down;\\n\\t}</style>\\n"],"names":[],"mappings":"AAsEC,4CAAiB,CAChB,MAAM,CAAE,IAAI,CACZ,QAAQ,CAAE,QAAQ,CAClB,SAAS,CAAE,IAAI,SAAS,CACzB,CAEA,8BAAgB,CAAC,oBAAO,CACvB,KAAK,CAAE,IAAI,WAAW,CAAC,CACvB,MAAM,CAAE,IAAI,WAAW,CAAC,CACxB,aAAa,CAAE,IAAI,WAAW,CAAC,CAE/B,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,eAAe,CAAE,MAClB,CAEA,wCAAa,CACZ,KAAK,CAAE,IAAI,CACX,MAAM,CAAE,IAAI,CACZ,OAAO,CAAE,IAAI,CACb,WAAW,CAAE,MAAM,CACnB,eAAe,CAAE,MAClB,CACA,0BAAY,CAAS,GAAK,CACzB,KAAK,CAAE,IAAI,WAAW,CAAC,CACvB,MAAM,CAAE,IAAI,WAAW,CAAC,CACxB,UAAU,CAAE,UACb,CAEA,uCAAY,CACX,MAAM,CAAE,SACT,CAEQ,wBAA0B,CACjC,QAAQ,CAAE,QAAQ,CAClB,GAAG,CAAE,GACN,CAEQ,2BAA6B,CACpC,gBAAgB,CAAE,KAAK,CACvB,OAAO,CAAE,IAAI,CACb,eAAe,CAAE,MAAM,CACvB,WAAW,CAAE,MACd,CAEQ,+BAAiC,CACxC,SAAS,CAAE,IAAI,CACf,UAAU,CAAE,IAAI,CAChB,UAAU,CAAE,UACb"}'
};
const ImagePreview = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value } = $$props;
  let { label = void 0 } = $$props;
  let { show_label } = $$props;
  let { show_download_button = true } = $$props;
  let { selectable = false } = $$props;
  let { show_share_button = false } = $$props;
  let { i18n } = $$props;
  let { show_fullscreen_button = true } = $$props;
  let { display_icon_button_wrapper_top_corner = false } = $$props;
  createEventDispatcher();
  let image_container;
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.show_download_button === void 0 && $$bindings.show_download_button && show_download_button !== void 0)
    $$bindings.show_download_button(show_download_button);
  if ($$props.selectable === void 0 && $$bindings.selectable && selectable !== void 0)
    $$bindings.selectable(selectable);
  if ($$props.show_share_button === void 0 && $$bindings.show_share_button && show_share_button !== void 0)
    $$bindings.show_share_button(show_share_button);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.show_fullscreen_button === void 0 && $$bindings.show_fullscreen_button && show_fullscreen_button !== void 0)
    $$bindings.show_fullscreen_button(show_fullscreen_button);
  if ($$props.display_icon_button_wrapper_top_corner === void 0 && $$bindings.display_icon_button_wrapper_top_corner && display_icon_button_wrapper_top_corner !== void 0)
    $$bindings.display_icon_button_wrapper_top_corner(display_icon_button_wrapper_top_corner);
  $$result.css.add(css);
  return `${validate_component(BlockLabel, "BlockLabel").$$render(
    $$result,
    {
      show_label,
      Icon: Image,
      label: !show_label ? "" : label || i18n("image.image")
    },
    {},
    {}
  )} ${value === null || !value.url ? `${validate_component(Empty, "Empty").$$render($$result, { unpadded_box: true, size: "large" }, {}, {
    default: () => {
      return `${validate_component(Image, "ImageIcon").$$render($$result, {}, {}, {})}`;
    }
  })}` : `<div class="image-container svelte-dpdy90"${add_attribute("this", image_container, 0)}>${validate_component(IconButtonWrapper, "IconButtonWrapper").$$render(
    $$result,
    {
      display_top_corner: display_icon_button_wrapper_top_corner
    },
    {},
    {
      default: () => {
        return `${show_fullscreen_button ? `${validate_component(FullscreenButton, "FullscreenButton").$$render($$result, { container: image_container }, {}, {})}` : ``} ${show_download_button ? `${validate_component(DownloadLink, "DownloadLink").$$render(
          $$result,
          {
            href: value.url,
            download: value.orig_name || "image"
          },
          {},
          {
            default: () => {
              return `${validate_component(IconButton, "IconButton").$$render(
                $$result,
                {
                  Icon: Download,
                  label: i18n("common.download")
                },
                {},
                {}
              )}`;
            }
          }
        )}` : ``} ${show_share_button ? `${validate_component(ShareButton, "ShareButton").$$render(
          $$result,
          {
            i18n,
            formatter: async (value2) => {
              if (!value2)
                return "";
              let url = await uploadToHuggingFace(value2);
              return `<img src="${url}" />`;
            },
            value
          },
          {},
          {}
        )}` : ``}`;
      }
    }
  )} <button class="svelte-dpdy90"><div class="${["image-frame svelte-dpdy90", selectable ? "selectable" : ""].join(" ").trim()}">${validate_component(Image$1, "Image").$$render($$result, { src: value.url, alt: "", loading: "lazy" }, {}, {})}</div></button></div>`}`;
});

export { ImagePreview as default };
//# sourceMappingURL=ImagePreview-C1CeGAWb.js.map
