import { c as create_ssr_component, a as createEventDispatcher, v as validate_component } from './ssr-fyTaU2Wq.js';
import { f as BlockLabel, M as Music, i as IconButtonWrapper, j as IconButton, D as Download, k as ShareButton, u as uploadToHuggingFace, h as Empty } from './2-CnaXPGyd.js';
import { A as AudioPlayer$1 } from './AudioPlayer-BPXpRZp-.js';
import { D as DownloadLink } from './DownloadLink-BKK_IWmU.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './Component-BeHry4b7.js';
import './hls-CrxM9YLy.js';

const StaticAudio = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value = null } = $$props;
  let { label } = $$props;
  let { show_label = true } = $$props;
  let { show_download_button = true } = $$props;
  let { show_share_button = false } = $$props;
  let { i18n } = $$props;
  let { waveform_settings = {} } = $$props;
  let { waveform_options = { show_recording_waveform: true } } = $$props;
  let { editable = true } = $$props;
  let { loop } = $$props;
  let { display_icon_button_wrapper_top_corner = false } = $$props;
  const dispatch = createEventDispatcher();
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.show_download_button === void 0 && $$bindings.show_download_button && show_download_button !== void 0)
    $$bindings.show_download_button(show_download_button);
  if ($$props.show_share_button === void 0 && $$bindings.show_share_button && show_share_button !== void 0)
    $$bindings.show_share_button(show_share_button);
  if ($$props.i18n === void 0 && $$bindings.i18n && i18n !== void 0)
    $$bindings.i18n(i18n);
  if ($$props.waveform_settings === void 0 && $$bindings.waveform_settings && waveform_settings !== void 0)
    $$bindings.waveform_settings(waveform_settings);
  if ($$props.waveform_options === void 0 && $$bindings.waveform_options && waveform_options !== void 0)
    $$bindings.waveform_options(waveform_options);
  if ($$props.editable === void 0 && $$bindings.editable && editable !== void 0)
    $$bindings.editable(editable);
  if ($$props.loop === void 0 && $$bindings.loop && loop !== void 0)
    $$bindings.loop(loop);
  if ($$props.display_icon_button_wrapper_top_corner === void 0 && $$bindings.display_icon_button_wrapper_top_corner && display_icon_button_wrapper_top_corner !== void 0)
    $$bindings.display_icon_button_wrapper_top_corner(display_icon_button_wrapper_top_corner);
  value && dispatch("change", value);
  return `${validate_component(BlockLabel, "BlockLabel").$$render(
    $$result,
    {
      show_label,
      Icon: Music,
      float: false,
      label: label || i18n("audio.audio")
    },
    {},
    {}
  )} ${value !== null ? `${validate_component(IconButtonWrapper, "IconButtonWrapper").$$render(
    $$result,
    {
      display_top_corner: display_icon_button_wrapper_top_corner
    },
    {},
    {
      default: () => {
        return `${show_download_button ? `${validate_component(DownloadLink, "DownloadLink").$$render(
          $$result,
          {
            href: value.is_stream ? value.url?.replace("playlist.m3u8", "playlist-file") : value.url,
            download: value.orig_name || value.path
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
              let url = await uploadToHuggingFace(value2.url);
              return `<audio controls src="${url}"></audio>`;
            },
            value
          },
          {},
          {}
        )}` : ``}`;
      }
    }
  )} ${validate_component(AudioPlayer$1, "AudioPlayer").$$render(
    $$result,
    {
      value,
      label,
      i18n,
      waveform_settings,
      waveform_options,
      editable,
      loop
    },
    {},
    {}
  )}` : `${validate_component(Empty, "Empty").$$render($$result, { size: "small" }, {}, {
    default: () => {
      return `${validate_component(Music, "Music").$$render($$result, {}, {}, {})}`;
    }
  })}`}`;
});

export { StaticAudio as default };
//# sourceMappingURL=StaticAudio-3T3n2fYj.js.map
