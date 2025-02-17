import { c as create_ssr_component, v as validate_component } from './ssr-fyTaU2Wq.js';
import StaticAudio from './StaticAudio-3T3n2fYj.js';
import { I as InteractiveAudio$1 } from './InteractiveAudio-CDElZSmK.js';
import { B as Block, S as Static, U as UploadText } from './2-CnaXPGyd.js';
export { A as BasePlayer } from './AudioPlayer-BPXpRZp-.js';
export { default as BaseExample } from './Example3-D4rTevcB.js';
import './DownloadLink-BKK_IWmU.js';
import './ModifyUpload-DkxchCER.js';
import './Component-BeHry4b7.js';
import './index-D2xO-t5a.js';
import 'tty';
import 'path';
import 'url';
import 'fs';
import './hls-CrxM9YLy.js';

const Index = create_ssr_component(($$result, $$props, $$bindings, slots) => {
  let { value_is_output = false } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { interactive } = $$props;
  let { value = null } = $$props;
  let { sources } = $$props;
  let { label } = $$props;
  let { root } = $$props;
  let { show_label } = $$props;
  let { container = true } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { loading_status } = $$props;
  let { autoplay = false } = $$props;
  let { loop = false } = $$props;
  let { show_download_button } = $$props;
  let { show_share_button = false } = $$props;
  let { editable = true } = $$props;
  let { waveform_options = { show_recording_waveform: true } } = $$props;
  let { pending } = $$props;
  let { streaming } = $$props;
  let { stream_every } = $$props;
  let { input_ready } = $$props;
  let { recording = false } = $$props;
  let uploading = false;
  let stream_state = "closed";
  let _modify_stream;
  function modify_stream_state(state) {
    stream_state = state;
    _modify_stream(state);
  }
  const get_stream_state = () => stream_state;
  let { set_time_limit } = $$props;
  let { gradio } = $$props;
  let old_value = null;
  let active_source;
  let initial_value = value;
  const handle_reset_value = () => {
    if (initial_value === null || value === initial_value) {
      return;
    }
    value = initial_value;
  };
  let dragging;
  let waveform_settings;
  const trim_region_settings = {
    color: waveform_options.trim_region_color,
    drag: true,
    resize: true
  };
  if ($$props.value_is_output === void 0 && $$bindings.value_is_output && value_is_output !== void 0)
    $$bindings.value_is_output(value_is_output);
  if ($$props.elem_id === void 0 && $$bindings.elem_id && elem_id !== void 0)
    $$bindings.elem_id(elem_id);
  if ($$props.elem_classes === void 0 && $$bindings.elem_classes && elem_classes !== void 0)
    $$bindings.elem_classes(elem_classes);
  if ($$props.visible === void 0 && $$bindings.visible && visible !== void 0)
    $$bindings.visible(visible);
  if ($$props.interactive === void 0 && $$bindings.interactive && interactive !== void 0)
    $$bindings.interactive(interactive);
  if ($$props.value === void 0 && $$bindings.value && value !== void 0)
    $$bindings.value(value);
  if ($$props.sources === void 0 && $$bindings.sources && sources !== void 0)
    $$bindings.sources(sources);
  if ($$props.label === void 0 && $$bindings.label && label !== void 0)
    $$bindings.label(label);
  if ($$props.root === void 0 && $$bindings.root && root !== void 0)
    $$bindings.root(root);
  if ($$props.show_label === void 0 && $$bindings.show_label && show_label !== void 0)
    $$bindings.show_label(show_label);
  if ($$props.container === void 0 && $$bindings.container && container !== void 0)
    $$bindings.container(container);
  if ($$props.scale === void 0 && $$bindings.scale && scale !== void 0)
    $$bindings.scale(scale);
  if ($$props.min_width === void 0 && $$bindings.min_width && min_width !== void 0)
    $$bindings.min_width(min_width);
  if ($$props.loading_status === void 0 && $$bindings.loading_status && loading_status !== void 0)
    $$bindings.loading_status(loading_status);
  if ($$props.autoplay === void 0 && $$bindings.autoplay && autoplay !== void 0)
    $$bindings.autoplay(autoplay);
  if ($$props.loop === void 0 && $$bindings.loop && loop !== void 0)
    $$bindings.loop(loop);
  if ($$props.show_download_button === void 0 && $$bindings.show_download_button && show_download_button !== void 0)
    $$bindings.show_download_button(show_download_button);
  if ($$props.show_share_button === void 0 && $$bindings.show_share_button && show_share_button !== void 0)
    $$bindings.show_share_button(show_share_button);
  if ($$props.editable === void 0 && $$bindings.editable && editable !== void 0)
    $$bindings.editable(editable);
  if ($$props.waveform_options === void 0 && $$bindings.waveform_options && waveform_options !== void 0)
    $$bindings.waveform_options(waveform_options);
  if ($$props.pending === void 0 && $$bindings.pending && pending !== void 0)
    $$bindings.pending(pending);
  if ($$props.streaming === void 0 && $$bindings.streaming && streaming !== void 0)
    $$bindings.streaming(streaming);
  if ($$props.stream_every === void 0 && $$bindings.stream_every && stream_every !== void 0)
    $$bindings.stream_every(stream_every);
  if ($$props.input_ready === void 0 && $$bindings.input_ready && input_ready !== void 0)
    $$bindings.input_ready(input_ready);
  if ($$props.recording === void 0 && $$bindings.recording && recording !== void 0)
    $$bindings.recording(recording);
  if ($$props.modify_stream_state === void 0 && $$bindings.modify_stream_state && modify_stream_state !== void 0)
    $$bindings.modify_stream_state(modify_stream_state);
  if ($$props.get_stream_state === void 0 && $$bindings.get_stream_state && get_stream_state !== void 0)
    $$bindings.get_stream_state(get_stream_state);
  if ($$props.set_time_limit === void 0 && $$bindings.set_time_limit && set_time_limit !== void 0)
    $$bindings.set_time_limit(set_time_limit);
  if ($$props.gradio === void 0 && $$bindings.gradio && gradio !== void 0)
    $$bindings.gradio(gradio);
  let $$settled;
  let $$rendered;
  let previous_head = $$result.head;
  do {
    $$settled = true;
    $$result.head = previous_head;
    input_ready = !uploading;
    {
      if (value && initial_value === null) {
        initial_value = value;
      }
    }
    {
      {
        if (JSON.stringify(value) !== JSON.stringify(old_value)) {
          old_value = value;
          gradio.dispatch("change");
          if (!value_is_output) {
            gradio.dispatch("input");
          }
        }
      }
    }
    {
      if (!active_source && sources) {
        active_source = sources[0];
      }
    }
    waveform_settings = {
      height: 50,
      barWidth: 2,
      barGap: 3,
      cursorWidth: 2,
      cursorColor: "#ddd5e9",
      autoplay,
      barRadius: 10,
      dragToSeek: true,
      normalize: true,
      minPxPerSec: 20
    };
    $$rendered = `  ${!interactive ? `${validate_component(Block, "Block").$$render(
      $$result,
      {
        variant: "solid",
        border_mode: dragging ? "focus" : "base",
        padding: false,
        allow_overflow: false,
        elem_id,
        elem_classes,
        visible,
        container,
        scale,
        min_width
      },
      {},
      {
        default: () => {
          return `${validate_component(Static, "StatusTracker").$$render($$result, Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status), {}, {})} ${validate_component(StaticAudio, "StaticAudio").$$render(
            $$result,
            {
              i18n: gradio.i18n,
              show_label,
              show_download_button,
              show_share_button,
              value,
              label,
              loop,
              waveform_settings,
              waveform_options,
              editable
            },
            {},
            {}
          )}`;
        }
      }
    )}` : `${validate_component(Block, "Block").$$render(
      $$result,
      {
        variant: value === null && active_source === "upload" ? "dashed" : "solid",
        border_mode: dragging ? "focus" : "base",
        padding: false,
        allow_overflow: false,
        elem_id,
        elem_classes,
        visible,
        container,
        scale,
        min_width
      },
      {},
      {
        default: () => {
          return `${validate_component(Static, "StatusTracker").$$render($$result, Object.assign({}, { autoscroll: gradio.autoscroll }, { i18n: gradio.i18n }, loading_status), {}, {})} ${validate_component(InteractiveAudio$1, "InteractiveAudio").$$render(
            $$result,
            {
              label,
              show_label,
              show_download_button,
              value,
              root,
              sources,
              active_source,
              pending,
              streaming,
              loop,
              max_file_size: gradio.max_file_size,
              handle_reset_value,
              editable,
              i18n: gradio.i18n,
              waveform_settings,
              waveform_options,
              trim_region_settings,
              stream_every,
              upload: (...args) => gradio.client.upload(...args),
              stream_handler: (...args) => gradio.client.stream(...args),
              recording,
              dragging,
              uploading,
              modify_stream: _modify_stream,
              set_time_limit
            },
            {
              recording: ($$value) => {
                recording = $$value;
                $$settled = false;
              },
              dragging: ($$value) => {
                dragging = $$value;
                $$settled = false;
              },
              uploading: ($$value) => {
                uploading = $$value;
                $$settled = false;
              },
              modify_stream: ($$value) => {
                _modify_stream = $$value;
                $$settled = false;
              },
              set_time_limit: ($$value) => {
                set_time_limit = $$value;
                $$settled = false;
              }
            },
            {
              default: () => {
                return `${validate_component(UploadText, "UploadText").$$render($$result, { i18n: gradio.i18n, type: "audio" }, {}, {})}`;
              }
            }
          )}`;
        }
      }
    )}`}`;
  } while (!$$settled);
  return $$rendered;
});
const Index$1 = Index;

export { InteractiveAudio$1 as BaseInteractiveAudio, StaticAudio as BaseStaticAudio, Index$1 as default };
//# sourceMappingURL=index15-C3qC1JfU.js.map
