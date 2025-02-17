import { SvelteComponent, init, safe_not_equal, svg_element, claim_svg_element, children, detach, attr, insert_hydration, append_hydration, noop, tick, element, create_component, space, claim_element, claim_component, claim_space, toggle_class, mount_component, set_input_value, action_destroyer, listen, prevent_default, transition_in, group_outros, transition_out, check_outros, is_function, destroy_component, run_all, createEventDispatcher, beforeUpdate, onMount, afterUpdate, text, claim_text, set_data, ensure_array_like, set_style, destroy_each, binding_callbacks, bind, add_flush_callback, bubble, flush, assign, get_spread_update, get_spread_object } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { i as BlockTitle, C as Clear, H as Send, J as Square, B as Block, S as Static } from "./2.8WKXZUMv.js";
import { F as File } from "./File.DqOJDDoa.js";
import { M as Microphone } from "./SelectSource.BQ7nwgVQ.js";
import { M as Music } from "./Music.BKn1BNLT.js";
import { V as Video } from "./Video.CzEOFOtQ.js";
import { U as Upload } from "./Upload.C9nHsuEq.js";
/* empty css                                                    */
import { I as Image } from "./Image.eJ_qOnkr.js";
/* empty css                                                    */
import { I as InteractiveAudio } from "./InteractiveAudio.BIw2E6Pd.js";
import { default as default2 } from "./Example.Bc0uctmv.js";
function create_fragment$2(ctx) {
  let svg;
  let g0;
  let g1;
  let g2;
  let path;
  return {
    c() {
      svg = svg_element("svg");
      g0 = svg_element("g");
      g1 = svg_element("g");
      g2 = svg_element("g");
      path = svg_element("path");
      this.h();
    },
    l(nodes) {
      svg = claim_svg_element(nodes, "svg", {
        fill: true,
        width: true,
        height: true,
        viewBox: true,
        xmlns: true
      });
      var svg_nodes = children(svg);
      g0 = claim_svg_element(svg_nodes, "g", { id: true, "stroke-width": true });
      children(g0).forEach(detach);
      g1 = claim_svg_element(svg_nodes, "g", {
        id: true,
        "stroke-linecap": true,
        "stroke-linejoin": true
      });
      children(g1).forEach(detach);
      g2 = claim_svg_element(svg_nodes, "g", { id: true });
      var g2_nodes = children(g2);
      path = claim_svg_element(g2_nodes, "path", { d: true, "fill-rule": true });
      children(path).forEach(detach);
      g2_nodes.forEach(detach);
      svg_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(g0, "id", "SVGRepo_bgCarrier");
      attr(g0, "stroke-width", "0");
      attr(g1, "id", "SVGRepo_tracerCarrier");
      attr(g1, "stroke-linecap", "round");
      attr(g1, "stroke-linejoin", "round");
      attr(path, "d", "M1752.768 221.109C1532.646.986 1174.283.986 954.161 221.109l-838.588 838.588c-154.052 154.165-154.052 404.894 0 558.946 149.534 149.421 409.976 149.308 559.059 0l758.738-758.626c87.982-88.094 87.982-231.417 0-319.51-88.32-88.208-231.642-87.982-319.51 0l-638.796 638.908 79.85 79.849 638.795-638.908c43.934-43.821 115.539-43.934 159.812 0 43.934 44.047 43.934 115.877 0 159.812l-758.739 758.625c-110.23 110.118-289.355 110.005-399.36 0-110.118-110.117-110.005-289.242 0-399.247l838.588-838.588c175.963-175.962 462.382-176.188 638.909 0 176.075 176.188 176.075 462.833 0 638.908l-798.607 798.72 79.849 79.85 798.607-798.72c220.01-220.123 220.01-578.485 0-798.607");
      attr(path, "fill-rule", "evenodd");
      attr(g2, "id", "SVGRepo_iconCarrier");
      attr(svg, "fill", "currentColor");
      attr(svg, "width", "100%");
      attr(svg, "height", "100%");
      attr(svg, "viewBox", "0 0 1920 1920");
      attr(svg, "xmlns", "http://www.w3.org/2000/svg");
    },
    m(target, anchor) {
      insert_hydration(target, svg, anchor);
      append_hydration(svg, g0);
      append_hydration(svg, g1);
      append_hydration(svg, g2);
      append_hydration(g2, path);
    },
    p: noop,
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(svg);
      }
    }
  };
}
class Paperclip extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, null, create_fragment$2, safe_not_equal, {});
  }
}
async function resize(target, lines, max_lines) {
  await tick();
  if (lines === max_lines)
    return;
  const computed_styles = window.getComputedStyle(target);
  const padding_top = parseFloat(computed_styles.paddingTop);
  const padding_bottom = parseFloat(computed_styles.paddingBottom);
  const line_height = parseFloat(computed_styles.lineHeight);
  let max = max_lines === void 0 ? false : padding_top + padding_bottom + line_height * max_lines;
  let min = padding_top + padding_bottom + lines * line_height;
  target.style.height = "1px";
  let scroll_height;
  if (max && target.scrollHeight > max) {
    scroll_height = max;
  } else if (target.scrollHeight < min) {
    scroll_height = min;
  } else {
    scroll_height = target.scrollHeight;
  }
  target.style.height = `${scroll_height}px`;
}
function text_area_resize(_el, _value) {
  if (_value.lines === _value.max_lines)
    return;
  _el.style.overflowY = "scroll";
  function handle_input(event) {
    resize(event.target, _value.lines, _value.max_lines);
  }
  _el.addEventListener("input", handle_input);
  if (!_value.text.trim())
    return;
  resize(_el, _value.lines, _value.max_lines);
  return {
    destroy: () => _el.removeEventListener("input", handle_input)
  };
}
function get_each_context(ctx, list, i) {
  const child_ctx = ctx.slice();
  child_ctx[71] = list[i];
  child_ctx[73] = i;
  return child_ctx;
}
function create_default_slot$1(ctx) {
  let t;
  return {
    c() {
      t = text(
        /*label*/
        ctx[7]
      );
    },
    l(nodes) {
      t = claim_text(
        nodes,
        /*label*/
        ctx[7]
      );
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*label*/
      128)
        set_data(
          t,
          /*label*/
          ctx2[7]
        );
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_if_block_7(ctx) {
  let div;
  let t;
  let current;
  let each_value = ensure_array_like(
    /*value*/
    ctx[0].files
  );
  let each_blocks = [];
  for (let i = 0; i < each_value.length; i += 1) {
    each_blocks[i] = create_each_block(get_each_context(ctx, each_value, i));
  }
  const out = (i) => transition_out(each_blocks[i], 1, 1, () => {
    each_blocks[i] = null;
  });
  let if_block = (
    /*uploading*/
    ctx[28] && create_if_block_8()
  );
  return {
    c() {
      div = element("div");
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].c();
      }
      t = space();
      if (if_block)
        if_block.c();
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", {
        class: true,
        "aria-label": true,
        "data-testid": true,
        style: true
      });
      var div_nodes = children(div);
      for (let i = 0; i < each_blocks.length; i += 1) {
        each_blocks[i].l(div_nodes);
      }
      t = claim_space(div_nodes);
      if (if_block)
        if_block.l(div_nodes);
      div_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(div, "class", "thumbnails scroll-hide svelte-d47mdf");
      attr(div, "aria-label", "Uploaded files");
      attr(div, "data-testid", "container_el");
      set_style(
        div,
        "display",
        /*value*/
        ctx[0].files.length > 0 || /*uploading*/
        ctx[28] ? "flex" : "none"
      );
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
      for (let i = 0; i < each_blocks.length; i += 1) {
        if (each_blocks[i]) {
          each_blocks[i].m(div, null);
        }
      }
      append_hydration(div, t);
      if (if_block)
        if_block.m(div, null);
      current = true;
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*value, disabled*/
      65 | dirty[1] & /*remove_thumbnail*/
      32) {
        each_value = ensure_array_like(
          /*value*/
          ctx2[0].files
        );
        let i;
        for (i = 0; i < each_value.length; i += 1) {
          const child_ctx = get_each_context(ctx2, each_value, i);
          if (each_blocks[i]) {
            each_blocks[i].p(child_ctx, dirty);
            transition_in(each_blocks[i], 1);
          } else {
            each_blocks[i] = create_each_block(child_ctx);
            each_blocks[i].c();
            transition_in(each_blocks[i], 1);
            each_blocks[i].m(div, t);
          }
        }
        group_outros();
        for (i = each_value.length; i < each_blocks.length; i += 1) {
          out(i);
        }
        check_outros();
      }
      if (
        /*uploading*/
        ctx2[28]
      ) {
        if (if_block)
          ;
        else {
          if_block = create_if_block_8();
          if_block.c();
          if_block.m(div, null);
        }
      } else if (if_block) {
        if_block.d(1);
        if_block = null;
      }
      if (!current || dirty[0] & /*value, uploading*/
      268435457) {
        set_style(
          div,
          "display",
          /*value*/
          ctx2[0].files.length > 0 || /*uploading*/
          ctx2[28] ? "flex" : "none"
        );
      }
    },
    i(local) {
      if (current)
        return;
      for (let i = 0; i < each_value.length; i += 1) {
        transition_in(each_blocks[i]);
      }
      current = true;
    },
    o(local) {
      each_blocks = each_blocks.filter(Boolean);
      for (let i = 0; i < each_blocks.length; i += 1) {
        transition_out(each_blocks[i]);
      }
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
      destroy_each(each_blocks, detaching);
      if (if_block)
        if_block.d();
    }
  };
}
function create_else_block_2(ctx) {
  let file_1;
  let current;
  file_1 = new File({});
  return {
    c() {
      create_component(file_1.$$.fragment);
    },
    l(nodes) {
      claim_component(file_1.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(file_1, target, anchor);
      current = true;
    },
    p: noop,
    i(local) {
      if (current)
        return;
      transition_in(file_1.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(file_1.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(file_1, detaching);
    }
  };
}
function create_if_block_11(ctx) {
  let video;
  let current;
  video = new Video({});
  return {
    c() {
      create_component(video.$$.fragment);
    },
    l(nodes) {
      claim_component(video.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(video, target, anchor);
      current = true;
    },
    p: noop,
    i(local) {
      if (current)
        return;
      transition_in(video.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(video.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(video, detaching);
    }
  };
}
function create_if_block_10(ctx) {
  let music;
  let current;
  music = new Music({});
  return {
    c() {
      create_component(music.$$.fragment);
    },
    l(nodes) {
      claim_component(music.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(music, target, anchor);
      current = true;
    },
    p: noop,
    i(local) {
      if (current)
        return;
      transition_in(music.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(music.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(music, detaching);
    }
  };
}
function create_if_block_9(ctx) {
  let image;
  let current;
  image = new Image({
    props: {
      src: (
        /*file*/
        ctx[71].url
      ),
      title: null,
      alt: "",
      loading: "lazy",
      class: "thumbnail-image"
    }
  });
  return {
    c() {
      create_component(image.$$.fragment);
    },
    l(nodes) {
      claim_component(image.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(image, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const image_changes = {};
      if (dirty[0] & /*value*/
      1)
        image_changes.src = /*file*/
        ctx2[71].url;
      image.$set(image_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(image.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(image.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(image, detaching);
    }
  };
}
function create_each_block(ctx) {
  let span;
  let button1;
  let button0;
  let clear;
  let t;
  let show_if;
  let show_if_1;
  let show_if_2;
  let current_block_type_index;
  let if_block;
  let current;
  let mounted;
  let dispose;
  clear = new Clear({});
  function click_handler(...args) {
    return (
      /*click_handler*/
      ctx[51](
        /*index*/
        ctx[73],
        ...args
      )
    );
  }
  const if_block_creators = [create_if_block_9, create_if_block_10, create_if_block_11, create_else_block_2];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (dirty[0] & /*value*/
    1)
      show_if = null;
    if (dirty[0] & /*value*/
    1)
      show_if_1 = null;
    if (dirty[0] & /*value*/
    1)
      show_if_2 = null;
    if (show_if == null)
      show_if = !!/*file*/
      (ctx2[71].mime_type && /*file*/
      ctx2[71].mime_type.includes("image"));
    if (show_if)
      return 0;
    if (show_if_1 == null)
      show_if_1 = !!/*file*/
      (ctx2[71].mime_type && /*file*/
      ctx2[71].mime_type.includes("audio"));
    if (show_if_1)
      return 1;
    if (show_if_2 == null)
      show_if_2 = !!/*file*/
      (ctx2[71].mime_type && /*file*/
      ctx2[71].mime_type.includes("video"));
    if (show_if_2)
      return 2;
    return 3;
  }
  current_block_type_index = select_block_type(ctx, [-1, -1, -1]);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      span = element("span");
      button1 = element("button");
      button0 = element("button");
      create_component(clear.$$.fragment);
      t = space();
      if_block.c();
      this.h();
    },
    l(nodes) {
      span = claim_element(nodes, "SPAN", { role: true, "aria-label": true });
      var span_nodes = children(span);
      button1 = claim_element(span_nodes, "BUTTON", { class: true });
      var button1_nodes = children(button1);
      button0 = claim_element(button1_nodes, "BUTTON", { class: true });
      var button0_nodes = children(button0);
      claim_component(clear.$$.fragment, button0_nodes);
      button0_nodes.forEach(detach);
      t = claim_space(button1_nodes);
      if_block.l(button1_nodes);
      button1_nodes.forEach(detach);
      span_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button0, "class", "delete-button svelte-d47mdf");
      toggle_class(
        button0,
        "disabled",
        /*disabled*/
        ctx[6]
      );
      attr(button1, "class", "thumbnail-item thumbnail-small svelte-d47mdf");
      attr(span, "role", "listitem");
      attr(span, "aria-label", "File thumbnail");
    },
    m(target, anchor) {
      insert_hydration(target, span, anchor);
      append_hydration(span, button1);
      append_hydration(button1, button0);
      mount_component(clear, button0, null);
      append_hydration(button1, t);
      if_blocks[current_block_type_index].m(button1, null);
      current = true;
      if (!mounted) {
        dispose = listen(button0, "click", click_handler);
        mounted = true;
      }
    },
    p(new_ctx, dirty) {
      ctx = new_ctx;
      if (!current || dirty[0] & /*disabled*/
      64) {
        toggle_class(
          button0,
          "disabled",
          /*disabled*/
          ctx[6]
        );
      }
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type(ctx, dirty);
      if (current_block_type_index === previous_block_index) {
        if_blocks[current_block_type_index].p(ctx, dirty);
      } else {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block = if_blocks[current_block_type_index];
        if (!if_block) {
          if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
          if_block.c();
        } else {
          if_block.p(ctx, dirty);
        }
        transition_in(if_block, 1);
        if_block.m(button1, null);
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(clear.$$.fragment, local);
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(clear.$$.fragment, local);
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(span);
      }
      destroy_component(clear);
      if_blocks[current_block_type_index].d();
      mounted = false;
      dispose();
    }
  };
}
function create_if_block_8(ctx) {
  let div;
  return {
    c() {
      div = element("div");
      this.h();
    },
    l(nodes) {
      div = claim_element(nodes, "DIV", {
        class: true,
        role: true,
        "aria-label": true
      });
      children(div).forEach(detach);
      this.h();
    },
    h() {
      attr(div, "class", "loader svelte-d47mdf");
      attr(div, "role", "status");
      attr(div, "aria-label", "Uploading");
    },
    m(target, anchor) {
      insert_hydration(target, div, anchor);
    },
    d(detaching) {
      if (detaching) {
        detach(div);
      }
    }
  };
}
function create_if_block_6(ctx) {
  let interactiveaudio;
  let current;
  interactiveaudio = new InteractiveAudio({
    props: {
      sources: ["microphone"],
      class_name: "compact-audio",
      recording,
      waveform_settings: (
        /*waveform_settings*/
        ctx[22]
      ),
      waveform_options: (
        /*waveform_options*/
        ctx[23]
      ),
      i18n: (
        /*i18n*/
        ctx[4]
      ),
      active_source: (
        /*active_source*/
        ctx[2]
      ),
      upload: (
        /*upload*/
        ctx[19]
      ),
      stream_handler: (
        /*stream_handler*/
        ctx[20]
      ),
      stream_every: 1,
      editable: true,
      label: (
        /*label*/
        ctx[7]
      ),
      root: (
        /*root*/
        ctx[16]
      ),
      loop: false,
      show_label: false,
      show_download_button: false,
      dragging: false
    }
  });
  interactiveaudio.$on(
    "change",
    /*change_handler*/
    ctx[52]
  );
  interactiveaudio.$on(
    "clear",
    /*clear_handler*/
    ctx[53]
  );
  interactiveaudio.$on(
    "start_recording",
    /*start_recording_handler*/
    ctx[54]
  );
  interactiveaudio.$on(
    "pause_recording",
    /*pause_recording_handler*/
    ctx[55]
  );
  interactiveaudio.$on(
    "stop_recording",
    /*stop_recording_handler*/
    ctx[56]
  );
  return {
    c() {
      create_component(interactiveaudio.$$.fragment);
    },
    l(nodes) {
      claim_component(interactiveaudio.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(interactiveaudio, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const interactiveaudio_changes = {};
      if (dirty[0] & /*waveform_settings*/
      4194304)
        interactiveaudio_changes.waveform_settings = /*waveform_settings*/
        ctx2[22];
      if (dirty[0] & /*waveform_options*/
      8388608)
        interactiveaudio_changes.waveform_options = /*waveform_options*/
        ctx2[23];
      if (dirty[0] & /*i18n*/
      16)
        interactiveaudio_changes.i18n = /*i18n*/
        ctx2[4];
      if (dirty[0] & /*active_source*/
      4)
        interactiveaudio_changes.active_source = /*active_source*/
        ctx2[2];
      if (dirty[0] & /*upload*/
      524288)
        interactiveaudio_changes.upload = /*upload*/
        ctx2[19];
      if (dirty[0] & /*stream_handler*/
      1048576)
        interactiveaudio_changes.stream_handler = /*stream_handler*/
        ctx2[20];
      if (dirty[0] & /*label*/
      128)
        interactiveaudio_changes.label = /*label*/
        ctx2[7];
      if (dirty[0] & /*root*/
      65536)
        interactiveaudio_changes.root = /*root*/
        ctx2[16];
      interactiveaudio.$set(interactiveaudio_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(interactiveaudio.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(interactiveaudio.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(interactiveaudio, detaching);
    }
  };
}
function create_if_block_5(ctx) {
  let upload_1;
  let updating_dragging;
  let updating_uploading;
  let updating_hidden_upload;
  let t;
  let button;
  let paperclip;
  let current;
  let mounted;
  let dispose;
  function upload_1_dragging_binding(value) {
    ctx[58](value);
  }
  function upload_1_uploading_binding(value) {
    ctx[59](value);
  }
  function upload_1_hidden_upload_binding(value) {
    ctx[60](value);
  }
  let upload_1_props = {
    file_count: (
      /*file_count*/
      ctx[21]
    ),
    filetype: (
      /*file_types*/
      ctx[17]
    ),
    root: (
      /*root*/
      ctx[16]
    ),
    max_file_size: (
      /*max_file_size*/
      ctx[18]
    ),
    show_progress: false,
    disable_click: true,
    hidden: true,
    upload: (
      /*upload*/
      ctx[19]
    ),
    stream_handler: (
      /*stream_handler*/
      ctx[20]
    )
  };
  if (
    /*dragging*/
    ctx[1] !== void 0
  ) {
    upload_1_props.dragging = /*dragging*/
    ctx[1];
  }
  if (
    /*uploading*/
    ctx[28] !== void 0
  ) {
    upload_1_props.uploading = /*uploading*/
    ctx[28];
  }
  if (
    /*hidden_upload*/
    ctx[27] !== void 0
  ) {
    upload_1_props.hidden_upload = /*hidden_upload*/
    ctx[27];
  }
  upload_1 = new Upload({ props: upload_1_props });
  ctx[57](upload_1);
  binding_callbacks.push(() => bind(upload_1, "dragging", upload_1_dragging_binding));
  binding_callbacks.push(() => bind(upload_1, "uploading", upload_1_uploading_binding));
  binding_callbacks.push(() => bind(upload_1, "hidden_upload", upload_1_hidden_upload_binding));
  upload_1.$on(
    "load",
    /*handle_upload*/
    ctx[35]
  );
  upload_1.$on(
    "error",
    /*error_handler*/
    ctx[61]
  );
  paperclip = new Paperclip({});
  return {
    c() {
      create_component(upload_1.$$.fragment);
      t = space();
      button = element("button");
      create_component(paperclip.$$.fragment);
      this.h();
    },
    l(nodes) {
      claim_component(upload_1.$$.fragment, nodes);
      t = claim_space(nodes);
      button = claim_element(nodes, "BUTTON", { "data-testid": true, class: true });
      var button_nodes = children(button);
      claim_component(paperclip.$$.fragment, button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "data-testid", "upload-button");
      attr(button, "class", "upload-button svelte-d47mdf");
    },
    m(target, anchor) {
      mount_component(upload_1, target, anchor);
      insert_hydration(target, t, anchor);
      insert_hydration(target, button, anchor);
      mount_component(paperclip, button, null);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*handle_upload_click*/
          ctx[37]
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      const upload_1_changes = {};
      if (dirty[0] & /*file_count*/
      2097152)
        upload_1_changes.file_count = /*file_count*/
        ctx2[21];
      if (dirty[0] & /*file_types*/
      131072)
        upload_1_changes.filetype = /*file_types*/
        ctx2[17];
      if (dirty[0] & /*root*/
      65536)
        upload_1_changes.root = /*root*/
        ctx2[16];
      if (dirty[0] & /*max_file_size*/
      262144)
        upload_1_changes.max_file_size = /*max_file_size*/
        ctx2[18];
      if (dirty[0] & /*upload*/
      524288)
        upload_1_changes.upload = /*upload*/
        ctx2[19];
      if (dirty[0] & /*stream_handler*/
      1048576)
        upload_1_changes.stream_handler = /*stream_handler*/
        ctx2[20];
      if (!updating_dragging && dirty[0] & /*dragging*/
      2) {
        updating_dragging = true;
        upload_1_changes.dragging = /*dragging*/
        ctx2[1];
        add_flush_callback(() => updating_dragging = false);
      }
      if (!updating_uploading && dirty[0] & /*uploading*/
      268435456) {
        updating_uploading = true;
        upload_1_changes.uploading = /*uploading*/
        ctx2[28];
        add_flush_callback(() => updating_uploading = false);
      }
      if (!updating_hidden_upload && dirty[0] & /*hidden_upload*/
      134217728) {
        updating_hidden_upload = true;
        upload_1_changes.hidden_upload = /*hidden_upload*/
        ctx2[27];
        add_flush_callback(() => updating_hidden_upload = false);
      }
      upload_1.$set(upload_1_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(upload_1.$$.fragment, local);
      transition_in(paperclip.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(upload_1.$$.fragment, local);
      transition_out(paperclip.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
        detach(button);
      }
      ctx[57](null);
      destroy_component(upload_1, detaching);
      destroy_component(paperclip);
      mounted = false;
      dispose();
    }
  };
}
function create_if_block_4(ctx) {
  let button;
  let microphone;
  let current;
  let mounted;
  let dispose;
  microphone = new Microphone({});
  return {
    c() {
      button = element("button");
      create_component(microphone.$$.fragment);
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { "data-testid": true, class: true });
      var button_nodes = children(button);
      claim_component(microphone.$$.fragment, button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "data-testid", "microphone-button");
      attr(button, "class", "microphone-button svelte-d47mdf");
      toggle_class(button, "recording", recording);
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      mount_component(microphone, button, null);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*click_handler_1*/
          ctx[62]
        );
        mounted = true;
      }
    },
    p: noop,
    i(local) {
      if (current)
        return;
      transition_in(microphone.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(microphone.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      destroy_component(microphone);
      mounted = false;
      dispose();
    }
  };
}
function create_if_block_2(ctx) {
  let button;
  let current_block_type_index;
  let if_block;
  let current;
  let mounted;
  let dispose;
  const if_block_creators = [create_if_block_3, create_else_block_1];
  const if_blocks = [];
  function select_block_type_1(ctx2, dirty) {
    if (
      /*submit_btn*/
      ctx2[11] === true
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type_1(ctx);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      button = element("button");
      if_block.c();
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { class: true });
      var button_nodes = children(button);
      if_block.l(button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "submit-button svelte-d47mdf");
      toggle_class(
        button,
        "padded-button",
        /*submit_btn*/
        ctx[11] !== true
      );
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      if_blocks[current_block_type_index].m(button, null);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*handle_submit*/
          ctx[39]
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type_1(ctx2);
      if (current_block_type_index === previous_block_index) {
        if_blocks[current_block_type_index].p(ctx2, dirty);
      } else {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block = if_blocks[current_block_type_index];
        if (!if_block) {
          if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block.c();
        } else {
          if_block.p(ctx2, dirty);
        }
        transition_in(if_block, 1);
        if_block.m(button, null);
      }
      if (!current || dirty[0] & /*submit_btn*/
      2048) {
        toggle_class(
          button,
          "padded-button",
          /*submit_btn*/
          ctx2[11] !== true
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      if_blocks[current_block_type_index].d();
      mounted = false;
      dispose();
    }
  };
}
function create_else_block_1(ctx) {
  let t;
  return {
    c() {
      t = text(
        /*submit_btn*/
        ctx[11]
      );
    },
    l(nodes) {
      t = claim_text(
        nodes,
        /*submit_btn*/
        ctx[11]
      );
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*submit_btn*/
      2048)
        set_data(
          t,
          /*submit_btn*/
          ctx2[11]
        );
    },
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_if_block_3(ctx) {
  let send;
  let current;
  send = new Send({});
  return {
    c() {
      create_component(send.$$.fragment);
    },
    l(nodes) {
      claim_component(send.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(send, target, anchor);
      current = true;
    },
    p: noop,
    i(local) {
      if (current)
        return;
      transition_in(send.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(send.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(send, detaching);
    }
  };
}
function create_if_block$1(ctx) {
  let button;
  let current_block_type_index;
  let if_block;
  let current;
  let mounted;
  let dispose;
  const if_block_creators = [create_if_block_1, create_else_block];
  const if_blocks = [];
  function select_block_type_2(ctx2, dirty) {
    if (
      /*stop_btn*/
      ctx2[12] === true
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type_2(ctx);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      button = element("button");
      if_block.c();
      this.h();
    },
    l(nodes) {
      button = claim_element(nodes, "BUTTON", { class: true });
      var button_nodes = children(button);
      if_block.l(button_nodes);
      button_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(button, "class", "stop-button svelte-d47mdf");
      toggle_class(
        button,
        "padded-button",
        /*stop_btn*/
        ctx[12] !== true
      );
    },
    m(target, anchor) {
      insert_hydration(target, button, anchor);
      if_blocks[current_block_type_index].m(button, null);
      current = true;
      if (!mounted) {
        dispose = listen(
          button,
          "click",
          /*handle_stop*/
          ctx[38]
        );
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type_2(ctx2);
      if (current_block_type_index === previous_block_index) {
        if_blocks[current_block_type_index].p(ctx2, dirty);
      } else {
        group_outros();
        transition_out(if_blocks[previous_block_index], 1, 1, () => {
          if_blocks[previous_block_index] = null;
        });
        check_outros();
        if_block = if_blocks[current_block_type_index];
        if (!if_block) {
          if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx2);
          if_block.c();
        } else {
          if_block.p(ctx2, dirty);
        }
        transition_in(if_block, 1);
        if_block.m(button, null);
      }
      if (!current || dirty[0] & /*stop_btn*/
      4096) {
        toggle_class(
          button,
          "padded-button",
          /*stop_btn*/
          ctx2[12] !== true
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(button);
      }
      if_blocks[current_block_type_index].d();
      mounted = false;
      dispose();
    }
  };
}
function create_else_block(ctx) {
  let t;
  return {
    c() {
      t = text(
        /*stop_btn*/
        ctx[12]
      );
    },
    l(nodes) {
      t = claim_text(
        nodes,
        /*stop_btn*/
        ctx[12]
      );
    },
    m(target, anchor) {
      insert_hydration(target, t, anchor);
    },
    p(ctx2, dirty) {
      if (dirty[0] & /*stop_btn*/
      4096)
        set_data(
          t,
          /*stop_btn*/
          ctx2[12]
        );
    },
    i: noop,
    o: noop,
    d(detaching) {
      if (detaching) {
        detach(t);
      }
    }
  };
}
function create_if_block_1(ctx) {
  let square;
  let current;
  square = new Square({
    props: { fill: "none", stroke_width: 2.5 }
  });
  return {
    c() {
      create_component(square.$$.fragment);
    },
    l(nodes) {
      claim_component(square.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(square, target, anchor);
      current = true;
    },
    p: noop,
    i(local) {
      if (current)
        return;
      transition_in(square.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(square.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(square, detaching);
    }
  };
}
function create_fragment$1(ctx) {
  let div1;
  let blocktitle;
  let t0;
  let t1;
  let show_if_2 = (
    /*sources*/
    ctx[24] && /*sources*/
    ctx[24].includes("microphone") && /*active_source*/
    ctx[2] === "microphone"
  );
  let t2;
  let div0;
  let show_if_1 = (
    /*sources*/
    ctx[24] && /*sources*/
    ctx[24].includes("upload") && !/*disabled*/
    ctx[6] && !/*file_count*/
    (ctx[21] === "single" && /*value*/
    ctx[0].files.length > 0)
  );
  let t3;
  let show_if = (
    /*sources*/
    ctx[24] && /*sources*/
    ctx[24].includes("microphone")
  );
  let t4;
  let textarea;
  let textarea_dir_value;
  let textarea_style_value;
  let text_area_resize_action;
  let t5;
  let t6;
  let current;
  let mounted;
  let dispose;
  blocktitle = new BlockTitle({
    props: {
      root: (
        /*root*/
        ctx[16]
      ),
      show_label: (
        /*show_label*/
        ctx[9]
      ),
      info: (
        /*info*/
        ctx[8]
      ),
      $$slots: { default: [create_default_slot$1] },
      $$scope: { ctx }
    }
  });
  let if_block0 = (
    /*value*/
    (ctx[0].files.length > 0 || /*uploading*/
    ctx[28]) && create_if_block_7(ctx)
  );
  let if_block1 = show_if_2 && create_if_block_6(ctx);
  let if_block2 = show_if_1 && create_if_block_5(ctx);
  let if_block3 = show_if && create_if_block_4(ctx);
  let if_block4 = (
    /*submit_btn*/
    ctx[11] && create_if_block_2(ctx)
  );
  let if_block5 = (
    /*stop_btn*/
    ctx[12] && create_if_block$1(ctx)
  );
  return {
    c() {
      div1 = element("div");
      create_component(blocktitle.$$.fragment);
      t0 = space();
      if (if_block0)
        if_block0.c();
      t1 = space();
      if (if_block1)
        if_block1.c();
      t2 = space();
      div0 = element("div");
      if (if_block2)
        if_block2.c();
      t3 = space();
      if (if_block3)
        if_block3.c();
      t4 = space();
      textarea = element("textarea");
      t5 = space();
      if (if_block4)
        if_block4.c();
      t6 = space();
      if (if_block5)
        if_block5.c();
      this.h();
    },
    l(nodes) {
      div1 = claim_element(nodes, "DIV", {
        class: true,
        role: true,
        "aria-label": true
      });
      var div1_nodes = children(div1);
      claim_component(blocktitle.$$.fragment, div1_nodes);
      t0 = claim_space(div1_nodes);
      if (if_block0)
        if_block0.l(div1_nodes);
      t1 = claim_space(div1_nodes);
      if (if_block1)
        if_block1.l(div1_nodes);
      t2 = claim_space(div1_nodes);
      div0 = claim_element(div1_nodes, "DIV", { class: true });
      var div0_nodes = children(div0);
      if (if_block2)
        if_block2.l(div0_nodes);
      t3 = claim_space(div0_nodes);
      if (if_block3)
        if_block3.l(div0_nodes);
      t4 = claim_space(div0_nodes);
      textarea = claim_element(div0_nodes, "TEXTAREA", {
        "data-testid": true,
        class: true,
        dir: true,
        placeholder: true,
        rows: true,
        style: true
      });
      children(textarea).forEach(detach);
      t5 = claim_space(div0_nodes);
      if (if_block4)
        if_block4.l(div0_nodes);
      t6 = claim_space(div0_nodes);
      if (if_block5)
        if_block5.l(div0_nodes);
      div0_nodes.forEach(detach);
      div1_nodes.forEach(detach);
      this.h();
    },
    h() {
      attr(textarea, "data-testid", "textbox");
      attr(textarea, "class", "scroll-hide svelte-d47mdf");
      attr(textarea, "dir", textarea_dir_value = /*rtl*/
      ctx[13] ? "rtl" : "ltr");
      attr(
        textarea,
        "placeholder",
        /*placeholder*/
        ctx[5]
      );
      attr(
        textarea,
        "rows",
        /*lines*/
        ctx[3]
      );
      textarea.disabled = /*disabled*/
      ctx[6];
      textarea.autofocus = /*autofocus*/
      ctx[14];
      attr(textarea, "style", textarea_style_value = /*text_align*/
      ctx[15] ? "text-align: " + /*text_align*/
      ctx[15] : "");
      toggle_class(textarea, "no-label", !/*show_label*/
      ctx[9]);
      attr(div0, "class", "input-container svelte-d47mdf");
      attr(div1, "class", "full-container svelte-d47mdf");
      attr(div1, "role", "group");
      attr(div1, "aria-label", "Multimedia input field");
      toggle_class(
        div1,
        "dragging",
        /*dragging*/
        ctx[1]
      );
    },
    m(target, anchor) {
      insert_hydration(target, div1, anchor);
      mount_component(blocktitle, div1, null);
      append_hydration(div1, t0);
      if (if_block0)
        if_block0.m(div1, null);
      append_hydration(div1, t1);
      if (if_block1)
        if_block1.m(div1, null);
      append_hydration(div1, t2);
      append_hydration(div1, div0);
      if (if_block2)
        if_block2.m(div0, null);
      append_hydration(div0, t3);
      if (if_block3)
        if_block3.m(div0, null);
      append_hydration(div0, t4);
      append_hydration(div0, textarea);
      set_input_value(
        textarea,
        /*value*/
        ctx[0].text
      );
      ctx[64](textarea);
      append_hydration(div0, t5);
      if (if_block4)
        if_block4.m(div0, null);
      append_hydration(div0, t6);
      if (if_block5)
        if_block5.m(div0, null);
      ctx[65](div1);
      current = true;
      if (
        /*autofocus*/
        ctx[14]
      )
        textarea.focus();
      if (!mounted) {
        dispose = [
          action_destroyer(text_area_resize_action = text_area_resize.call(null, textarea, {
            text: (
              /*value*/
              ctx[0].text
            ),
            lines: (
              /*lines*/
              ctx[3]
            ),
            max_lines: (
              /*max_lines*/
              ctx[10]
            )
          })),
          listen(
            textarea,
            "input",
            /*textarea_input_handler*/
            ctx[63]
          ),
          listen(
            textarea,
            "keypress",
            /*handle_keypress*/
            ctx[33]
          ),
          listen(
            textarea,
            "blur",
            /*blur_handler*/
            ctx[49]
          ),
          listen(
            textarea,
            "select",
            /*handle_select*/
            ctx[32]
          ),
          listen(
            textarea,
            "focus",
            /*focus_handler*/
            ctx[50]
          ),
          listen(
            textarea,
            "scroll",
            /*handle_scroll*/
            ctx[34]
          ),
          listen(
            textarea,
            "paste",
            /*handle_paste*/
            ctx[40]
          ),
          listen(
            div1,
            "dragenter",
            /*handle_dragenter*/
            ctx[41]
          ),
          listen(
            div1,
            "dragleave",
            /*handle_dragleave*/
            ctx[42]
          ),
          listen(div1, "dragover", prevent_default(
            /*dragover_handler*/
            ctx[48]
          )),
          listen(
            div1,
            "drop",
            /*handle_drop*/
            ctx[43]
          )
        ];
        mounted = true;
      }
    },
    p(ctx2, dirty) {
      const blocktitle_changes = {};
      if (dirty[0] & /*root*/
      65536)
        blocktitle_changes.root = /*root*/
        ctx2[16];
      if (dirty[0] & /*show_label*/
      512)
        blocktitle_changes.show_label = /*show_label*/
        ctx2[9];
      if (dirty[0] & /*info*/
      256)
        blocktitle_changes.info = /*info*/
        ctx2[8];
      if (dirty[0] & /*label*/
      128 | dirty[2] & /*$$scope*/
      4096) {
        blocktitle_changes.$$scope = { dirty, ctx: ctx2 };
      }
      blocktitle.$set(blocktitle_changes);
      if (
        /*value*/
        ctx2[0].files.length > 0 || /*uploading*/
        ctx2[28]
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
          if (dirty[0] & /*value, uploading*/
          268435457) {
            transition_in(if_block0, 1);
          }
        } else {
          if_block0 = create_if_block_7(ctx2);
          if_block0.c();
          transition_in(if_block0, 1);
          if_block0.m(div1, t1);
        }
      } else if (if_block0) {
        group_outros();
        transition_out(if_block0, 1, 1, () => {
          if_block0 = null;
        });
        check_outros();
      }
      if (dirty[0] & /*sources, active_source*/
      16777220)
        show_if_2 = /*sources*/
        ctx2[24] && /*sources*/
        ctx2[24].includes("microphone") && /*active_source*/
        ctx2[2] === "microphone";
      if (show_if_2) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty[0] & /*sources, active_source*/
          16777220) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block_6(ctx2);
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(div1, t2);
        }
      } else if (if_block1) {
        group_outros();
        transition_out(if_block1, 1, 1, () => {
          if_block1 = null;
        });
        check_outros();
      }
      if (dirty[0] & /*sources, disabled, file_count, value*/
      18874433)
        show_if_1 = /*sources*/
        ctx2[24] && /*sources*/
        ctx2[24].includes("upload") && !/*disabled*/
        ctx2[6] && !/*file_count*/
        (ctx2[21] === "single" && /*value*/
        ctx2[0].files.length > 0);
      if (show_if_1) {
        if (if_block2) {
          if_block2.p(ctx2, dirty);
          if (dirty[0] & /*sources, disabled, file_count, value*/
          18874433) {
            transition_in(if_block2, 1);
          }
        } else {
          if_block2 = create_if_block_5(ctx2);
          if_block2.c();
          transition_in(if_block2, 1);
          if_block2.m(div0, t3);
        }
      } else if (if_block2) {
        group_outros();
        transition_out(if_block2, 1, 1, () => {
          if_block2 = null;
        });
        check_outros();
      }
      if (dirty[0] & /*sources*/
      16777216)
        show_if = /*sources*/
        ctx2[24] && /*sources*/
        ctx2[24].includes("microphone");
      if (show_if) {
        if (if_block3) {
          if_block3.p(ctx2, dirty);
          if (dirty[0] & /*sources*/
          16777216) {
            transition_in(if_block3, 1);
          }
        } else {
          if_block3 = create_if_block_4(ctx2);
          if_block3.c();
          transition_in(if_block3, 1);
          if_block3.m(div0, t4);
        }
      } else if (if_block3) {
        group_outros();
        transition_out(if_block3, 1, 1, () => {
          if_block3 = null;
        });
        check_outros();
      }
      if (!current || dirty[0] & /*rtl*/
      8192 && textarea_dir_value !== (textarea_dir_value = /*rtl*/
      ctx2[13] ? "rtl" : "ltr")) {
        attr(textarea, "dir", textarea_dir_value);
      }
      if (!current || dirty[0] & /*placeholder*/
      32) {
        attr(
          textarea,
          "placeholder",
          /*placeholder*/
          ctx2[5]
        );
      }
      if (!current || dirty[0] & /*lines*/
      8) {
        attr(
          textarea,
          "rows",
          /*lines*/
          ctx2[3]
        );
      }
      if (!current || dirty[0] & /*disabled*/
      64) {
        textarea.disabled = /*disabled*/
        ctx2[6];
      }
      if (!current || dirty[0] & /*autofocus*/
      16384) {
        textarea.autofocus = /*autofocus*/
        ctx2[14];
      }
      if (!current || dirty[0] & /*text_align*/
      32768 && textarea_style_value !== (textarea_style_value = /*text_align*/
      ctx2[15] ? "text-align: " + /*text_align*/
      ctx2[15] : "")) {
        attr(textarea, "style", textarea_style_value);
      }
      if (text_area_resize_action && is_function(text_area_resize_action.update) && dirty[0] & /*value, lines, max_lines*/
      1033)
        text_area_resize_action.update.call(null, {
          text: (
            /*value*/
            ctx2[0].text
          ),
          lines: (
            /*lines*/
            ctx2[3]
          ),
          max_lines: (
            /*max_lines*/
            ctx2[10]
          )
        });
      if (dirty[0] & /*value*/
      1) {
        set_input_value(
          textarea,
          /*value*/
          ctx2[0].text
        );
      }
      if (!current || dirty[0] & /*show_label*/
      512) {
        toggle_class(textarea, "no-label", !/*show_label*/
        ctx2[9]);
      }
      if (
        /*submit_btn*/
        ctx2[11]
      ) {
        if (if_block4) {
          if_block4.p(ctx2, dirty);
          if (dirty[0] & /*submit_btn*/
          2048) {
            transition_in(if_block4, 1);
          }
        } else {
          if_block4 = create_if_block_2(ctx2);
          if_block4.c();
          transition_in(if_block4, 1);
          if_block4.m(div0, t6);
        }
      } else if (if_block4) {
        group_outros();
        transition_out(if_block4, 1, 1, () => {
          if_block4 = null;
        });
        check_outros();
      }
      if (
        /*stop_btn*/
        ctx2[12]
      ) {
        if (if_block5) {
          if_block5.p(ctx2, dirty);
          if (dirty[0] & /*stop_btn*/
          4096) {
            transition_in(if_block5, 1);
          }
        } else {
          if_block5 = create_if_block$1(ctx2);
          if_block5.c();
          transition_in(if_block5, 1);
          if_block5.m(div0, null);
        }
      } else if (if_block5) {
        group_outros();
        transition_out(if_block5, 1, 1, () => {
          if_block5 = null;
        });
        check_outros();
      }
      if (!current || dirty[0] & /*dragging*/
      2) {
        toggle_class(
          div1,
          "dragging",
          /*dragging*/
          ctx2[1]
        );
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(blocktitle.$$.fragment, local);
      transition_in(if_block0);
      transition_in(if_block1);
      transition_in(if_block2);
      transition_in(if_block3);
      transition_in(if_block4);
      transition_in(if_block5);
      current = true;
    },
    o(local) {
      transition_out(blocktitle.$$.fragment, local);
      transition_out(if_block0);
      transition_out(if_block1);
      transition_out(if_block2);
      transition_out(if_block3);
      transition_out(if_block4);
      transition_out(if_block5);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(div1);
      }
      destroy_component(blocktitle);
      if (if_block0)
        if_block0.d();
      if (if_block1)
        if_block1.d();
      if (if_block2)
        if_block2.d();
      if (if_block3)
        if_block3.d();
      ctx[64](null);
      if (if_block4)
        if_block4.d();
      if (if_block5)
        if_block5.d();
      ctx[65](null);
      mounted = false;
      run_all(dispose);
    }
  };
}
let recording = false;
function instance$1($$self, $$props, $$invalidate) {
  let { value = { text: "", files: [] } } = $$props;
  let { value_is_output = false } = $$props;
  let { lines = 1 } = $$props;
  let { i18n } = $$props;
  let { placeholder = "Type here..." } = $$props;
  let { disabled = false } = $$props;
  let { label } = $$props;
  let { info = void 0 } = $$props;
  let { show_label = true } = $$props;
  let { max_lines } = $$props;
  let { submit_btn = null } = $$props;
  let { stop_btn = null } = $$props;
  let { rtl = false } = $$props;
  let { autofocus = false } = $$props;
  let { text_align = void 0 } = $$props;
  let { autoscroll = true } = $$props;
  let { root } = $$props;
  let { file_types = null } = $$props;
  let { max_file_size = null } = $$props;
  let { upload } = $$props;
  let { stream_handler } = $$props;
  let { file_count = "multiple" } = $$props;
  let { max_plain_text_length = 1e3 } = $$props;
  let { waveform_settings } = $$props;
  let { waveform_options = { show_recording_waveform: true } } = $$props;
  let { sources = ["upload"] } = $$props;
  let { active_source = null } = $$props;
  let upload_component;
  let hidden_upload;
  let el;
  let can_scroll;
  let previous_scroll_top = 0;
  let user_has_scrolled_up = false;
  let { dragging = false } = $$props;
  let uploading = false;
  let oldValue = value.text;
  let mic_audio = null;
  let full_container;
  const dispatch = createEventDispatcher();
  beforeUpdate(() => {
    can_scroll = el && el.offsetHeight + el.scrollTop > el.scrollHeight - 100;
  });
  const scroll = () => {
    if (can_scroll && autoscroll && !user_has_scrolled_up) {
      el.scrollTo(0, el.scrollHeight);
    }
  };
  async function handle_change() {
    dispatch("change", value);
    if (!value_is_output) {
      dispatch("input");
    }
  }
  onMount(() => {
    if (autofocus && el !== null) {
      el.focus();
    }
  });
  afterUpdate(() => {
    if (can_scroll && autoscroll) {
      scroll();
    }
    $$invalidate(44, value_is_output = false);
  });
  function handle_select(event) {
    const target = event.target;
    const text2 = target.value;
    const index = [target.selectionStart, target.selectionEnd];
    dispatch("select", { value: text2.substring(...index), index });
  }
  async function handle_keypress(e) {
    await tick();
    if (e.key === "Enter" && e.shiftKey && lines > 1) {
      e.preventDefault();
      dispatch("submit");
    } else if (e.key === "Enter" && !e.shiftKey && lines === 1 && max_lines >= 1) {
      e.preventDefault();
      dispatch("submit");
      $$invalidate(2, active_source = null);
      if (mic_audio) {
        value.files.push(mic_audio);
        $$invalidate(0, value);
        $$invalidate(29, mic_audio = null);
      }
    }
  }
  function handle_scroll(event) {
    const target = event.target;
    const current_scroll_top = target.scrollTop;
    if (current_scroll_top < previous_scroll_top) {
      user_has_scrolled_up = true;
    }
    previous_scroll_top = current_scroll_top;
    const max_scroll_top = target.scrollHeight - target.clientHeight;
    const user_has_scrolled_to_bottom = current_scroll_top >= max_scroll_top;
    if (user_has_scrolled_to_bottom) {
      user_has_scrolled_up = false;
    }
  }
  async function handle_upload({ detail }) {
    handle_change();
    if (Array.isArray(detail)) {
      for (let file of detail) {
        value.files.push(file);
      }
      $$invalidate(0, value);
    } else {
      value.files.push(detail);
      $$invalidate(0, value);
    }
    await tick();
    dispatch("change", value);
    dispatch("upload", detail);
  }
  function remove_thumbnail(event, index) {
    handle_change();
    event.stopPropagation();
    value.files.splice(index, 1);
    $$invalidate(0, value);
  }
  function handle_upload_click() {
    if (hidden_upload) {
      $$invalidate(27, hidden_upload.value = "", hidden_upload);
      hidden_upload.click();
    }
  }
  function handle_stop() {
    dispatch("stop");
  }
  function handle_submit() {
    dispatch("submit");
    $$invalidate(2, active_source = null);
    if (mic_audio) {
      value.files.push(mic_audio);
      $$invalidate(0, value);
      $$invalidate(29, mic_audio = null);
    }
  }
  async function handle_paste(event) {
    if (!event.clipboardData)
      return;
    const items = event.clipboardData.items;
    const text2 = event.clipboardData.getData("text");
    if (text2 && text2.length > max_plain_text_length) {
      event.preventDefault();
      const file = new window.File(
        [text2],
        "pasted_text.txt",
        {
          type: "text/plain",
          lastModified: Date.now()
        }
      );
      if (upload_component) {
        upload_component.load_files([file]);
      }
      return;
    }
    for (let index in items) {
      const item = items[index];
      if (item.kind === "file" && item.type.includes("image")) {
        const blob = item.getAsFile();
        if (blob)
          upload_component.load_files([blob]);
      }
    }
  }
  function handle_dragenter(event) {
    event.preventDefault();
    $$invalidate(1, dragging = true);
  }
  function handle_dragleave(event) {
    event.preventDefault();
    const rect = full_container.getBoundingClientRect();
    const { clientX, clientY } = event;
    if (clientX <= rect.left || clientX >= rect.right || clientY <= rect.top || clientY >= rect.bottom) {
      $$invalidate(1, dragging = false);
    }
  }
  function handle_drop(event) {
    event.preventDefault();
    $$invalidate(1, dragging = false);
    if (event.dataTransfer && event.dataTransfer.files) {
      const files = Array.from(event.dataTransfer.files);
      if (file_types) {
        const valid_files = files.filter((file) => {
          return file_types.some((type) => {
            if (type.startsWith(".")) {
              return file.name.toLowerCase().endsWith(type.toLowerCase());
            }
            return file.type.match(new RegExp(type.replace("*", ".*")));
          });
        });
        const invalid_files = files.length - valid_files.length;
        if (invalid_files > 0) {
          dispatch("error", `${invalid_files} file(s) were rejected. Accepted formats: ${file_types.join(", ")}`);
        }
        if (valid_files.length > 0) {
          upload_component.load_files(valid_files);
        }
      } else {
        upload_component.load_files(files);
      }
    }
  }
  function dragover_handler(event) {
    bubble.call(this, $$self, event);
  }
  function blur_handler(event) {
    bubble.call(this, $$self, event);
  }
  function focus_handler(event) {
    bubble.call(this, $$self, event);
  }
  const click_handler = (index, event) => remove_thumbnail(event, index);
  const change_handler = ({ detail }) => {
    if (detail !== null) {
      $$invalidate(29, mic_audio = detail);
    }
  };
  const clear_handler = () => {
    $$invalidate(2, active_source = null);
  };
  const start_recording_handler = () => dispatch("start_recording");
  const pause_recording_handler = () => dispatch("pause_recording");
  const stop_recording_handler = () => dispatch("stop_recording");
  function upload_1_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      upload_component = $$value;
      $$invalidate(26, upload_component);
    });
  }
  function upload_1_dragging_binding(value2) {
    dragging = value2;
    $$invalidate(1, dragging);
  }
  function upload_1_uploading_binding(value2) {
    uploading = value2;
    $$invalidate(28, uploading);
  }
  function upload_1_hidden_upload_binding(value2) {
    hidden_upload = value2;
    $$invalidate(27, hidden_upload);
  }
  function error_handler(event) {
    bubble.call(this, $$self, event);
  }
  const click_handler_1 = () => {
    $$invalidate(2, active_source = active_source !== "microphone" ? "microphone" : null);
  };
  function textarea_input_handler() {
    value.text = this.value;
    $$invalidate(0, value);
  }
  function textarea_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      el = $$value;
      $$invalidate(25, el);
    });
  }
  function div1_binding($$value) {
    binding_callbacks[$$value ? "unshift" : "push"](() => {
      full_container = $$value;
      $$invalidate(30, full_container);
    });
  }
  $$self.$$set = ($$props2) => {
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("value_is_output" in $$props2)
      $$invalidate(44, value_is_output = $$props2.value_is_output);
    if ("lines" in $$props2)
      $$invalidate(3, lines = $$props2.lines);
    if ("i18n" in $$props2)
      $$invalidate(4, i18n = $$props2.i18n);
    if ("placeholder" in $$props2)
      $$invalidate(5, placeholder = $$props2.placeholder);
    if ("disabled" in $$props2)
      $$invalidate(6, disabled = $$props2.disabled);
    if ("label" in $$props2)
      $$invalidate(7, label = $$props2.label);
    if ("info" in $$props2)
      $$invalidate(8, info = $$props2.info);
    if ("show_label" in $$props2)
      $$invalidate(9, show_label = $$props2.show_label);
    if ("max_lines" in $$props2)
      $$invalidate(10, max_lines = $$props2.max_lines);
    if ("submit_btn" in $$props2)
      $$invalidate(11, submit_btn = $$props2.submit_btn);
    if ("stop_btn" in $$props2)
      $$invalidate(12, stop_btn = $$props2.stop_btn);
    if ("rtl" in $$props2)
      $$invalidate(13, rtl = $$props2.rtl);
    if ("autofocus" in $$props2)
      $$invalidate(14, autofocus = $$props2.autofocus);
    if ("text_align" in $$props2)
      $$invalidate(15, text_align = $$props2.text_align);
    if ("autoscroll" in $$props2)
      $$invalidate(45, autoscroll = $$props2.autoscroll);
    if ("root" in $$props2)
      $$invalidate(16, root = $$props2.root);
    if ("file_types" in $$props2)
      $$invalidate(17, file_types = $$props2.file_types);
    if ("max_file_size" in $$props2)
      $$invalidate(18, max_file_size = $$props2.max_file_size);
    if ("upload" in $$props2)
      $$invalidate(19, upload = $$props2.upload);
    if ("stream_handler" in $$props2)
      $$invalidate(20, stream_handler = $$props2.stream_handler);
    if ("file_count" in $$props2)
      $$invalidate(21, file_count = $$props2.file_count);
    if ("max_plain_text_length" in $$props2)
      $$invalidate(46, max_plain_text_length = $$props2.max_plain_text_length);
    if ("waveform_settings" in $$props2)
      $$invalidate(22, waveform_settings = $$props2.waveform_settings);
    if ("waveform_options" in $$props2)
      $$invalidate(23, waveform_options = $$props2.waveform_options);
    if ("sources" in $$props2)
      $$invalidate(24, sources = $$props2.sources);
    if ("active_source" in $$props2)
      $$invalidate(2, active_source = $$props2.active_source);
    if ("dragging" in $$props2)
      $$invalidate(1, dragging = $$props2.dragging);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*dragging*/
    2) {
      dispatch("drag", dragging);
    }
    if ($$self.$$.dirty[0] & /*value*/
    1) {
      if (value === null)
        $$invalidate(0, value = { text: "", files: [] });
    }
    if ($$self.$$.dirty[0] & /*value*/
    1 | $$self.$$.dirty[1] & /*oldValue*/
    65536) {
      if (oldValue !== value.text) {
        dispatch("change", value);
        $$invalidate(47, oldValue = value.text);
      }
    }
    if ($$self.$$.dirty[0] & /*value, el, lines, max_lines*/
    33555465) {
      el && lines !== max_lines && resize(el, lines, max_lines);
    }
  };
  return [
    value,
    dragging,
    active_source,
    lines,
    i18n,
    placeholder,
    disabled,
    label,
    info,
    show_label,
    max_lines,
    submit_btn,
    stop_btn,
    rtl,
    autofocus,
    text_align,
    root,
    file_types,
    max_file_size,
    upload,
    stream_handler,
    file_count,
    waveform_settings,
    waveform_options,
    sources,
    el,
    upload_component,
    hidden_upload,
    uploading,
    mic_audio,
    full_container,
    dispatch,
    handle_select,
    handle_keypress,
    handle_scroll,
    handle_upload,
    remove_thumbnail,
    handle_upload_click,
    handle_stop,
    handle_submit,
    handle_paste,
    handle_dragenter,
    handle_dragleave,
    handle_drop,
    value_is_output,
    autoscroll,
    max_plain_text_length,
    oldValue,
    dragover_handler,
    blur_handler,
    focus_handler,
    click_handler,
    change_handler,
    clear_handler,
    start_recording_handler,
    pause_recording_handler,
    stop_recording_handler,
    upload_1_binding,
    upload_1_dragging_binding,
    upload_1_uploading_binding,
    upload_1_hidden_upload_binding,
    error_handler,
    click_handler_1,
    textarea_input_handler,
    textarea_binding,
    div1_binding
  ];
}
class MultimodalTextbox extends SvelteComponent {
  constructor(options) {
    super();
    init(
      this,
      options,
      instance$1,
      create_fragment$1,
      safe_not_equal,
      {
        value: 0,
        value_is_output: 44,
        lines: 3,
        i18n: 4,
        placeholder: 5,
        disabled: 6,
        label: 7,
        info: 8,
        show_label: 9,
        max_lines: 10,
        submit_btn: 11,
        stop_btn: 12,
        rtl: 13,
        autofocus: 14,
        text_align: 15,
        autoscroll: 45,
        root: 16,
        file_types: 17,
        max_file_size: 18,
        upload: 19,
        stream_handler: 20,
        file_count: 21,
        max_plain_text_length: 46,
        waveform_settings: 22,
        waveform_options: 23,
        sources: 24,
        active_source: 2,
        dragging: 1
      },
      null,
      [-1, -1, -1]
    );
  }
}
const MultimodalTextbox$1 = MultimodalTextbox;
function create_if_block(ctx) {
  let statustracker;
  let current;
  const statustracker_spread_levels = [
    { autoscroll: (
      /*gradio*/
      ctx[2].autoscroll
    ) },
    { i18n: (
      /*gradio*/
      ctx[2].i18n
    ) },
    /*loading_status*/
    ctx[17]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[31]
  );
  return {
    c() {
      create_component(statustracker.$$.fragment);
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty[0] & /*gradio, loading_status*/
      131076 ? get_spread_update(statustracker_spread_levels, [
        dirty[0] & /*gradio*/
        4 && { autoscroll: (
          /*gradio*/
          ctx2[2].autoscroll
        ) },
        dirty[0] & /*gradio*/
        4 && { i18n: (
          /*gradio*/
          ctx2[2].i18n
        ) },
        dirty[0] & /*loading_status*/
        131072 && get_spread_object(
          /*loading_status*/
          ctx2[17]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(statustracker, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let t;
  let multimodaltextbox;
  let updating_value;
  let updating_value_is_output;
  let updating_dragging;
  let updating_active_source;
  let current;
  let if_block = (
    /*loading_status*/
    ctx[17] && create_if_block(ctx)
  );
  function multimodaltextbox_value_binding(value) {
    ctx[34](value);
  }
  function multimodaltextbox_value_is_output_binding(value) {
    ctx[35](value);
  }
  function multimodaltextbox_dragging_binding(value) {
    ctx[36](value);
  }
  function multimodaltextbox_active_source_binding(value) {
    ctx[37](value);
  }
  let multimodaltextbox_props = {
    file_types: (
      /*file_types*/
      ctx[6]
    ),
    root: (
      /*root*/
      ctx[23]
    ),
    label: (
      /*label*/
      ctx[9]
    ),
    info: (
      /*info*/
      ctx[10]
    ),
    show_label: (
      /*show_label*/
      ctx[11]
    ),
    lines: (
      /*lines*/
      ctx[7]
    ),
    rtl: (
      /*rtl*/
      ctx[18]
    ),
    text_align: (
      /*text_align*/
      ctx[19]
    ),
    waveform_settings: (
      /*waveform_settings*/
      ctx[29]
    ),
    i18n: (
      /*gradio*/
      ctx[2].i18n
    ),
    max_lines: !/*max_lines*/
    ctx[12] ? (
      /*lines*/
      ctx[7] + 1
    ) : (
      /*max_lines*/
      ctx[12]
    ),
    placeholder: (
      /*placeholder*/
      ctx[8]
    ),
    submit_btn: (
      /*submit_btn*/
      ctx[15]
    ),
    stop_btn: (
      /*stop_btn*/
      ctx[16]
    ),
    autofocus: (
      /*autofocus*/
      ctx[20]
    ),
    autoscroll: (
      /*autoscroll*/
      ctx[21]
    ),
    file_count: (
      /*file_count*/
      ctx[24]
    ),
    sources: (
      /*sources*/
      ctx[26]
    ),
    max_file_size: (
      /*gradio*/
      ctx[2].max_file_size
    ),
    disabled: !/*interactive*/
    ctx[22],
    upload: (
      /*func*/
      ctx[32]
    ),
    stream_handler: (
      /*func_1*/
      ctx[33]
    ),
    max_plain_text_length: (
      /*max_plain_text_length*/
      ctx[25]
    )
  };
  if (
    /*value*/
    ctx[0] !== void 0
  ) {
    multimodaltextbox_props.value = /*value*/
    ctx[0];
  }
  if (
    /*value_is_output*/
    ctx[1] !== void 0
  ) {
    multimodaltextbox_props.value_is_output = /*value_is_output*/
    ctx[1];
  }
  if (
    /*dragging*/
    ctx[27] !== void 0
  ) {
    multimodaltextbox_props.dragging = /*dragging*/
    ctx[27];
  }
  if (
    /*active_source*/
    ctx[28] !== void 0
  ) {
    multimodaltextbox_props.active_source = /*active_source*/
    ctx[28];
  }
  multimodaltextbox = new MultimodalTextbox$1({ props: multimodaltextbox_props });
  binding_callbacks.push(() => bind(multimodaltextbox, "value", multimodaltextbox_value_binding));
  binding_callbacks.push(() => bind(multimodaltextbox, "value_is_output", multimodaltextbox_value_is_output_binding));
  binding_callbacks.push(() => bind(multimodaltextbox, "dragging", multimodaltextbox_dragging_binding));
  binding_callbacks.push(() => bind(multimodaltextbox, "active_source", multimodaltextbox_active_source_binding));
  multimodaltextbox.$on(
    "change",
    /*change_handler*/
    ctx[38]
  );
  multimodaltextbox.$on(
    "input",
    /*input_handler*/
    ctx[39]
  );
  multimodaltextbox.$on(
    "submit",
    /*submit_handler*/
    ctx[40]
  );
  multimodaltextbox.$on(
    "stop",
    /*stop_handler*/
    ctx[41]
  );
  multimodaltextbox.$on(
    "blur",
    /*blur_handler*/
    ctx[42]
  );
  multimodaltextbox.$on(
    "select",
    /*select_handler*/
    ctx[43]
  );
  multimodaltextbox.$on(
    "focus",
    /*focus_handler*/
    ctx[44]
  );
  multimodaltextbox.$on(
    "error",
    /*error_handler*/
    ctx[45]
  );
  multimodaltextbox.$on(
    "start_recording",
    /*start_recording_handler*/
    ctx[46]
  );
  multimodaltextbox.$on(
    "pause_recording",
    /*pause_recording_handler*/
    ctx[47]
  );
  multimodaltextbox.$on(
    "stop_recording",
    /*stop_recording_handler*/
    ctx[48]
  );
  multimodaltextbox.$on(
    "upload",
    /*upload_handler*/
    ctx[49]
  );
  multimodaltextbox.$on(
    "clear",
    /*clear_handler*/
    ctx[50]
  );
  return {
    c() {
      if (if_block)
        if_block.c();
      t = space();
      create_component(multimodaltextbox.$$.fragment);
    },
    l(nodes) {
      if (if_block)
        if_block.l(nodes);
      t = claim_space(nodes);
      claim_component(multimodaltextbox.$$.fragment, nodes);
    },
    m(target, anchor) {
      if (if_block)
        if_block.m(target, anchor);
      insert_hydration(target, t, anchor);
      mount_component(multimodaltextbox, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      if (
        /*loading_status*/
        ctx2[17]
      ) {
        if (if_block) {
          if_block.p(ctx2, dirty);
          if (dirty[0] & /*loading_status*/
          131072) {
            transition_in(if_block, 1);
          }
        } else {
          if_block = create_if_block(ctx2);
          if_block.c();
          transition_in(if_block, 1);
          if_block.m(t.parentNode, t);
        }
      } else if (if_block) {
        group_outros();
        transition_out(if_block, 1, 1, () => {
          if_block = null;
        });
        check_outros();
      }
      const multimodaltextbox_changes = {};
      if (dirty[0] & /*file_types*/
      64)
        multimodaltextbox_changes.file_types = /*file_types*/
        ctx2[6];
      if (dirty[0] & /*root*/
      8388608)
        multimodaltextbox_changes.root = /*root*/
        ctx2[23];
      if (dirty[0] & /*label*/
      512)
        multimodaltextbox_changes.label = /*label*/
        ctx2[9];
      if (dirty[0] & /*info*/
      1024)
        multimodaltextbox_changes.info = /*info*/
        ctx2[10];
      if (dirty[0] & /*show_label*/
      2048)
        multimodaltextbox_changes.show_label = /*show_label*/
        ctx2[11];
      if (dirty[0] & /*lines*/
      128)
        multimodaltextbox_changes.lines = /*lines*/
        ctx2[7];
      if (dirty[0] & /*rtl*/
      262144)
        multimodaltextbox_changes.rtl = /*rtl*/
        ctx2[18];
      if (dirty[0] & /*text_align*/
      524288)
        multimodaltextbox_changes.text_align = /*text_align*/
        ctx2[19];
      if (dirty[0] & /*waveform_settings*/
      536870912)
        multimodaltextbox_changes.waveform_settings = /*waveform_settings*/
        ctx2[29];
      if (dirty[0] & /*gradio*/
      4)
        multimodaltextbox_changes.i18n = /*gradio*/
        ctx2[2].i18n;
      if (dirty[0] & /*max_lines, lines*/
      4224)
        multimodaltextbox_changes.max_lines = !/*max_lines*/
        ctx2[12] ? (
          /*lines*/
          ctx2[7] + 1
        ) : (
          /*max_lines*/
          ctx2[12]
        );
      if (dirty[0] & /*placeholder*/
      256)
        multimodaltextbox_changes.placeholder = /*placeholder*/
        ctx2[8];
      if (dirty[0] & /*submit_btn*/
      32768)
        multimodaltextbox_changes.submit_btn = /*submit_btn*/
        ctx2[15];
      if (dirty[0] & /*stop_btn*/
      65536)
        multimodaltextbox_changes.stop_btn = /*stop_btn*/
        ctx2[16];
      if (dirty[0] & /*autofocus*/
      1048576)
        multimodaltextbox_changes.autofocus = /*autofocus*/
        ctx2[20];
      if (dirty[0] & /*autoscroll*/
      2097152)
        multimodaltextbox_changes.autoscroll = /*autoscroll*/
        ctx2[21];
      if (dirty[0] & /*file_count*/
      16777216)
        multimodaltextbox_changes.file_count = /*file_count*/
        ctx2[24];
      if (dirty[0] & /*sources*/
      67108864)
        multimodaltextbox_changes.sources = /*sources*/
        ctx2[26];
      if (dirty[0] & /*gradio*/
      4)
        multimodaltextbox_changes.max_file_size = /*gradio*/
        ctx2[2].max_file_size;
      if (dirty[0] & /*interactive*/
      4194304)
        multimodaltextbox_changes.disabled = !/*interactive*/
        ctx2[22];
      if (dirty[0] & /*gradio*/
      4)
        multimodaltextbox_changes.upload = /*func*/
        ctx2[32];
      if (dirty[0] & /*gradio*/
      4)
        multimodaltextbox_changes.stream_handler = /*func_1*/
        ctx2[33];
      if (dirty[0] & /*max_plain_text_length*/
      33554432)
        multimodaltextbox_changes.max_plain_text_length = /*max_plain_text_length*/
        ctx2[25];
      if (!updating_value && dirty[0] & /*value*/
      1) {
        updating_value = true;
        multimodaltextbox_changes.value = /*value*/
        ctx2[0];
        add_flush_callback(() => updating_value = false);
      }
      if (!updating_value_is_output && dirty[0] & /*value_is_output*/
      2) {
        updating_value_is_output = true;
        multimodaltextbox_changes.value_is_output = /*value_is_output*/
        ctx2[1];
        add_flush_callback(() => updating_value_is_output = false);
      }
      if (!updating_dragging && dirty[0] & /*dragging*/
      134217728) {
        updating_dragging = true;
        multimodaltextbox_changes.dragging = /*dragging*/
        ctx2[27];
        add_flush_callback(() => updating_dragging = false);
      }
      if (!updating_active_source && dirty[0] & /*active_source*/
      268435456) {
        updating_active_source = true;
        multimodaltextbox_changes.active_source = /*active_source*/
        ctx2[28];
        add_flush_callback(() => updating_active_source = false);
      }
      multimodaltextbox.$set(multimodaltextbox_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block);
      transition_in(multimodaltextbox.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(if_block);
      transition_out(multimodaltextbox.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
      if (if_block)
        if_block.d(detaching);
      destroy_component(multimodaltextbox, detaching);
    }
  };
}
function create_fragment(ctx) {
  let block;
  let current;
  block = new Block({
    props: {
      visible: (
        /*visible*/
        ctx[5]
      ),
      elem_id: (
        /*elem_id*/
        ctx[3]
      ),
      elem_classes: [.../*elem_classes*/
      ctx[4], "multimodal-textbox"],
      scale: (
        /*scale*/
        ctx[13]
      ),
      min_width: (
        /*min_width*/
        ctx[14]
      ),
      allow_overflow: false,
      padding: false,
      border_mode: (
        /*dragging*/
        ctx[27] ? "focus" : "base"
      ),
      $$slots: { default: [create_default_slot] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      create_component(block.$$.fragment);
    },
    l(nodes) {
      claim_component(block.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(block, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const block_changes = {};
      if (dirty[0] & /*visible*/
      32)
        block_changes.visible = /*visible*/
        ctx2[5];
      if (dirty[0] & /*elem_id*/
      8)
        block_changes.elem_id = /*elem_id*/
        ctx2[3];
      if (dirty[0] & /*elem_classes*/
      16)
        block_changes.elem_classes = [.../*elem_classes*/
        ctx2[4], "multimodal-textbox"];
      if (dirty[0] & /*scale*/
      8192)
        block_changes.scale = /*scale*/
        ctx2[13];
      if (dirty[0] & /*min_width*/
      16384)
        block_changes.min_width = /*min_width*/
        ctx2[14];
      if (dirty[0] & /*dragging*/
      134217728)
        block_changes.border_mode = /*dragging*/
        ctx2[27] ? "focus" : "base";
      if (dirty[0] & /*file_types, root, label, info, show_label, lines, rtl, text_align, waveform_settings, gradio, max_lines, placeholder, submit_btn, stop_btn, autofocus, autoscroll, file_count, sources, interactive, max_plain_text_length, value, value_is_output, dragging, active_source, loading_status*/
      1073717191 | dirty[1] & /*$$scope*/
      8388608) {
        block_changes.$$scope = { dirty, ctx: ctx2 };
      }
      block.$set(block_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(block.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(block.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(block, detaching);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
  let { gradio } = $$props;
  let { elem_id = "" } = $$props;
  let { elem_classes = [] } = $$props;
  let { visible = true } = $$props;
  let { value = { text: "", files: [] } } = $$props;
  let { file_types = null } = $$props;
  let { lines } = $$props;
  let { placeholder = "" } = $$props;
  let { label = "MultimodalTextbox" } = $$props;
  let { info = void 0 } = $$props;
  let { show_label } = $$props;
  let { max_lines } = $$props;
  let { scale = null } = $$props;
  let { min_width = void 0 } = $$props;
  let { submit_btn = null } = $$props;
  let { stop_btn = null } = $$props;
  let { loading_status = void 0 } = $$props;
  let { value_is_output = false } = $$props;
  let { rtl = false } = $$props;
  let { text_align = void 0 } = $$props;
  let { autofocus = false } = $$props;
  let { autoscroll = true } = $$props;
  let { interactive } = $$props;
  let { root } = $$props;
  let { file_count } = $$props;
  let { max_plain_text_length } = $$props;
  let { sources = ["upload"] } = $$props;
  let { waveform_options = {} } = $$props;
  let dragging;
  let active_source = null;
  let waveform_settings;
  let color_accent = "darkorange";
  onMount(() => {
    color_accent = getComputedStyle(document == null ? void 0 : document.documentElement).getPropertyValue("--color-accent");
    set_trim_region_colour();
    $$invalidate(29, waveform_settings.waveColor = waveform_options.waveform_color || "#9ca3af", waveform_settings);
    $$invalidate(29, waveform_settings.progressColor = waveform_options.waveform_progress_color || color_accent, waveform_settings);
    $$invalidate(29, waveform_settings.mediaControls = waveform_options.show_controls, waveform_settings);
    $$invalidate(29, waveform_settings.sampleRate = waveform_options.sample_rate || 44100, waveform_settings);
  });
  const trim_region_settings = {
    color: waveform_options.trim_region_color,
    drag: true,
    resize: true
  };
  function set_trim_region_colour() {
    document.documentElement.style.setProperty("--trim-region-color", trim_region_settings.color || color_accent);
  }
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  const func = (...args) => gradio.client.upload(...args);
  const func_1 = (...args) => gradio.client.stream(...args);
  function multimodaltextbox_value_binding(value$1) {
    value = value$1;
    $$invalidate(0, value);
  }
  function multimodaltextbox_value_is_output_binding(value2) {
    value_is_output = value2;
    $$invalidate(1, value_is_output);
  }
  function multimodaltextbox_dragging_binding(value2) {
    dragging = value2;
    $$invalidate(27, dragging);
  }
  function multimodaltextbox_active_source_binding(value2) {
    active_source = value2;
    $$invalidate(28, active_source);
  }
  const change_handler = () => gradio.dispatch("change", value);
  const input_handler = () => gradio.dispatch("input");
  const submit_handler = () => gradio.dispatch("submit");
  const stop_handler = () => gradio.dispatch("stop");
  const blur_handler = () => gradio.dispatch("blur");
  const select_handler = (e) => gradio.dispatch("select", e.detail);
  const focus_handler = () => gradio.dispatch("focus");
  const error_handler = ({ detail }) => {
    gradio.dispatch("error", detail);
  };
  const start_recording_handler = () => gradio.dispatch("start_recording");
  const pause_recording_handler = () => gradio.dispatch("pause_recording");
  const stop_recording_handler = () => gradio.dispatch("stop_recording");
  const upload_handler = (e) => gradio.dispatch("upload", e.detail);
  const clear_handler = () => gradio.dispatch("clear");
  $$self.$$set = ($$props2) => {
    if ("gradio" in $$props2)
      $$invalidate(2, gradio = $$props2.gradio);
    if ("elem_id" in $$props2)
      $$invalidate(3, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(4, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(5, visible = $$props2.visible);
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("file_types" in $$props2)
      $$invalidate(6, file_types = $$props2.file_types);
    if ("lines" in $$props2)
      $$invalidate(7, lines = $$props2.lines);
    if ("placeholder" in $$props2)
      $$invalidate(8, placeholder = $$props2.placeholder);
    if ("label" in $$props2)
      $$invalidate(9, label = $$props2.label);
    if ("info" in $$props2)
      $$invalidate(10, info = $$props2.info);
    if ("show_label" in $$props2)
      $$invalidate(11, show_label = $$props2.show_label);
    if ("max_lines" in $$props2)
      $$invalidate(12, max_lines = $$props2.max_lines);
    if ("scale" in $$props2)
      $$invalidate(13, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(14, min_width = $$props2.min_width);
    if ("submit_btn" in $$props2)
      $$invalidate(15, submit_btn = $$props2.submit_btn);
    if ("stop_btn" in $$props2)
      $$invalidate(16, stop_btn = $$props2.stop_btn);
    if ("loading_status" in $$props2)
      $$invalidate(17, loading_status = $$props2.loading_status);
    if ("value_is_output" in $$props2)
      $$invalidate(1, value_is_output = $$props2.value_is_output);
    if ("rtl" in $$props2)
      $$invalidate(18, rtl = $$props2.rtl);
    if ("text_align" in $$props2)
      $$invalidate(19, text_align = $$props2.text_align);
    if ("autofocus" in $$props2)
      $$invalidate(20, autofocus = $$props2.autofocus);
    if ("autoscroll" in $$props2)
      $$invalidate(21, autoscroll = $$props2.autoscroll);
    if ("interactive" in $$props2)
      $$invalidate(22, interactive = $$props2.interactive);
    if ("root" in $$props2)
      $$invalidate(23, root = $$props2.root);
    if ("file_count" in $$props2)
      $$invalidate(24, file_count = $$props2.file_count);
    if ("max_plain_text_length" in $$props2)
      $$invalidate(25, max_plain_text_length = $$props2.max_plain_text_length);
    if ("sources" in $$props2)
      $$invalidate(26, sources = $$props2.sources);
    if ("waveform_options" in $$props2)
      $$invalidate(30, waveform_options = $$props2.waveform_options);
  };
  $$invalidate(29, waveform_settings = {
    height: 50,
    barWidth: 2,
    barGap: 3,
    cursorWidth: 2,
    cursorColor: "#ddd5e9",
    autoplay: false,
    barRadius: 10,
    dragToSeek: true,
    normalize: true,
    minPxPerSec: 20
  });
  return [
    value,
    value_is_output,
    gradio,
    elem_id,
    elem_classes,
    visible,
    file_types,
    lines,
    placeholder,
    label,
    info,
    show_label,
    max_lines,
    scale,
    min_width,
    submit_btn,
    stop_btn,
    loading_status,
    rtl,
    text_align,
    autofocus,
    autoscroll,
    interactive,
    root,
    file_count,
    max_plain_text_length,
    sources,
    dragging,
    active_source,
    waveform_settings,
    waveform_options,
    clear_status_handler,
    func,
    func_1,
    multimodaltextbox_value_binding,
    multimodaltextbox_value_is_output_binding,
    multimodaltextbox_dragging_binding,
    multimodaltextbox_active_source_binding,
    change_handler,
    input_handler,
    submit_handler,
    stop_handler,
    blur_handler,
    select_handler,
    focus_handler,
    error_handler,
    start_recording_handler,
    pause_recording_handler,
    stop_recording_handler,
    upload_handler,
    clear_handler
  ];
}
class Index extends SvelteComponent {
  constructor(options) {
    super();
    init(
      this,
      options,
      instance,
      create_fragment,
      safe_not_equal,
      {
        gradio: 2,
        elem_id: 3,
        elem_classes: 4,
        visible: 5,
        value: 0,
        file_types: 6,
        lines: 7,
        placeholder: 8,
        label: 9,
        info: 10,
        show_label: 11,
        max_lines: 12,
        scale: 13,
        min_width: 14,
        submit_btn: 15,
        stop_btn: 16,
        loading_status: 17,
        value_is_output: 1,
        rtl: 18,
        text_align: 19,
        autofocus: 20,
        autoscroll: 21,
        interactive: 22,
        root: 23,
        file_count: 24,
        max_plain_text_length: 25,
        sources: 26,
        waveform_options: 30
      },
      null,
      [-1, -1]
    );
  }
  get gradio() {
    return this.$$.ctx[2];
  }
  set gradio(gradio) {
    this.$$set({ gradio });
    flush();
  }
  get elem_id() {
    return this.$$.ctx[3];
  }
  set elem_id(elem_id) {
    this.$$set({ elem_id });
    flush();
  }
  get elem_classes() {
    return this.$$.ctx[4];
  }
  set elem_classes(elem_classes) {
    this.$$set({ elem_classes });
    flush();
  }
  get visible() {
    return this.$$.ctx[5];
  }
  set visible(visible) {
    this.$$set({ visible });
    flush();
  }
  get value() {
    return this.$$.ctx[0];
  }
  set value(value) {
    this.$$set({ value });
    flush();
  }
  get file_types() {
    return this.$$.ctx[6];
  }
  set file_types(file_types) {
    this.$$set({ file_types });
    flush();
  }
  get lines() {
    return this.$$.ctx[7];
  }
  set lines(lines) {
    this.$$set({ lines });
    flush();
  }
  get placeholder() {
    return this.$$.ctx[8];
  }
  set placeholder(placeholder) {
    this.$$set({ placeholder });
    flush();
  }
  get label() {
    return this.$$.ctx[9];
  }
  set label(label) {
    this.$$set({ label });
    flush();
  }
  get info() {
    return this.$$.ctx[10];
  }
  set info(info) {
    this.$$set({ info });
    flush();
  }
  get show_label() {
    return this.$$.ctx[11];
  }
  set show_label(show_label) {
    this.$$set({ show_label });
    flush();
  }
  get max_lines() {
    return this.$$.ctx[12];
  }
  set max_lines(max_lines) {
    this.$$set({ max_lines });
    flush();
  }
  get scale() {
    return this.$$.ctx[13];
  }
  set scale(scale) {
    this.$$set({ scale });
    flush();
  }
  get min_width() {
    return this.$$.ctx[14];
  }
  set min_width(min_width) {
    this.$$set({ min_width });
    flush();
  }
  get submit_btn() {
    return this.$$.ctx[15];
  }
  set submit_btn(submit_btn) {
    this.$$set({ submit_btn });
    flush();
  }
  get stop_btn() {
    return this.$$.ctx[16];
  }
  set stop_btn(stop_btn) {
    this.$$set({ stop_btn });
    flush();
  }
  get loading_status() {
    return this.$$.ctx[17];
  }
  set loading_status(loading_status) {
    this.$$set({ loading_status });
    flush();
  }
  get value_is_output() {
    return this.$$.ctx[1];
  }
  set value_is_output(value_is_output) {
    this.$$set({ value_is_output });
    flush();
  }
  get rtl() {
    return this.$$.ctx[18];
  }
  set rtl(rtl) {
    this.$$set({ rtl });
    flush();
  }
  get text_align() {
    return this.$$.ctx[19];
  }
  set text_align(text_align) {
    this.$$set({ text_align });
    flush();
  }
  get autofocus() {
    return this.$$.ctx[20];
  }
  set autofocus(autofocus) {
    this.$$set({ autofocus });
    flush();
  }
  get autoscroll() {
    return this.$$.ctx[21];
  }
  set autoscroll(autoscroll) {
    this.$$set({ autoscroll });
    flush();
  }
  get interactive() {
    return this.$$.ctx[22];
  }
  set interactive(interactive) {
    this.$$set({ interactive });
    flush();
  }
  get root() {
    return this.$$.ctx[23];
  }
  set root(root) {
    this.$$set({ root });
    flush();
  }
  get file_count() {
    return this.$$.ctx[24];
  }
  set file_count(file_count) {
    this.$$set({ file_count });
    flush();
  }
  get max_plain_text_length() {
    return this.$$.ctx[25];
  }
  set max_plain_text_length(max_plain_text_length) {
    this.$$set({ max_plain_text_length });
    flush();
  }
  get sources() {
    return this.$$.ctx[26];
  }
  set sources(sources) {
    this.$$set({ sources });
    flush();
  }
  get waveform_options() {
    return this.$$.ctx[30];
  }
  set waveform_options(waveform_options) {
    this.$$set({ waveform_options });
    flush();
  }
}
export {
  default2 as BaseExample,
  MultimodalTextbox$1 as BaseMultimodalTextbox,
  Index as default
};
