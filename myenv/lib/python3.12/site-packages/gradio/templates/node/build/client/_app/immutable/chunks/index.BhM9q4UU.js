import { SvelteComponent, init, safe_not_equal, flush, empty, insert_hydration, group_outros, transition_out, check_outros, transition_in, detach, onMount, afterUpdate, create_component, claim_component, mount_component, destroy_component, assign, binding_callbacks, bind, space, claim_space, get_spread_update, get_spread_object, add_flush_callback } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import StaticAudio from "./StaticAudio.Ct2rC3UK.js";
import { I as InteractiveAudio } from "./InteractiveAudio.BIw2E6Pd.js";
import { B as Block, S as Static } from "./2.8WKXZUMv.js";
import { U as UploadText } from "./UploadText.hK-RoU4L.js";
import { A } from "./AudioPlayer.CKSIBYDR.js";
import { default as default2 } from "./Example.Ydb40JSe.js";
function create_else_block(ctx) {
  let block;
  let current;
  block = new Block({
    props: {
      variant: (
        /*value*/
        ctx[0] === null && /*active_source*/
        ctx[25] === "upload" ? "dashed" : "solid"
      ),
      border_mode: (
        /*dragging*/
        ctx[27] ? "focus" : "base"
      ),
      padding: false,
      allow_overflow: false,
      elem_id: (
        /*elem_id*/
        ctx[4]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[5]
      ),
      visible: (
        /*visible*/
        ctx[6]
      ),
      container: (
        /*container*/
        ctx[12]
      ),
      scale: (
        /*scale*/
        ctx[13]
      ),
      min_width: (
        /*min_width*/
        ctx[14]
      ),
      $$slots: { default: [create_default_slot_1] },
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
      if (dirty[0] & /*value, active_source*/
      33554433)
        block_changes.variant = /*value*/
        ctx2[0] === null && /*active_source*/
        ctx2[25] === "upload" ? "dashed" : "solid";
      if (dirty[0] & /*dragging*/
      134217728)
        block_changes.border_mode = /*dragging*/
        ctx2[27] ? "focus" : "base";
      if (dirty[0] & /*elem_id*/
      16)
        block_changes.elem_id = /*elem_id*/
        ctx2[4];
      if (dirty[0] & /*elem_classes*/
      32)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[5];
      if (dirty[0] & /*visible*/
      64)
        block_changes.visible = /*visible*/
        ctx2[6];
      if (dirty[0] & /*container*/
      4096)
        block_changes.container = /*container*/
        ctx2[12];
      if (dirty[0] & /*scale*/
      8192)
        block_changes.scale = /*scale*/
        ctx2[13];
      if (dirty[0] & /*min_width*/
      16384)
        block_changes.min_width = /*min_width*/
        ctx2[14];
      if (dirty[0] & /*label, show_label, show_download_button, value, root, sources, active_source, pending, streaming, loop, gradio, editable, waveform_settings, waveform_options, stream_every, recording, dragging, uploading, _modify_stream, set_time_limit, loading_status*/
      536710927 | dirty[2] & /*$$scope*/
      128) {
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
function create_if_block(ctx) {
  let block;
  let current;
  block = new Block({
    props: {
      variant: "solid",
      border_mode: (
        /*dragging*/
        ctx[27] ? "focus" : "base"
      ),
      padding: false,
      allow_overflow: false,
      elem_id: (
        /*elem_id*/
        ctx[4]
      ),
      elem_classes: (
        /*elem_classes*/
        ctx[5]
      ),
      visible: (
        /*visible*/
        ctx[6]
      ),
      container: (
        /*container*/
        ctx[12]
      ),
      scale: (
        /*scale*/
        ctx[13]
      ),
      min_width: (
        /*min_width*/
        ctx[14]
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
      if (dirty[0] & /*dragging*/
      134217728)
        block_changes.border_mode = /*dragging*/
        ctx2[27] ? "focus" : "base";
      if (dirty[0] & /*elem_id*/
      16)
        block_changes.elem_id = /*elem_id*/
        ctx2[4];
      if (dirty[0] & /*elem_classes*/
      32)
        block_changes.elem_classes = /*elem_classes*/
        ctx2[5];
      if (dirty[0] & /*visible*/
      64)
        block_changes.visible = /*visible*/
        ctx2[6];
      if (dirty[0] & /*container*/
      4096)
        block_changes.container = /*container*/
        ctx2[12];
      if (dirty[0] & /*scale*/
      8192)
        block_changes.scale = /*scale*/
        ctx2[13];
      if (dirty[0] & /*min_width*/
      16384)
        block_changes.min_width = /*min_width*/
        ctx2[14];
      if (dirty[0] & /*gradio, show_label, show_download_button, show_share_button, value, label, loop, waveform_settings, waveform_options, editable, loading_status*/
      277842435 | dirty[2] & /*$$scope*/
      128) {
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
function create_default_slot_2(ctx) {
  let uploadtext;
  let current;
  uploadtext = new UploadText({
    props: {
      i18n: (
        /*gradio*/
        ctx[23].i18n
      ),
      type: "audio"
    }
  });
  return {
    c() {
      create_component(uploadtext.$$.fragment);
    },
    l(nodes) {
      claim_component(uploadtext.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(uploadtext, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const uploadtext_changes = {};
      if (dirty[0] & /*gradio*/
      8388608)
        uploadtext_changes.i18n = /*gradio*/
        ctx2[23].i18n;
      uploadtext.$set(uploadtext_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(uploadtext.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(uploadtext.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(uploadtext, detaching);
    }
  };
}
function create_default_slot_1(ctx) {
  let statustracker;
  let t;
  let interactiveaudio;
  let updating_recording;
  let updating_dragging;
  let updating_uploading;
  let updating_modify_stream;
  let updating_set_time_limit;
  let current;
  const statustracker_spread_levels = [
    {
      autoscroll: (
        /*gradio*/
        ctx[23].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      ctx[23].i18n
    ) },
    /*loading_status*/
    ctx[1]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler_1*/
    ctx[45]
  );
  function interactiveaudio_recording_binding(value) {
    ctx[48](value);
  }
  function interactiveaudio_dragging_binding(value) {
    ctx[49](value);
  }
  function interactiveaudio_uploading_binding(value) {
    ctx[50](value);
  }
  function interactiveaudio_modify_stream_binding(value) {
    ctx[51](value);
  }
  function interactiveaudio_set_time_limit_binding(value) {
    ctx[52](value);
  }
  let interactiveaudio_props = {
    label: (
      /*label*/
      ctx[9]
    ),
    show_label: (
      /*show_label*/
      ctx[11]
    ),
    show_download_button: (
      /*show_download_button*/
      ctx[16]
    ),
    value: (
      /*value*/
      ctx[0]
    ),
    root: (
      /*root*/
      ctx[10]
    ),
    sources: (
      /*sources*/
      ctx[8]
    ),
    active_source: (
      /*active_source*/
      ctx[25]
    ),
    pending: (
      /*pending*/
      ctx[20]
    ),
    streaming: (
      /*streaming*/
      ctx[21]
    ),
    loop: (
      /*loop*/
      ctx[15]
    ),
    max_file_size: (
      /*gradio*/
      ctx[23].max_file_size
    ),
    handle_reset_value: (
      /*handle_reset_value*/
      ctx[29]
    ),
    editable: (
      /*editable*/
      ctx[18]
    ),
    i18n: (
      /*gradio*/
      ctx[23].i18n
    ),
    waveform_settings: (
      /*waveform_settings*/
      ctx[28]
    ),
    waveform_options: (
      /*waveform_options*/
      ctx[19]
    ),
    trim_region_settings: (
      /*trim_region_settings*/
      ctx[30]
    ),
    stream_every: (
      /*stream_every*/
      ctx[22]
    ),
    upload: (
      /*func*/
      ctx[46]
    ),
    stream_handler: (
      /*func_1*/
      ctx[47]
    ),
    $$slots: { default: [create_default_slot_2] },
    $$scope: { ctx }
  };
  if (
    /*recording*/
    ctx[2] !== void 0
  ) {
    interactiveaudio_props.recording = /*recording*/
    ctx[2];
  }
  if (
    /*dragging*/
    ctx[27] !== void 0
  ) {
    interactiveaudio_props.dragging = /*dragging*/
    ctx[27];
  }
  if (
    /*uploading*/
    ctx[24] !== void 0
  ) {
    interactiveaudio_props.uploading = /*uploading*/
    ctx[24];
  }
  if (
    /*_modify_stream*/
    ctx[26] !== void 0
  ) {
    interactiveaudio_props.modify_stream = /*_modify_stream*/
    ctx[26];
  }
  if (
    /*set_time_limit*/
    ctx[3] !== void 0
  ) {
    interactiveaudio_props.set_time_limit = /*set_time_limit*/
    ctx[3];
  }
  interactiveaudio = new InteractiveAudio({ props: interactiveaudio_props });
  binding_callbacks.push(() => bind(interactiveaudio, "recording", interactiveaudio_recording_binding));
  binding_callbacks.push(() => bind(interactiveaudio, "dragging", interactiveaudio_dragging_binding));
  binding_callbacks.push(() => bind(interactiveaudio, "uploading", interactiveaudio_uploading_binding));
  binding_callbacks.push(() => bind(interactiveaudio, "modify_stream", interactiveaudio_modify_stream_binding));
  binding_callbacks.push(() => bind(interactiveaudio, "set_time_limit", interactiveaudio_set_time_limit_binding));
  interactiveaudio.$on(
    "change",
    /*change_handler*/
    ctx[53]
  );
  interactiveaudio.$on(
    "stream",
    /*stream_handler*/
    ctx[54]
  );
  interactiveaudio.$on(
    "drag",
    /*drag_handler*/
    ctx[55]
  );
  interactiveaudio.$on(
    "edit",
    /*edit_handler*/
    ctx[56]
  );
  interactiveaudio.$on(
    "play",
    /*play_handler_1*/
    ctx[57]
  );
  interactiveaudio.$on(
    "pause",
    /*pause_handler_1*/
    ctx[58]
  );
  interactiveaudio.$on(
    "stop",
    /*stop_handler_1*/
    ctx[59]
  );
  interactiveaudio.$on(
    "start_recording",
    /*start_recording_handler*/
    ctx[60]
  );
  interactiveaudio.$on(
    "pause_recording",
    /*pause_recording_handler*/
    ctx[61]
  );
  interactiveaudio.$on(
    "stop_recording",
    /*stop_recording_handler*/
    ctx[62]
  );
  interactiveaudio.$on(
    "upload",
    /*upload_handler*/
    ctx[63]
  );
  interactiveaudio.$on(
    "clear",
    /*clear_handler*/
    ctx[64]
  );
  interactiveaudio.$on(
    "error",
    /*handle_error*/
    ctx[31]
  );
  interactiveaudio.$on(
    "close_stream",
    /*close_stream_handler*/
    ctx[65]
  );
  return {
    c() {
      create_component(statustracker.$$.fragment);
      t = space();
      create_component(interactiveaudio.$$.fragment);
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
      t = claim_space(nodes);
      claim_component(interactiveaudio.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t, anchor);
      mount_component(interactiveaudio, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty[0] & /*gradio, loading_status*/
      8388610 ? get_spread_update(statustracker_spread_levels, [
        dirty[0] & /*gradio*/
        8388608 && {
          autoscroll: (
            /*gradio*/
            ctx2[23].autoscroll
          )
        },
        dirty[0] & /*gradio*/
        8388608 && { i18n: (
          /*gradio*/
          ctx2[23].i18n
        ) },
        dirty[0] & /*loading_status*/
        2 && get_spread_object(
          /*loading_status*/
          ctx2[1]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
      const interactiveaudio_changes = {};
      if (dirty[0] & /*label*/
      512)
        interactiveaudio_changes.label = /*label*/
        ctx2[9];
      if (dirty[0] & /*show_label*/
      2048)
        interactiveaudio_changes.show_label = /*show_label*/
        ctx2[11];
      if (dirty[0] & /*show_download_button*/
      65536)
        interactiveaudio_changes.show_download_button = /*show_download_button*/
        ctx2[16];
      if (dirty[0] & /*value*/
      1)
        interactiveaudio_changes.value = /*value*/
        ctx2[0];
      if (dirty[0] & /*root*/
      1024)
        interactiveaudio_changes.root = /*root*/
        ctx2[10];
      if (dirty[0] & /*sources*/
      256)
        interactiveaudio_changes.sources = /*sources*/
        ctx2[8];
      if (dirty[0] & /*active_source*/
      33554432)
        interactiveaudio_changes.active_source = /*active_source*/
        ctx2[25];
      if (dirty[0] & /*pending*/
      1048576)
        interactiveaudio_changes.pending = /*pending*/
        ctx2[20];
      if (dirty[0] & /*streaming*/
      2097152)
        interactiveaudio_changes.streaming = /*streaming*/
        ctx2[21];
      if (dirty[0] & /*loop*/
      32768)
        interactiveaudio_changes.loop = /*loop*/
        ctx2[15];
      if (dirty[0] & /*gradio*/
      8388608)
        interactiveaudio_changes.max_file_size = /*gradio*/
        ctx2[23].max_file_size;
      if (dirty[0] & /*editable*/
      262144)
        interactiveaudio_changes.editable = /*editable*/
        ctx2[18];
      if (dirty[0] & /*gradio*/
      8388608)
        interactiveaudio_changes.i18n = /*gradio*/
        ctx2[23].i18n;
      if (dirty[0] & /*waveform_settings*/
      268435456)
        interactiveaudio_changes.waveform_settings = /*waveform_settings*/
        ctx2[28];
      if (dirty[0] & /*waveform_options*/
      524288)
        interactiveaudio_changes.waveform_options = /*waveform_options*/
        ctx2[19];
      if (dirty[0] & /*stream_every*/
      4194304)
        interactiveaudio_changes.stream_every = /*stream_every*/
        ctx2[22];
      if (dirty[0] & /*gradio*/
      8388608)
        interactiveaudio_changes.upload = /*func*/
        ctx2[46];
      if (dirty[0] & /*gradio*/
      8388608)
        interactiveaudio_changes.stream_handler = /*func_1*/
        ctx2[47];
      if (dirty[0] & /*gradio*/
      8388608 | dirty[2] & /*$$scope*/
      128) {
        interactiveaudio_changes.$$scope = { dirty, ctx: ctx2 };
      }
      if (!updating_recording && dirty[0] & /*recording*/
      4) {
        updating_recording = true;
        interactiveaudio_changes.recording = /*recording*/
        ctx2[2];
        add_flush_callback(() => updating_recording = false);
      }
      if (!updating_dragging && dirty[0] & /*dragging*/
      134217728) {
        updating_dragging = true;
        interactiveaudio_changes.dragging = /*dragging*/
        ctx2[27];
        add_flush_callback(() => updating_dragging = false);
      }
      if (!updating_uploading && dirty[0] & /*uploading*/
      16777216) {
        updating_uploading = true;
        interactiveaudio_changes.uploading = /*uploading*/
        ctx2[24];
        add_flush_callback(() => updating_uploading = false);
      }
      if (!updating_modify_stream && dirty[0] & /*_modify_stream*/
      67108864) {
        updating_modify_stream = true;
        interactiveaudio_changes.modify_stream = /*_modify_stream*/
        ctx2[26];
        add_flush_callback(() => updating_modify_stream = false);
      }
      if (!updating_set_time_limit && dirty[0] & /*set_time_limit*/
      8) {
        updating_set_time_limit = true;
        interactiveaudio_changes.set_time_limit = /*set_time_limit*/
        ctx2[3];
        add_flush_callback(() => updating_set_time_limit = false);
      }
      interactiveaudio.$set(interactiveaudio_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      transition_in(interactiveaudio.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      transition_out(interactiveaudio.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
      destroy_component(statustracker, detaching);
      destroy_component(interactiveaudio, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let statustracker;
  let t;
  let staticaudio;
  let current;
  const statustracker_spread_levels = [
    {
      autoscroll: (
        /*gradio*/
        ctx[23].autoscroll
      )
    },
    { i18n: (
      /*gradio*/
      ctx[23].i18n
    ) },
    /*loading_status*/
    ctx[1]
  ];
  let statustracker_props = {};
  for (let i = 0; i < statustracker_spread_levels.length; i += 1) {
    statustracker_props = assign(statustracker_props, statustracker_spread_levels[i]);
  }
  statustracker = new Static({ props: statustracker_props });
  statustracker.$on(
    "clear_status",
    /*clear_status_handler*/
    ctx[39]
  );
  staticaudio = new StaticAudio({
    props: {
      i18n: (
        /*gradio*/
        ctx[23].i18n
      ),
      show_label: (
        /*show_label*/
        ctx[11]
      ),
      show_download_button: (
        /*show_download_button*/
        ctx[16]
      ),
      show_share_button: (
        /*show_share_button*/
        ctx[17]
      ),
      value: (
        /*value*/
        ctx[0]
      ),
      label: (
        /*label*/
        ctx[9]
      ),
      loop: (
        /*loop*/
        ctx[15]
      ),
      waveform_settings: (
        /*waveform_settings*/
        ctx[28]
      ),
      waveform_options: (
        /*waveform_options*/
        ctx[19]
      ),
      editable: (
        /*editable*/
        ctx[18]
      )
    }
  });
  staticaudio.$on(
    "share",
    /*share_handler*/
    ctx[40]
  );
  staticaudio.$on(
    "error",
    /*error_handler*/
    ctx[41]
  );
  staticaudio.$on(
    "play",
    /*play_handler*/
    ctx[42]
  );
  staticaudio.$on(
    "pause",
    /*pause_handler*/
    ctx[43]
  );
  staticaudio.$on(
    "stop",
    /*stop_handler*/
    ctx[44]
  );
  return {
    c() {
      create_component(statustracker.$$.fragment);
      t = space();
      create_component(staticaudio.$$.fragment);
    },
    l(nodes) {
      claim_component(statustracker.$$.fragment, nodes);
      t = claim_space(nodes);
      claim_component(staticaudio.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(statustracker, target, anchor);
      insert_hydration(target, t, anchor);
      mount_component(staticaudio, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const statustracker_changes = dirty[0] & /*gradio, loading_status*/
      8388610 ? get_spread_update(statustracker_spread_levels, [
        dirty[0] & /*gradio*/
        8388608 && {
          autoscroll: (
            /*gradio*/
            ctx2[23].autoscroll
          )
        },
        dirty[0] & /*gradio*/
        8388608 && { i18n: (
          /*gradio*/
          ctx2[23].i18n
        ) },
        dirty[0] & /*loading_status*/
        2 && get_spread_object(
          /*loading_status*/
          ctx2[1]
        )
      ]) : {};
      statustracker.$set(statustracker_changes);
      const staticaudio_changes = {};
      if (dirty[0] & /*gradio*/
      8388608)
        staticaudio_changes.i18n = /*gradio*/
        ctx2[23].i18n;
      if (dirty[0] & /*show_label*/
      2048)
        staticaudio_changes.show_label = /*show_label*/
        ctx2[11];
      if (dirty[0] & /*show_download_button*/
      65536)
        staticaudio_changes.show_download_button = /*show_download_button*/
        ctx2[16];
      if (dirty[0] & /*show_share_button*/
      131072)
        staticaudio_changes.show_share_button = /*show_share_button*/
        ctx2[17];
      if (dirty[0] & /*value*/
      1)
        staticaudio_changes.value = /*value*/
        ctx2[0];
      if (dirty[0] & /*label*/
      512)
        staticaudio_changes.label = /*label*/
        ctx2[9];
      if (dirty[0] & /*loop*/
      32768)
        staticaudio_changes.loop = /*loop*/
        ctx2[15];
      if (dirty[0] & /*waveform_settings*/
      268435456)
        staticaudio_changes.waveform_settings = /*waveform_settings*/
        ctx2[28];
      if (dirty[0] & /*waveform_options*/
      524288)
        staticaudio_changes.waveform_options = /*waveform_options*/
        ctx2[19];
      if (dirty[0] & /*editable*/
      262144)
        staticaudio_changes.editable = /*editable*/
        ctx2[18];
      staticaudio.$set(staticaudio_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(statustracker.$$.fragment, local);
      transition_in(staticaudio.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(statustracker.$$.fragment, local);
      transition_out(staticaudio.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
      destroy_component(statustracker, detaching);
      destroy_component(staticaudio, detaching);
    }
  };
}
function create_fragment(ctx) {
  let current_block_type_index;
  let if_block;
  let if_block_anchor;
  let current;
  const if_block_creators = [create_if_block, create_else_block];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (!/*interactive*/
    ctx2[7])
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      if_block.c();
      if_block_anchor = empty();
    },
    l(nodes) {
      if_block.l(nodes);
      if_block_anchor = empty();
    },
    m(target, anchor) {
      if_blocks[current_block_type_index].m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      let previous_block_index = current_block_type_index;
      current_block_type_index = select_block_type(ctx2);
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
        if_block.m(if_block_anchor.parentNode, if_block_anchor);
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
        detach(if_block_anchor);
      }
      if_blocks[current_block_type_index].d(detaching);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
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
    $$invalidate(0, value = initial_value);
  };
  let dragging;
  let waveform_settings;
  let color_accent = "darkorange";
  onMount(() => {
    color_accent = getComputedStyle(document == null ? void 0 : document.documentElement).getPropertyValue("--color-accent");
    set_trim_region_colour();
    $$invalidate(28, waveform_settings.waveColor = waveform_options.waveform_color || "#9ca3af", waveform_settings);
    $$invalidate(28, waveform_settings.progressColor = waveform_options.waveform_progress_color || color_accent, waveform_settings);
    $$invalidate(28, waveform_settings.mediaControls = waveform_options.show_controls, waveform_settings);
    $$invalidate(28, waveform_settings.sampleRate = waveform_options.sample_rate || 44100, waveform_settings);
  });
  const trim_region_settings = {
    color: waveform_options.trim_region_color,
    drag: true,
    resize: true
  };
  function set_trim_region_colour() {
    document.documentElement.style.setProperty("--trim-region-color", trim_region_settings.color || color_accent);
  }
  function handle_error({ detail }) {
    const [level, status] = detail.includes("Invalid file type") ? ["warning", "complete"] : ["error", "error"];
    $$invalidate(1, loading_status = loading_status || {});
    $$invalidate(1, loading_status.status = status, loading_status);
    $$invalidate(1, loading_status.message = detail, loading_status);
    gradio.dispatch(level, detail);
  }
  afterUpdate(() => {
    $$invalidate(32, value_is_output = false);
  });
  const clear_status_handler = () => gradio.dispatch("clear_status", loading_status);
  const share_handler = (e) => gradio.dispatch("share", e.detail);
  const error_handler = (e) => gradio.dispatch("error", e.detail);
  const play_handler = () => gradio.dispatch("play");
  const pause_handler = () => gradio.dispatch("pause");
  const stop_handler = () => gradio.dispatch("stop");
  const clear_status_handler_1 = () => gradio.dispatch("clear_status", loading_status);
  const func = (...args) => gradio.client.upload(...args);
  const func_1 = (...args) => gradio.client.stream(...args);
  function interactiveaudio_recording_binding(value2) {
    recording = value2;
    $$invalidate(2, recording);
  }
  function interactiveaudio_dragging_binding(value2) {
    dragging = value2;
    $$invalidate(27, dragging);
  }
  function interactiveaudio_uploading_binding(value2) {
    uploading = value2;
    $$invalidate(24, uploading);
  }
  function interactiveaudio_modify_stream_binding(value2) {
    _modify_stream = value2;
    $$invalidate(26, _modify_stream);
  }
  function interactiveaudio_set_time_limit_binding(value2) {
    set_time_limit = value2;
    $$invalidate(3, set_time_limit);
  }
  const change_handler = ({ detail }) => $$invalidate(0, value = detail);
  const stream_handler = ({ detail }) => {
    $$invalidate(0, value = detail);
    gradio.dispatch("stream", value);
  };
  const drag_handler = ({ detail }) => $$invalidate(27, dragging = detail);
  const edit_handler = () => gradio.dispatch("edit");
  const play_handler_1 = () => gradio.dispatch("play");
  const pause_handler_1 = () => gradio.dispatch("pause");
  const stop_handler_1 = () => gradio.dispatch("stop");
  const start_recording_handler = () => gradio.dispatch("start_recording");
  const pause_recording_handler = () => gradio.dispatch("pause_recording");
  const stop_recording_handler = (e) => gradio.dispatch("stop_recording");
  const upload_handler = () => gradio.dispatch("upload");
  const clear_handler = () => gradio.dispatch("clear");
  const close_stream_handler = () => gradio.dispatch("close_stream", "stream");
  $$self.$$set = ($$props2) => {
    if ("value_is_output" in $$props2)
      $$invalidate(32, value_is_output = $$props2.value_is_output);
    if ("elem_id" in $$props2)
      $$invalidate(4, elem_id = $$props2.elem_id);
    if ("elem_classes" in $$props2)
      $$invalidate(5, elem_classes = $$props2.elem_classes);
    if ("visible" in $$props2)
      $$invalidate(6, visible = $$props2.visible);
    if ("interactive" in $$props2)
      $$invalidate(7, interactive = $$props2.interactive);
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("sources" in $$props2)
      $$invalidate(8, sources = $$props2.sources);
    if ("label" in $$props2)
      $$invalidate(9, label = $$props2.label);
    if ("root" in $$props2)
      $$invalidate(10, root = $$props2.root);
    if ("show_label" in $$props2)
      $$invalidate(11, show_label = $$props2.show_label);
    if ("container" in $$props2)
      $$invalidate(12, container = $$props2.container);
    if ("scale" in $$props2)
      $$invalidate(13, scale = $$props2.scale);
    if ("min_width" in $$props2)
      $$invalidate(14, min_width = $$props2.min_width);
    if ("loading_status" in $$props2)
      $$invalidate(1, loading_status = $$props2.loading_status);
    if ("autoplay" in $$props2)
      $$invalidate(34, autoplay = $$props2.autoplay);
    if ("loop" in $$props2)
      $$invalidate(15, loop = $$props2.loop);
    if ("show_download_button" in $$props2)
      $$invalidate(16, show_download_button = $$props2.show_download_button);
    if ("show_share_button" in $$props2)
      $$invalidate(17, show_share_button = $$props2.show_share_button);
    if ("editable" in $$props2)
      $$invalidate(18, editable = $$props2.editable);
    if ("waveform_options" in $$props2)
      $$invalidate(19, waveform_options = $$props2.waveform_options);
    if ("pending" in $$props2)
      $$invalidate(20, pending = $$props2.pending);
    if ("streaming" in $$props2)
      $$invalidate(21, streaming = $$props2.streaming);
    if ("stream_every" in $$props2)
      $$invalidate(22, stream_every = $$props2.stream_every);
    if ("input_ready" in $$props2)
      $$invalidate(33, input_ready = $$props2.input_ready);
    if ("recording" in $$props2)
      $$invalidate(2, recording = $$props2.recording);
    if ("set_time_limit" in $$props2)
      $$invalidate(3, set_time_limit = $$props2.set_time_limit);
    if ("gradio" in $$props2)
      $$invalidate(23, gradio = $$props2.gradio);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty[0] & /*uploading*/
    16777216) {
      $$invalidate(33, input_ready = !uploading);
    }
    if ($$self.$$.dirty[0] & /*value*/
    1 | $$self.$$.dirty[1] & /*initial_value*/
    128) {
      if (value && initial_value === null) {
        $$invalidate(38, initial_value = value);
      }
    }
    if ($$self.$$.dirty[0] & /*value, gradio*/
    8388609 | $$self.$$.dirty[1] & /*old_value, value_is_output*/
    66) {
      {
        if (JSON.stringify(value) !== JSON.stringify(old_value)) {
          $$invalidate(37, old_value = value);
          gradio.dispatch("change");
          if (!value_is_output) {
            gradio.dispatch("input");
          }
        }
      }
    }
    if ($$self.$$.dirty[0] & /*active_source, sources*/
    33554688) {
      if (!active_source && sources) {
        $$invalidate(25, active_source = sources[0]);
      }
    }
    if ($$self.$$.dirty[1] & /*autoplay*/
    8) {
      $$invalidate(28, waveform_settings = {
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
      });
    }
  };
  return [
    value,
    loading_status,
    recording,
    set_time_limit,
    elem_id,
    elem_classes,
    visible,
    interactive,
    sources,
    label,
    root,
    show_label,
    container,
    scale,
    min_width,
    loop,
    show_download_button,
    show_share_button,
    editable,
    waveform_options,
    pending,
    streaming,
    stream_every,
    gradio,
    uploading,
    active_source,
    _modify_stream,
    dragging,
    waveform_settings,
    handle_reset_value,
    trim_region_settings,
    handle_error,
    value_is_output,
    input_ready,
    autoplay,
    modify_stream_state,
    get_stream_state,
    old_value,
    initial_value,
    clear_status_handler,
    share_handler,
    error_handler,
    play_handler,
    pause_handler,
    stop_handler,
    clear_status_handler_1,
    func,
    func_1,
    interactiveaudio_recording_binding,
    interactiveaudio_dragging_binding,
    interactiveaudio_uploading_binding,
    interactiveaudio_modify_stream_binding,
    interactiveaudio_set_time_limit_binding,
    change_handler,
    stream_handler,
    drag_handler,
    edit_handler,
    play_handler_1,
    pause_handler_1,
    stop_handler_1,
    start_recording_handler,
    pause_recording_handler,
    stop_recording_handler,
    upload_handler,
    clear_handler,
    close_stream_handler
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
        value_is_output: 32,
        elem_id: 4,
        elem_classes: 5,
        visible: 6,
        interactive: 7,
        value: 0,
        sources: 8,
        label: 9,
        root: 10,
        show_label: 11,
        container: 12,
        scale: 13,
        min_width: 14,
        loading_status: 1,
        autoplay: 34,
        loop: 15,
        show_download_button: 16,
        show_share_button: 17,
        editable: 18,
        waveform_options: 19,
        pending: 20,
        streaming: 21,
        stream_every: 22,
        input_ready: 33,
        recording: 2,
        modify_stream_state: 35,
        get_stream_state: 36,
        set_time_limit: 3,
        gradio: 23
      },
      null,
      [-1, -1, -1]
    );
  }
  get value_is_output() {
    return this.$$.ctx[32];
  }
  set value_is_output(value_is_output) {
    this.$$set({ value_is_output });
    flush();
  }
  get elem_id() {
    return this.$$.ctx[4];
  }
  set elem_id(elem_id) {
    this.$$set({ elem_id });
    flush();
  }
  get elem_classes() {
    return this.$$.ctx[5];
  }
  set elem_classes(elem_classes) {
    this.$$set({ elem_classes });
    flush();
  }
  get visible() {
    return this.$$.ctx[6];
  }
  set visible(visible) {
    this.$$set({ visible });
    flush();
  }
  get interactive() {
    return this.$$.ctx[7];
  }
  set interactive(interactive) {
    this.$$set({ interactive });
    flush();
  }
  get value() {
    return this.$$.ctx[0];
  }
  set value(value) {
    this.$$set({ value });
    flush();
  }
  get sources() {
    return this.$$.ctx[8];
  }
  set sources(sources) {
    this.$$set({ sources });
    flush();
  }
  get label() {
    return this.$$.ctx[9];
  }
  set label(label) {
    this.$$set({ label });
    flush();
  }
  get root() {
    return this.$$.ctx[10];
  }
  set root(root) {
    this.$$set({ root });
    flush();
  }
  get show_label() {
    return this.$$.ctx[11];
  }
  set show_label(show_label) {
    this.$$set({ show_label });
    flush();
  }
  get container() {
    return this.$$.ctx[12];
  }
  set container(container) {
    this.$$set({ container });
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
  get loading_status() {
    return this.$$.ctx[1];
  }
  set loading_status(loading_status) {
    this.$$set({ loading_status });
    flush();
  }
  get autoplay() {
    return this.$$.ctx[34];
  }
  set autoplay(autoplay) {
    this.$$set({ autoplay });
    flush();
  }
  get loop() {
    return this.$$.ctx[15];
  }
  set loop(loop) {
    this.$$set({ loop });
    flush();
  }
  get show_download_button() {
    return this.$$.ctx[16];
  }
  set show_download_button(show_download_button) {
    this.$$set({ show_download_button });
    flush();
  }
  get show_share_button() {
    return this.$$.ctx[17];
  }
  set show_share_button(show_share_button) {
    this.$$set({ show_share_button });
    flush();
  }
  get editable() {
    return this.$$.ctx[18];
  }
  set editable(editable) {
    this.$$set({ editable });
    flush();
  }
  get waveform_options() {
    return this.$$.ctx[19];
  }
  set waveform_options(waveform_options) {
    this.$$set({ waveform_options });
    flush();
  }
  get pending() {
    return this.$$.ctx[20];
  }
  set pending(pending) {
    this.$$set({ pending });
    flush();
  }
  get streaming() {
    return this.$$.ctx[21];
  }
  set streaming(streaming) {
    this.$$set({ streaming });
    flush();
  }
  get stream_every() {
    return this.$$.ctx[22];
  }
  set stream_every(stream_every) {
    this.$$set({ stream_every });
    flush();
  }
  get input_ready() {
    return this.$$.ctx[33];
  }
  set input_ready(input_ready) {
    this.$$set({ input_ready });
    flush();
  }
  get recording() {
    return this.$$.ctx[2];
  }
  set recording(recording) {
    this.$$set({ recording });
    flush();
  }
  get modify_stream_state() {
    return this.$$.ctx[35];
  }
  get get_stream_state() {
    return this.$$.ctx[36];
  }
  get set_time_limit() {
    return this.$$.ctx[3];
  }
  set set_time_limit(set_time_limit) {
    this.$$set({ set_time_limit });
    flush();
  }
  get gradio() {
    return this.$$.ctx[23];
  }
  set gradio(gradio) {
    this.$$set({ gradio });
    flush();
  }
}
const Index$1 = Index;
export {
  default2 as BaseExample,
  InteractiveAudio as BaseInteractiveAudio,
  A as BasePlayer,
  StaticAudio as BaseStaticAudio,
  Index$1 as default
};
