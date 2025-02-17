import { SvelteComponent, init, safe_not_equal, create_component, space, empty, claim_component, claim_space, mount_component, insert_hydration, group_outros, transition_out, check_outros, transition_in, detach, destroy_component, createEventDispatcher, bubble } from "../../../svelte/svelte.js";
import "../../../svelte/svelte-submodules.js";
import { u as uploadToHuggingFace, I as IconButton } from "./2.8WKXZUMv.js";
import { B as BlockLabel } from "./BlockLabel.BcwqJKEo.js";
import { E as Empty } from "./Empty.DwcIP_BA.js";
import { S as ShareButton } from "./ShareButton.BcDJqKEx.js";
import { D as Download } from "./Download.BLM_J5wv.js";
import { M as Music } from "./Music.BKn1BNLT.js";
import { I as IconButtonWrapper } from "./IconButtonWrapper.CePV_r65.js";
import { A as AudioPlayer } from "./AudioPlayer.CKSIBYDR.js";
import { D as DownloadLink } from "./DownloadLink.CzZp0moC.js";
function create_else_block(ctx) {
  let empty_1;
  let current;
  empty_1 = new Empty({
    props: {
      size: "small",
      $$slots: { default: [create_default_slot_2] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      create_component(empty_1.$$.fragment);
    },
    l(nodes) {
      claim_component(empty_1.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(empty_1, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const empty_1_changes = {};
      if (dirty & /*$$scope*/
      524288) {
        empty_1_changes.$$scope = { dirty, ctx: ctx2 };
      }
      empty_1.$set(empty_1_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(empty_1.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(empty_1.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(empty_1, detaching);
    }
  };
}
function create_if_block(ctx) {
  let iconbuttonwrapper;
  let t;
  let audioplayer;
  let current;
  iconbuttonwrapper = new IconButtonWrapper({
    props: {
      display_top_corner: (
        /*display_icon_button_wrapper_top_corner*/
        ctx[10]
      ),
      $$slots: { default: [create_default_slot] },
      $$scope: { ctx }
    }
  });
  audioplayer = new AudioPlayer({
    props: {
      value: (
        /*value*/
        ctx[0]
      ),
      label: (
        /*label*/
        ctx[1]
      ),
      i18n: (
        /*i18n*/
        ctx[5]
      ),
      waveform_settings: (
        /*waveform_settings*/
        ctx[6]
      ),
      waveform_options: (
        /*waveform_options*/
        ctx[7]
      ),
      editable: (
        /*editable*/
        ctx[8]
      ),
      loop: (
        /*loop*/
        ctx[9]
      )
    }
  });
  audioplayer.$on(
    "pause",
    /*pause_handler*/
    ctx[14]
  );
  audioplayer.$on(
    "play",
    /*play_handler*/
    ctx[15]
  );
  audioplayer.$on(
    "stop",
    /*stop_handler*/
    ctx[16]
  );
  audioplayer.$on(
    "load",
    /*load_handler*/
    ctx[17]
  );
  return {
    c() {
      create_component(iconbuttonwrapper.$$.fragment);
      t = space();
      create_component(audioplayer.$$.fragment);
    },
    l(nodes) {
      claim_component(iconbuttonwrapper.$$.fragment, nodes);
      t = claim_space(nodes);
      claim_component(audioplayer.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(iconbuttonwrapper, target, anchor);
      insert_hydration(target, t, anchor);
      mount_component(audioplayer, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const iconbuttonwrapper_changes = {};
      if (dirty & /*display_icon_button_wrapper_top_corner*/
      1024)
        iconbuttonwrapper_changes.display_top_corner = /*display_icon_button_wrapper_top_corner*/
        ctx2[10];
      if (dirty & /*$$scope, i18n, value, show_share_button, show_download_button*/
      524345) {
        iconbuttonwrapper_changes.$$scope = { dirty, ctx: ctx2 };
      }
      iconbuttonwrapper.$set(iconbuttonwrapper_changes);
      const audioplayer_changes = {};
      if (dirty & /*value*/
      1)
        audioplayer_changes.value = /*value*/
        ctx2[0];
      if (dirty & /*label*/
      2)
        audioplayer_changes.label = /*label*/
        ctx2[1];
      if (dirty & /*i18n*/
      32)
        audioplayer_changes.i18n = /*i18n*/
        ctx2[5];
      if (dirty & /*waveform_settings*/
      64)
        audioplayer_changes.waveform_settings = /*waveform_settings*/
        ctx2[6];
      if (dirty & /*waveform_options*/
      128)
        audioplayer_changes.waveform_options = /*waveform_options*/
        ctx2[7];
      if (dirty & /*editable*/
      256)
        audioplayer_changes.editable = /*editable*/
        ctx2[8];
      if (dirty & /*loop*/
      512)
        audioplayer_changes.loop = /*loop*/
        ctx2[9];
      audioplayer.$set(audioplayer_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(iconbuttonwrapper.$$.fragment, local);
      transition_in(audioplayer.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(iconbuttonwrapper.$$.fragment, local);
      transition_out(audioplayer.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
      }
      destroy_component(iconbuttonwrapper, detaching);
      destroy_component(audioplayer, detaching);
    }
  };
}
function create_default_slot_2(ctx) {
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
function create_if_block_2(ctx) {
  var _a;
  let downloadlink;
  let current;
  downloadlink = new DownloadLink({
    props: {
      href: (
        /*value*/
        ctx[0].is_stream ? (
          /*value*/
          (_a = ctx[0].url) == null ? void 0 : _a.replace("playlist.m3u8", "playlist-file")
        ) : (
          /*value*/
          ctx[0].url
        )
      ),
      download: (
        /*value*/
        ctx[0].orig_name || /*value*/
        ctx[0].path
      ),
      $$slots: { default: [create_default_slot_1] },
      $$scope: { ctx }
    }
  });
  return {
    c() {
      create_component(downloadlink.$$.fragment);
    },
    l(nodes) {
      claim_component(downloadlink.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(downloadlink, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      var _a2;
      const downloadlink_changes = {};
      if (dirty & /*value*/
      1)
        downloadlink_changes.href = /*value*/
        ctx2[0].is_stream ? (
          /*value*/
          (_a2 = ctx2[0].url) == null ? void 0 : _a2.replace("playlist.m3u8", "playlist-file")
        ) : (
          /*value*/
          ctx2[0].url
        );
      if (dirty & /*value*/
      1)
        downloadlink_changes.download = /*value*/
        ctx2[0].orig_name || /*value*/
        ctx2[0].path;
      if (dirty & /*$$scope, i18n*/
      524320) {
        downloadlink_changes.$$scope = { dirty, ctx: ctx2 };
      }
      downloadlink.$set(downloadlink_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(downloadlink.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(downloadlink.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(downloadlink, detaching);
    }
  };
}
function create_default_slot_1(ctx) {
  let iconbutton;
  let current;
  iconbutton = new IconButton({
    props: {
      Icon: Download,
      label: (
        /*i18n*/
        ctx[5]("common.download")
      )
    }
  });
  return {
    c() {
      create_component(iconbutton.$$.fragment);
    },
    l(nodes) {
      claim_component(iconbutton.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(iconbutton, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const iconbutton_changes = {};
      if (dirty & /*i18n*/
      32)
        iconbutton_changes.label = /*i18n*/
        ctx2[5]("common.download");
      iconbutton.$set(iconbutton_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(iconbutton.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(iconbutton.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(iconbutton, detaching);
    }
  };
}
function create_if_block_1(ctx) {
  let sharebutton;
  let current;
  sharebutton = new ShareButton({
    props: {
      i18n: (
        /*i18n*/
        ctx[5]
      ),
      formatter: (
        /*func*/
        ctx[11]
      ),
      value: (
        /*value*/
        ctx[0]
      )
    }
  });
  sharebutton.$on(
    "error",
    /*error_handler*/
    ctx[12]
  );
  sharebutton.$on(
    "share",
    /*share_handler*/
    ctx[13]
  );
  return {
    c() {
      create_component(sharebutton.$$.fragment);
    },
    l(nodes) {
      claim_component(sharebutton.$$.fragment, nodes);
    },
    m(target, anchor) {
      mount_component(sharebutton, target, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      const sharebutton_changes = {};
      if (dirty & /*i18n*/
      32)
        sharebutton_changes.i18n = /*i18n*/
        ctx2[5];
      if (dirty & /*value*/
      1)
        sharebutton_changes.value = /*value*/
        ctx2[0];
      sharebutton.$set(sharebutton_changes);
    },
    i(local) {
      if (current)
        return;
      transition_in(sharebutton.$$.fragment, local);
      current = true;
    },
    o(local) {
      transition_out(sharebutton.$$.fragment, local);
      current = false;
    },
    d(detaching) {
      destroy_component(sharebutton, detaching);
    }
  };
}
function create_default_slot(ctx) {
  let t;
  let if_block1_anchor;
  let current;
  let if_block0 = (
    /*show_download_button*/
    ctx[3] && create_if_block_2(ctx)
  );
  let if_block1 = (
    /*show_share_button*/
    ctx[4] && create_if_block_1(ctx)
  );
  return {
    c() {
      if (if_block0)
        if_block0.c();
      t = space();
      if (if_block1)
        if_block1.c();
      if_block1_anchor = empty();
    },
    l(nodes) {
      if (if_block0)
        if_block0.l(nodes);
      t = claim_space(nodes);
      if (if_block1)
        if_block1.l(nodes);
      if_block1_anchor = empty();
    },
    m(target, anchor) {
      if (if_block0)
        if_block0.m(target, anchor);
      insert_hydration(target, t, anchor);
      if (if_block1)
        if_block1.m(target, anchor);
      insert_hydration(target, if_block1_anchor, anchor);
      current = true;
    },
    p(ctx2, dirty) {
      if (
        /*show_download_button*/
        ctx2[3]
      ) {
        if (if_block0) {
          if_block0.p(ctx2, dirty);
          if (dirty & /*show_download_button*/
          8) {
            transition_in(if_block0, 1);
          }
        } else {
          if_block0 = create_if_block_2(ctx2);
          if_block0.c();
          transition_in(if_block0, 1);
          if_block0.m(t.parentNode, t);
        }
      } else if (if_block0) {
        group_outros();
        transition_out(if_block0, 1, 1, () => {
          if_block0 = null;
        });
        check_outros();
      }
      if (
        /*show_share_button*/
        ctx2[4]
      ) {
        if (if_block1) {
          if_block1.p(ctx2, dirty);
          if (dirty & /*show_share_button*/
          16) {
            transition_in(if_block1, 1);
          }
        } else {
          if_block1 = create_if_block_1(ctx2);
          if_block1.c();
          transition_in(if_block1, 1);
          if_block1.m(if_block1_anchor.parentNode, if_block1_anchor);
        }
      } else if (if_block1) {
        group_outros();
        transition_out(if_block1, 1, 1, () => {
          if_block1 = null;
        });
        check_outros();
      }
    },
    i(local) {
      if (current)
        return;
      transition_in(if_block0);
      transition_in(if_block1);
      current = true;
    },
    o(local) {
      transition_out(if_block0);
      transition_out(if_block1);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
        detach(if_block1_anchor);
      }
      if (if_block0)
        if_block0.d(detaching);
      if (if_block1)
        if_block1.d(detaching);
    }
  };
}
function create_fragment(ctx) {
  let blocklabel;
  let t;
  let current_block_type_index;
  let if_block;
  let if_block_anchor;
  let current;
  blocklabel = new BlockLabel({
    props: {
      show_label: (
        /*show_label*/
        ctx[2]
      ),
      Icon: Music,
      float: false,
      label: (
        /*label*/
        ctx[1] || /*i18n*/
        ctx[5]("audio.audio")
      )
    }
  });
  const if_block_creators = [create_if_block, create_else_block];
  const if_blocks = [];
  function select_block_type(ctx2, dirty) {
    if (
      /*value*/
      ctx2[0] !== null
    )
      return 0;
    return 1;
  }
  current_block_type_index = select_block_type(ctx);
  if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
  return {
    c() {
      create_component(blocklabel.$$.fragment);
      t = space();
      if_block.c();
      if_block_anchor = empty();
    },
    l(nodes) {
      claim_component(blocklabel.$$.fragment, nodes);
      t = claim_space(nodes);
      if_block.l(nodes);
      if_block_anchor = empty();
    },
    m(target, anchor) {
      mount_component(blocklabel, target, anchor);
      insert_hydration(target, t, anchor);
      if_blocks[current_block_type_index].m(target, anchor);
      insert_hydration(target, if_block_anchor, anchor);
      current = true;
    },
    p(ctx2, [dirty]) {
      const blocklabel_changes = {};
      if (dirty & /*show_label*/
      4)
        blocklabel_changes.show_label = /*show_label*/
        ctx2[2];
      if (dirty & /*label, i18n*/
      34)
        blocklabel_changes.label = /*label*/
        ctx2[1] || /*i18n*/
        ctx2[5]("audio.audio");
      blocklabel.$set(blocklabel_changes);
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
      transition_in(blocklabel.$$.fragment, local);
      transition_in(if_block);
      current = true;
    },
    o(local) {
      transition_out(blocklabel.$$.fragment, local);
      transition_out(if_block);
      current = false;
    },
    d(detaching) {
      if (detaching) {
        detach(t);
        detach(if_block_anchor);
      }
      destroy_component(blocklabel, detaching);
      if_blocks[current_block_type_index].d(detaching);
    }
  };
}
function instance($$self, $$props, $$invalidate) {
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
  const func = async (value2) => {
    if (!value2)
      return "";
    let url = await uploadToHuggingFace(value2.url);
    return `<audio controls src="${url}"></audio>`;
  };
  function error_handler(event) {
    bubble.call(this, $$self, event);
  }
  function share_handler(event) {
    bubble.call(this, $$self, event);
  }
  function pause_handler(event) {
    bubble.call(this, $$self, event);
  }
  function play_handler(event) {
    bubble.call(this, $$self, event);
  }
  function stop_handler(event) {
    bubble.call(this, $$self, event);
  }
  function load_handler(event) {
    bubble.call(this, $$self, event);
  }
  $$self.$$set = ($$props2) => {
    if ("value" in $$props2)
      $$invalidate(0, value = $$props2.value);
    if ("label" in $$props2)
      $$invalidate(1, label = $$props2.label);
    if ("show_label" in $$props2)
      $$invalidate(2, show_label = $$props2.show_label);
    if ("show_download_button" in $$props2)
      $$invalidate(3, show_download_button = $$props2.show_download_button);
    if ("show_share_button" in $$props2)
      $$invalidate(4, show_share_button = $$props2.show_share_button);
    if ("i18n" in $$props2)
      $$invalidate(5, i18n = $$props2.i18n);
    if ("waveform_settings" in $$props2)
      $$invalidate(6, waveform_settings = $$props2.waveform_settings);
    if ("waveform_options" in $$props2)
      $$invalidate(7, waveform_options = $$props2.waveform_options);
    if ("editable" in $$props2)
      $$invalidate(8, editable = $$props2.editable);
    if ("loop" in $$props2)
      $$invalidate(9, loop = $$props2.loop);
    if ("display_icon_button_wrapper_top_corner" in $$props2)
      $$invalidate(10, display_icon_button_wrapper_top_corner = $$props2.display_icon_button_wrapper_top_corner);
  };
  $$self.$$.update = () => {
    if ($$self.$$.dirty & /*value*/
    1) {
      value && dispatch("change", value);
    }
  };
  return [
    value,
    label,
    show_label,
    show_download_button,
    show_share_button,
    i18n,
    waveform_settings,
    waveform_options,
    editable,
    loop,
    display_icon_button_wrapper_top_corner,
    func,
    error_handler,
    share_handler,
    pause_handler,
    play_handler,
    stop_handler,
    load_handler
  ];
}
class StaticAudio extends SvelteComponent {
  constructor(options) {
    super();
    init(this, options, instance, create_fragment, safe_not_equal, {
      value: 0,
      label: 1,
      show_label: 2,
      show_download_button: 3,
      show_share_button: 4,
      i18n: 5,
      waveform_settings: 6,
      waveform_options: 7,
      editable: 8,
      loop: 9,
      display_icon_button_wrapper_top_corner: 10
    });
  }
}
export {
  StaticAudio as default
};
